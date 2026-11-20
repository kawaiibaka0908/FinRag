import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.base_models import BaseSummarizationModel, BaseQAModel

logger = logging.getLogger(__name__)

MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

class GeminiSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gemini-3.5-flash-lite", api_key=None):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._setup()

    def _setup(self):
        if not self.api_key:
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to start Gemini client: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
    def summarize(self, texts, max_tokens=300):
        if not texts:
            return ""

        body = "\n\n".join(texts)[:20000]
        prompt = (
            "Summarize the key points from these financial excerpts. "
            "Highlight numbers, order books, revenue/profit, and guidance:\n\n"
            f"{body}"
        )

        if not self.client:
            return ". ".join(body.split(".")[:3]) + "."

        for m in [self.model_name] + [x for x in MODELS_TO_TRY if x != self.model_name]:
            try:
                res = self.client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    return res.text.strip()
            except Exception as e:
                logger.warning(f"Summarizer model {m} failed: {e}")
                continue

        return ". ".join(body.split(".")[:3]) + "."


STOCK_PROMPT = """You are a Principal Equity Research Analyst & Financial Advisor covering Indian Equities (NSE & BSE).
You have been provided with real-time exchange data, live market news, and corporate filing excerpts.

Context:
{context}

Client Inquiry:
{question}

Deliver an Institutional Equity Advisory Memorandum structured as follows:

### 1. Executive Advisory Summary & Stance
- Clear investment stance (e.g. Accumulate on Dips / Overweight / Hold) with investment horizon (12-24 months).
- Key quantitative takeaway comparing the target companies.

### 2. Live Market Fundamentals Comparison Table
- Markdown comparison table with exact numbers: Live Price (₹), Market Cap (Cr), P/E Ratio, ROE (%), Debt/Equity, Revenue Growth (%), 1Y Return (%).

### 3. Live News, Catalysts & Macro Impact Analysis
- Explain specifically how the recent news articles provided in the context affect the operational and financial outlook of these companies.
- Point out regulatory, order book, or sector trends that a static model would miss.

### 4. Company-by-Company Financial Health & Moat Assessment
- Detailed breakdown for each company using the context data (margins, capital efficiency, balance sheet strength).

### 5. Downside Risks & Monitoring Triggers
- Bulleted list of specific financial risks (valuation multiple compression, leverage, raw material headwinds).

Advisory Memo:"""

SECTOR_PROMPT = """You are a Senior Thematic Portfolio Strategist & Investment Advisor specializing in Indian Markets (NSE & BSE).
You have been provided with deterministic real-time market metrics, live breaking news feeds, and sectoral data across multiple peer baskets.

Context:
{context}

Client Inquiry:
{question}

Deliver an Institutional Thematic Research Memorandum structured as follows:

### 1. Sector Strategy & Clear Winner Verdict
- Direct, definitive verdict on which sector and specific company holds the superior risk-reward profile right now.
- Why the numbers (growth vs valuation) justify this call.

### 2. Multi-Sector Peer Comparison Matrix
- Markdown comparison table across the top representative stocks in each sector:
  Columns: Company | Ticker | Sector | Live Price | Market Cap | P/E | ROE | Revenue Growth | 1-Yr Return

### 3. Breaking News & Catalyst Impact Analysis
- Analyze the latest news headlines from the context (e.g., government subsidies, export deals, capital expenditure announcements).
- Detail how these live events directly impact earnings velocity and competitive moats.

### 4. Deep-Dive Sectoral Breakdown
- **Sector A (e.g., Space & Precision Tech):** Top company analysis, operational catalysts, margin trajectory.
- **Sector B (e.g., EV Mobility & Infrastructure):** Top company analysis, supply chain factors, policy dependencies.

### 5. Tactical Downside Risks & Valuation Alerts
- Specific risks (e.g., frothy P/E multiples, subsidy delays, interest rate sensitivity).

Advisory Memo:"""


class GeminiQAModel(BaseQAModel):
    def __init__(self, model="gemini-3.5-flash-lite", api_key=None):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._setup()

    def _setup(self):
        if not self.api_key:
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to start Gemini client: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
    def answer_question(self, context, question, query_type="stock"):
        if not context or len(context.strip()) < 20:
            return {
                "answer": "Not enough context found to answer this properly.",
                "confidence": 0.0,
                "sources": []
            }

        template = SECTOR_PROMPT if query_type == "sector" else STOCK_PROMPT
        prompt = template.format(context=context[:30000], question=question)

        if not self.client:
            return {
                "answer": f"Analysis based on context:\n\n{context[:500]}...",
                "confidence": 0.5,
                "sources": ["Local Context"]
            }

        for m in [self.model_name] + [x for x in MODELS_TO_TRY if x != self.model_name]:
            try:
                res = self.client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    return {
                        "answer": res.text.strip(),
                        "confidence": 0.95,
                        "sources": ["Corporate Filings RAG", "Live NSE/BSE Market Data", "Reuters & Financial News Feeds"]
                    }
            except Exception as e:
                logger.warning(f"QA model {m} failed: {e}")
                continue

        return {
            "answer": "Could not complete analysis due to temporary API limits. Please retry in a few seconds.",
            "confidence": 0.0,
            "sources": []
        }
