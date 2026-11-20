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


STOCK_PROMPT = """You are an equity research analyst covering the Indian stock market (NSE & BSE).
Use the context below (live news, market fundamentals, and report excerpts) to answer the user's question.

Context:
{context}

Question:
{question}

Guidelines:
- Give a direct, structured answer with clear headings or markdown tables where comparing companies.
- Mention real numbers (P/E, market cap, order wins, margins) if present in context.
- Keep tone professional, objective, and easy to read.
- End with a quick note on key risks or things to watch out for.

Answer:"""

SECTOR_PROMPT = """You are a senior equity research analyst specializing in Indian markets (NSE & BSE).
You have been given live market data for multiple companies across different sectors.

Context (live fundamentals, news, and market data):
{context}

User's Question:
{question}

Instructions:
- Compare the sectors and companies using the actual live data provided above.
- Create a structured answer with clear sections/headings for each sector.
- Include a comparison table with key metrics (P/E, Market Cap, ROE, Revenue Growth) for the top stocks.
- Give a clear verdict on which sector/company looks stronger and WHY based on the numbers.
- Mention any recent news or events that affect the outlook.
- End with risks and things to watch.
- Be specific with numbers from the context, don't be vague.

Answer:"""


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
                        "sources": ["Financial Reports", "Market News", "Live Fundamentals"]
                    }
            except Exception as e:
                logger.warning(f"QA model {m} failed: {e}")
                continue

        return {
            "answer": "Could not complete analysis due to temporary API limits. Please retry in a few seconds.",
            "confidence": 0.0,
            "sources": []
        }
