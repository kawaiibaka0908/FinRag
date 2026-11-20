import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.base_models import BaseSummarizationModel, BaseQAModel

logger = logging.getLogger(__name__)

class GeminiSummarizationModel(BaseSummarizationModel):
    def __init__(self, model: str = "gemini-3.8-flash", api_key: str = None):
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
    def summarize(self, texts, max_tokens: int = 300) -> str:
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

        for m in [self.model_name, "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
            try:
                res = self.client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    return res.text.strip()
            except Exception:
                continue

        return ". ".join(body.split(".")[:3]) + "."


class GeminiQAModel(BaseQAModel):
    def __init__(self, model: str = "gemini-3.8-flash", api_key: str = None):
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
    def answer_question(self, context: str, question: str):
        if not context or not context.strip():
            return {
                "answer": "Not enough context found to answer this properly.",
                "confidence": 0.0,
                "sources": []
            }

        prompt = f"""You are an equity analyst covering the Indian stock market (NSE & BSE).
Use the context below (live news, market fundamentals, and report excerpts) to answer the question.

Context:
{context}

Question:
{question}

Guidelines:
- Give a direct, structured answer with clear headings or tables where comparing companies.
- Mention real numbers (P/E, market cap, order wins, margins) if present in context.
- Keep tone professional, objective, and easy to read.
- End with a quick note on key risks or things to watch out for.

Answer:"""

        if not self.client:
            return {
                "answer": f"Analysis based on context:\n\n{context[:500]}...",
                "confidence": 0.5,
                "sources": ["Local Context"]
            }

        for m in [self.model_name, "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
            try:
                res = self.client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    return {
                        "answer": res.text.strip(),
                        "confidence": 0.95,
                        "sources": ["Financial Reports", "Market News"]
                    }
            except Exception:
                continue

        return {
            "answer": "Could not complete analysis due to temporary API limits. Please retry in a few seconds.",
            "confidence": 0.0,
            "sources": []
        }
