"""
Gemini model implementations using the official modern Google GenAI SDK.
"""
from typing import List, Dict, Any, Optional
import os
import logging
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.base_models import BaseSummarizationModel, BaseQAModel
from .fallback_models import SentenceTransformerEmbeddingModel

logger = logging.getLogger(__name__)


class GeminiSummarizationModel(BaseSummarizationModel):
    """Google Gemini summarization model for financial text."""

    def __init__(self, model: str = "gemini-3.8-flash", api_key: Optional[str] = None):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found for GeminiSummarizationModel")
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Initialized Google GenAI client for summarization (%s)", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize Google GenAI client: %s", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def summarize(self, texts: List[str], max_tokens: int = 300) -> str:
        """Summarize text chunks with deep Indian market financial awareness."""
        if not texts:
            return ""

        combined_text = "\n\n".join(texts)
        if len(combined_text) > 25000:
            combined_text = combined_text[:25000]

        prompt = f"""You are a senior equity research analyst specializing in the Indian stock market (NSE/BSE).
Summarize the following excerpts from annual reports / financial filings.
Preserve key financial facts, revenue/profit figures, debt numbers, capex plans, order book sizes, and strategic commentary:

{combined_text}

Provide a concise, highly factual executive summary:"""

        if not self.client:
            # Fallback to simple extractive summary
            sentences = combined_text.split(".")[:4]
            return ". ".join(s.strip() for s in sentences if s.strip()) + "."

        # Try models in priority order to handle temporary capacity spikes
        candidate_models = [self.model_name, "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        for m in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning("Gemini model %s encountered error: %s. Trying next candidate...", m, e)

        # Fallback
        return ". ".join(combined_text.split(".")[:3]) + "."


class GeminiQAModel(BaseQAModel):
    """Google Gemini QA model for complex financial analysis."""

    def __init__(self, model: str = "gemini-3.8-flash", api_key: Optional[str] = None):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found for GeminiQAModel")
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Initialized Google GenAI client for QA (%s)", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize Google GenAI client: %s", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """Answer financial question using retrieved context and live news."""
        if not context or not context.strip():
            return {
                "answer": "I do not have sufficient context to answer this question accurately.",
                "confidence": 0.0,
                "sources": [],
                "reasoning": "Empty context"
            }

        prompt = f"""You are an elite Indian market financial analyst and portfolio strategist (NSE & BSE expert).
Answer the following investment / financial question thoroughly, using the provided context (which includes Annual Report disclosures, fundamentals, and recent live news):

Context:
{context}

Question:
{question}

Instructions:
1. Provide a direct, data-backed answer citing numbers, percentages, order book values, or specific disclosures where available.
2. If comparing companies (e.g. HAL vs BDL, ICICI vs HDFC/SBI, Reliance vs Airtel), structure your comparison with clear headings and bullet points.
3. Incorporate both long-term fundamentals from annual reports and short-term impacts from recent news.
4. Conclude with an objective 'Strategic Outlook & Key Risks'.

Answer:"""

        if not self.client:
            return {
                "answer": f"Analysis based on context: {context[:400]}...",
                "confidence": 0.5,
                "sources": ["Local Context"],
                "reasoning": "Fallback mode"
            }

        candidate_models = [self.model_name, "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        for m in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=prompt,
                )
                if response and response.text:
                    return {
                        "answer": response.text.strip(),
                        "confidence": 0.95,
                        "sources": ["Annual Reports", "Live Market News"],
                        "model": m,
                    }
            except Exception as e:
                logger.warning("Gemini QA model %s encountered error: %s. Trying next candidate...", m, e)

        return {
            "answer": "Unable to generate answer due to temporary LLM API unavailability.",
            "confidence": 0.0,
            "sources": []
        }
