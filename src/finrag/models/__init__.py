"""
Model implementations for FinRAG.
"""
import importlib

__all__ = [
    "OpenAIEmbeddingModel",
    "OpenAISummarizationModel",
    "OpenAIQAModel",
    "FinancialChunker",
    "SentenceTransformerEmbeddingModel",
    "FlanT5SummarizationModel",
    "FlanT5QAModel",
    "check_openai_key_valid",
    "GeminiSummarizationModel",
    "GeminiQAModel"
]

def __getattr__(name: str):
    if name in ("OpenAIEmbeddingModel", "OpenAISummarizationModel", "OpenAIQAModel", "FinancialChunker"):
        mod = importlib.import_module(".models", __name__)
        return getattr(mod, name)
    elif name in ("SentenceTransformerEmbeddingModel", "FlanT5SummarizationModel", "FlanT5QAModel", "check_openai_key_valid"):
        mod = importlib.import_module(".fallback_models", __name__)
        return getattr(mod, name)
    elif name in ("GeminiSummarizationModel", "GeminiQAModel"):
        mod = importlib.import_module(".gemini_models", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
