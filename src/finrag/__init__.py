"""
FinRAG: Financial Retrieval-Augmented Generation System
Based on RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
"""
import importlib

__version__ = "1.0.0"

__all__ = [
    "FinRAG",
    "FinRAGConfig",
    "EnsembleScorer",
    "ScoringConfig",
    "ScoringResult",
    "PathwayVectorStore",
    "PathwayConfig",
    "trace_query",
    "trace_document_parsing",
    "langfuse_context",
    "flush_langfuse",
    "get_evaluator"
]

def __getattr__(name: str):
    if name in ("FinRAG",):
        mod = importlib.import_module(".finrag", __name__)
        return getattr(mod, name)
    elif name in ("FinRAGConfig",):
        mod = importlib.import_module(".config", __name__)
        return getattr(mod, name)
    elif name in ("EnsembleScorer", "ScoringConfig", "ScoringResult"):
        mod = importlib.import_module(".scoring", __name__)
        return getattr(mod, name)
    elif name in ("PathwayVectorStore", "PathwayConfig"):
        mod = importlib.import_module(".vectorstore", __name__)
        return getattr(mod, name)
    elif name in ("trace_query", "trace_document_parsing", "langfuse_context", "flush_langfuse"):
        mod = importlib.import_module(".observability", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def get_evaluator():
    from .evaluation import RAGASEvaluator, EvaluationConfig
    return RAGASEvaluator, EvaluationConfig
