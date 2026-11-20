"""
Core components of FinRAG system.
"""
import importlib

__all__ = [
    "BaseEmbeddingModel",
    "BaseSummarizationModel", 
    "BaseQAModel",
    "BaseChunker",
    "RAPTORClustering",
    "RAPTORTree",
    "ClusterNode",
    "RAPTORRetriever"
]

def __getattr__(name: str):
    if name in ("BaseEmbeddingModel", "BaseSummarizationModel", "BaseQAModel", "BaseChunker"):
        mod = importlib.import_module(".base_models", __name__)
        return getattr(mod, name)
    elif name in ("RAPTORClustering",):
        mod = importlib.import_module(".clustering", __name__)
        return getattr(mod, name)
    elif name in ("RAPTORTree", "ClusterNode"):
        mod = importlib.import_module(".tree", __name__)
        return getattr(mod, name)
    elif name in ("RAPTORRetriever",):
        mod = importlib.import_module(".retrieval", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
