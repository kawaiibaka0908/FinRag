"""Load the persistent Chroma index used by the annual report workflow."""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from ..config import FinRAGConfig
from ..models.fallback_models import SentenceTransformerEmbeddingModel
from ..models.models import OpenAIEmbeddingModel
from .chroma_store import ChromaConfig, ChromaVectorStore

logger = logging.getLogger(__name__)
ANNUAL_REPORT_TREE_PATH = Path(__file__).resolve().parents[3] / "finrag_tree"


def _embedding_model_for_index(model_id: str, dimension: Optional[int], config: FinRAGConfig):
    """Resolve only explicitly supported model identities and validate their dimensions."""
    if model_id == "sentence-transformers:all-MiniLM-L6-v2":
        if dimension != 384:
            raise ValueError(f"Index declares {model_id} but stores {dimension}-dimensional vectors")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True,
            )
        except Exception as exc:
            raise ValueError(
                "The Chroma index uses sentence-transformers:all-MiniLM-L6-v2, "
                "but that exact model is not fully available in the local Hugging Face cache. "
                "Run scripts/build_tree.py once with model download access."
            ) from exc
        return SentenceTransformerEmbeddingModel(
            "all-MiniLM-L6-v2", local_files_only=True
        )
    openai_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    if model_id.startswith("openai:") and model_id.split(":", 1)[1] in openai_dimensions:
        model_name = model_id.split(":", 1)[1]
        expected_dimension = openai_dimensions[model_name]
        if dimension != expected_dimension:
            raise ValueError(f"Index declares {model_id} but stores {dimension}-dimensional vectors")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(f"Index uses {model_id}; set OPENAI_API_KEY to query it")
        if config.embedding_model != model_name:
            raise ValueError(
                f"Index uses {model_id} but FINRAG_EMBEDDING_MODEL is "
                f"{config.embedding_model}; restore the indexed model configuration"
            )
        return OpenAIEmbeddingModel(model_name, api_key=api_key)
    raise ValueError(
        f"Unknown or unsupported annual report embedding model identity '{model_id}'. "
        "Rebuild the index with scripts/build_tree.py using a supported embedding model."
    )


def load_annual_report_store() -> Tuple[Optional[ChromaVectorStore], Optional[str]]:
    """Load the RAPTOR build script's persisted Chroma collection without indexing PDFs."""
    persist_directory = ANNUAL_REPORT_TREE_PATH / "chroma_vectorstore"
    config = FinRAGConfig()
    if not persist_directory.is_dir():
        return None, (
            f"Annual report Chroma index not found at {persist_directory}. "
            "Add reports to data/ and run `python scripts/build_tree.py` to index them."
        )

    try:
        store = ChromaVectorStore(ChromaConfig(
            collection_name=config.chroma_collection_name,
            persist_directory=str(persist_directory),
        ))
        count = store.collection.count()
        if count == 0:
            return None, (
                f"Annual report index at {persist_directory} is empty. "
                "Run `python scripts/build_tree.py` to index the reports in data/."
            )

        metadata = store.collection.metadata or {}
        model_id = metadata.get("embedding_model_id")
        if not model_id:
            return None, (
                "Annual report index has no recorded embedding model identity. "
                "Its query vectors cannot be validated; rebuild it with `python scripts/build_tree.py`."
            )
        sample = store.collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) else None
        store.embedding_model = _embedding_model_for_index(model_id, dimension, config)

        logger.info("Loaded %s annual report vectors from %s", count, persist_directory)
        return store, None
    except Exception as exc:
        logger.exception("Failed to initialize annual report ChromaDB index")
        return None, f"Could not initialize annual report ChromaDB: {exc}"
