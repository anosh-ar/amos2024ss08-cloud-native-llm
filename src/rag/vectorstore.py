"""Shared helpers for the embedding model and the Chroma vector store."""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the (cached) local embedding model, running on CPU."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": config.EMBED_QUERY_PREFIX},
    )


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection on disk."""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
