"""
src/rag/vectorstore.py
───────────────────────
STEP 2C — Vector Store (ChromaDB)

Stores all embedded chunks in a persistent ChromaDB collection.
ChromaDB stores data on disk — survives restarts, no server needed.

Storage location: data/vectorstore/  (created automatically)
Collection name : hotel_reviews
"""

import json
from pathlib import Path
from typing import Optional
from src.utils import load_config, get_logger

logger = get_logger("rag.vectorstore")

COLLECTION_NAME = "hotel_reviews"


def get_vectorstore_path() -> Path:
    cfg  = load_config()
    path = cfg["_project_root"] / "data" / "vectorstore"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_client():
    """Create/connect to persistent ChromaDB client."""
    try:
        import chromadb
    except ImportError:
        raise ImportError(
            "chromadb not installed.\n"
            "Run: pip install chromadb"
        )

    db_path = get_vectorstore_path()
    client  = chromadb.PersistentClient(path=str(db_path))
    return client


def get_or_create_collection(client, reset: bool = False):
    """
    Get existing collection or create a new one.
    reset=True will delete and recreate (use when re-indexing).
    """
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"  Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB metadata values must be str, int, float, or bool.
    Replace None with empty string or 0.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def store_chunks(
    chunks: list[dict],
    reset: bool = True,
    batch_size: int = 500,
) -> None:
    """
    Upsert all embedded chunks into ChromaDB.

    Args:
        chunks:     list of chunk dicts with 'embedding' key
        reset:      if True, wipe and rebuild the collection
        batch_size: how many chunks to upsert at once
    """
    logger.info("=" * 55)
    logger.info("  STEP 2C — STORING IN VECTOR DATABASE")
    logger.info("=" * 55)

    client     = get_client()
    collection = get_or_create_collection(client, reset=reset)
    total      = len(chunks)

    logger.info(f"  Collection   : {COLLECTION_NAME}")
    logger.info(f"  Total chunks : {total:,}")
    logger.info(f"  Batch size   : {batch_size}")
    logger.info(f"  Storage      : {get_vectorstore_path()}")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        ids        = [c["chunk_id"]  for c in batch]
        texts      = [c["text"]      for c in batch]
        embeddings = [c["embedding"] for c in batch]
        metadatas  = [_sanitize_metadata(c["metadata"]) for c in batch]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        pct = min(100, int((i + len(batch)) / total * 100))
        logger.info(f"  Stored batch {i//batch_size + 1}: {i + len(batch):,}/{total:,} chunks ({pct}%)")

    final_count = collection.count()
    logger.info(f"\n  ✅ ChromaDB collection ready")
    logger.info(f"  Collection count: {final_count:,} chunks indexed")


def load_collection():
    """Load existing ChromaDB collection for querying."""
    client     = get_client()
    collection = client.get_collection(COLLECTION_NAME)
    return collection
