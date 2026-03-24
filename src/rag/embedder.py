"""
src/rag/embedder.py
────────────────────
STEP 2B — Embedding Generation

Uses sentence-transformers (all-MiniLM-L6-v2) to convert
text chunks into dense vector embeddings.

Model choice: all-MiniLM-L6-v2
  - Fast, lightweight (22MB)
  - 384-dimensional embeddings
  - Great for semantic similarity search
  - Free, runs fully locally — no API key needed

Output: embeddings returned as list of float lists
"""

import json
from pathlib import Path
from typing import Optional
from src.utils import load_config, get_logger

logger = get_logger("rag.embedder")

# Model name — change here if you want a different model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_model():
    """Load sentence-transformers model (downloads on first run ~22MB)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Run: pip install sentence-transformers"
        )

    logger.info(f"  Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("  (Downloads ~22MB on first run — cached after that)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("  Model loaded ✅")
    return model


def embed_chunks(
    chunks: list[dict],
    model=None,
    batch_size: int = 64,
) -> list[dict]:
    """
    Takes chunk dicts → adds 'embedding' key to each → returns enriched list.

    Args:
        chunks:     list of chunk dicts from chunker.py
        model:      pre-loaded SentenceTransformer (loads if None)
        batch_size: how many chunks to embed at once

    Returns:
        Same list with 'embedding': list[float] added to each chunk
    """
    if not chunks:
        logger.warning("No chunks to embed.")
        return []

    if model is None:
        model = load_model()

    logger.info("=" * 55)
    logger.info("  STEP 2B — GENERATING EMBEDDINGS")
    logger.info("=" * 55)
    logger.info(f"  Total chunks : {len(chunks):,}")
    logger.info(f"  Batch size   : {batch_size}")
    logger.info(f"  Model        : {EMBEDDING_MODEL}")
    logger.info(f"  Dimensions   : 384")

    texts = [c["text"] for c in chunks]

    # Batch encode with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity ready
    )

    # Attach embedding to each chunk
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    logger.info(f"  Embeddings generated: {len(embeddings):,}")
    logger.info(f"  Embedding shape     : ({len(embeddings)}, {len(embeddings[0])})")

    return chunks
