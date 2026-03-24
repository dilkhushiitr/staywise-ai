"""
src/rag/rag_runner.py
──────────────────────
Step 2 Orchestrator — runs all 4 RAG sub-steps:
    2A  Chunk unified_hotels.csv
    2B  Generate embeddings
    2C  Store in ChromaDB
    2D  Verify retrieval works with a test query
"""

import sys
from src.utils        import get_logger
from .chunker         import run_chunking
from .embedder        import load_model, embed_chunks
from .vectorstore     import store_chunks
from .retriever       import HotelRetriever

logger = get_logger("rag.runner")


def run_step2() -> None:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   STAYWISE AI — STEP 2: RAG SYSTEM              ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    try:
        # 2A — Chunk
        chunks = run_chunking()

        # 2B — Embed
        model          = load_model()
        chunks_with_emb = embed_chunks(chunks, model=model)

        # 2C — Store
        store_chunks(chunks_with_emb, reset=True)

        # 2D — Verify with test query
        _verify_retrieval()

        logger.info("\n")
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║   ✅  STEP 2 COMPLETE                            ║")
        logger.info("║   Vector DB : data/vectorstore/                  ║")
        logger.info("║   Chunks    : data/processed/chunks.jsonl        ║")
        logger.info("║   Next      : Step 3 — LLM Integration           ║")
        logger.info("╚══════════════════════════════════════════════════╝")

    except Exception as e:
        logger.exception(f"RAG pipeline failed: {e}")
        sys.exit(1)


def _verify_retrieval() -> None:
    """Run 3 test queries to confirm the RAG system works end-to-end."""
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 2D — VERIFYING RETRIEVAL")
    logger.info("=" * 55)

    retriever = HotelRetriever()
    stats     = retriever.get_stats()
    logger.info(f"  Collection  : {stats['collection']}")
    logger.info(f"  Total chunks: {stats['total_chunks']:,}")
    logger.info(f"  Model       : {stats['model']}")

    test_queries = [
        {"query": "good hotel for family with swimming pool in Goa",  "city": None},
        {"query": "budget hotel clean rooms under 2000",               "city": None},
        {"query": "luxury 5 star hotel with spa",                      "city": None},
    ]

    logger.info("\n  Test Queries:")
    for tq in test_queries:
        results = retriever.search(tq["query"], city=tq["city"], top_k=3)
        logger.info(f"\n  Q: \"{tq['query']}\"")
        for i, r in enumerate(results, 1):
            price = f"₹{r['price_inr']:,.0f}" if r.get("price_inr") else "N/A"
            score = r.get("trust_score") or "N/A"
            logger.info(
                f"    {i}. {r.get('hotel_name','?'):<35} "
                f"| {r.get('city','?'):<15} "
                f"| Price: {price:<12} "
                f"| Trust: {score} "
                f"| Relevance: {r['relevance']}"
            )

    logger.info("\n  ✅ Retrieval verified")
