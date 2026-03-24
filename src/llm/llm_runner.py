"""
src/llm/llm_runner.py
──────────────────────
Step 3 Orchestrator
Tests all 3 LLM features end-to-end with real queries.
"""

import sys
from src.utils   import get_logger
from .qa          import HotelQA
from .summarizer  import HotelSummarizer
from .explainer   import HotelExplainer

logger = get_logger("llm.runner")


def run_step3() -> None:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   STAYWISE AI — STEP 3: LLM INTEGRATION         ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    try:
        _test_qa()
        _test_summarizer()
        _test_explainer()

        logger.info("\n")
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║   ✅  STEP 3 COMPLETE                            ║")
        logger.info("║   Features: Q&A · Summarizer · Explainer         ║")
        logger.info("║   Next    : Step 4 — Ranking Engine              ║")
        logger.info("╚══════════════════════════════════════════════════╝")

    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Step 3 failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# TEST FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _test_qa():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 3A — Q&A SYSTEM TEST")
    logger.info("=" * 55)

    qa = HotelQA()

    test_questions = [
        {
            "question": "Which hotels in Goa have a swimming pool?",
            "city": "goa",
        },
        {
            "question": "Is this hotel good for a family with kids?",
            "city": None,
        },
        {
            "question": "What are the best budget hotels in Delhi under ₹2000?",
            "city": "delhi",
            "max_price": 2000,
        },
    ]

    for tq in test_questions:
        result = qa.ask(
            question  = tq["question"],
            city      = tq.get("city"),
            max_price = tq.get("max_price"),
        )
        logger.info(f"\n  Q: {result['question']}")
        logger.info(f"  A: {result['answer']}")
        logger.info(f"  Sources ({result['num_chunks']} chunks): {result['sources'][:3]}")

    logger.info("\n  ✅ Q&A tests passed")


def _test_summarizer():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 3B — SUMMARIZER TEST")
    logger.info("=" * 55)

    summarizer = HotelSummarizer()

    test_hotels = [
        {"name": "Taj Lands End",    "city": "mumbai"},
        {"name": "BB Palace",        "city": "delhi"},
    ]

    for hotel in test_hotels:
        result = summarizer.summarize_by_name(
            hotel_name = hotel["name"],
            city       = hotel["city"],
        )
        logger.info(f"\n  Hotel : {result['hotel_name']} ({result['city']})")
        logger.info(f"\n{result['summary']}")
        logger.info("-" * 50)

    logger.info("\n  ✅ Summarizer tests passed")


def _test_explainer():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 3C — EXPLAINER TEST")
    logger.info("=" * 55)

    explainer = HotelExplainer()

    test_cases = [
        {
            "city":        "goa",
            "travel_type": "family",
            "budget":      8000,
            "priorities":  ["pool", "good food", "beach access"],
        },
        {
            "city":        "mumbai",
            "travel_type": "business",
            "budget":      6000,
            "priorities":  ["wifi", "city center", "meeting rooms"],
        },
    ]

    for tc in test_cases:
        result = explainer.recommend(**tc, top_k=3)
        logger.info(
            f"\n  🔍 {tc['travel_type'].title()} trip to {tc['city'].title()} "
            f"| Budget: ₹{tc['budget']:,} | Priorities: {tc['priorities']}"
        )
        logger.info(f"\n{result['recommendations']}")
        logger.info("-" * 55)

    logger.info("\n  ✅ Explainer tests passed")
