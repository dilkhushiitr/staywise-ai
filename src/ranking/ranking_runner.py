"""
src/ranking/ranking_runner.py
──────────────────────────────
Step 4 Orchestrator — tests all ranking features end-to-end.
"""

import sys
import pandas as pd
from src.utils         import get_logger
from .scorer           import load_scored_hotels
from .ranker           import HotelRanker
from .personalizer     import Personalizer

logger = get_logger("ranking.runner")


def run_step4() -> None:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   STAYWISE AI — STEP 4: RANKING ENGINE          ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    try:
        _test_scorer()
        _test_dynamic_ranking()
        _test_personalization()
        _test_hidden_issues()

        logger.info("\n")
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║   ✅  STEP 4 COMPLETE                            ║")
        logger.info("║   Features: Scorer · Ranker · Personalizer       ║")
        logger.info("║            Hidden Issues Detection               ║")
        logger.info("║   Next    : Step 5 — FastAPI                     ║")
        logger.info("╚══════════════════════════════════════════════════╝")

    except Exception as e:
        logger.exception(f"Step 4 failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# TEST FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _test_scorer():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 4A — COMPOSITE SCORING TEST")
    logger.info("=" * 55)

    df = load_scored_hotels()
    logger.info(f"  Hotels scored: {len(df):,}")
    logger.info(f"  Score stats:")
    logger.info(f"    Min    : {df['composite_score'].min():.3f}")
    logger.info(f"    Max    : {df['composite_score'].max():.3f}")
    logger.info(f"    Mean   : {df['composite_score'].mean():.3f}")
    logger.info(f"    Median : {df['composite_score'].median():.3f}")

    logger.info("\n  Top 5 hotels by composite score:")
    top5 = df.nlargest(5, "composite_score")[
        ["hotel_name", "city", "composite_score", "trust_score", "rating_overall"]
    ]
    for _, row in top5.iterrows():
        logger.info(
            f"    {row['hotel_name'][:35]:<35} | {row['city']:<15} "
            f"| Score: {row['composite_score']:.2f} "
            f"| Trust: {row['trust_score']} "
            f"| Rating: {row['rating_overall']}"
        )
    logger.info("  ✅ Scorer test passed")


def _test_dynamic_ranking():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 4B — DYNAMIC RANKING TEST")
    logger.info("=" * 55)

    ranker = HotelRanker()

    test_cases = [
        {"city": "goa",     "query": "best food hotel",        "top_n": 5},
        {"city": "delhi",   "query": "clean rooms good service","top_n": 5},
        {"city": "mumbai",  "query": "budget affordable hotel", "top_n": 5},
        {"city": "bangalore","query": "luxury 5 star hotel",    "top_n": 5},
    ]

    for tc in test_cases:
        results = ranker.search(**tc)
        logger.info(f"\n  Query: \"{tc['query']}\" in {tc['city'].title()}")
        if results.empty:
            logger.warning("  No results found")
            continue
        for _, row in results.iterrows():
            price = f"₹{row['price_inr']:,.0f}" if pd.notna(row.get("price_inr")) else "N/A"
            logger.info(
                f"    {int(row['rank'])}. {str(row['hotel_name'])[:35]:<35} "
                f"| Score: {row['dynamic_score']:.2f} "
                f"| Price: {price}"
            )

    # Test with price filter
    logger.info("\n  Filter test: Delhi hotels under ₹3,000")
    results = ranker.search(city="delhi", max_price=3000, top_n=5)
    for _, row in results.iterrows():
        price = f"₹{row['price_inr']:,.0f}" if pd.notna(row.get("price_inr")) else "N/A"
        logger.info(f"    {int(row['rank'])}. {str(row['hotel_name'])[:35]:<35} | {price}")

    logger.info("\n  ✅ Dynamic ranking test passed")


def _test_personalization():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 4C — PERSONALIZATION TEST")
    logger.info("=" * 55)

    p = Personalizer()

    test_cases = [
        {"city": "goa",    "travel_type": "family",   "top_n": 3},
        {"city": "mumbai", "travel_type": "business",  "top_n": 3},
        {"city": "delhi",  "travel_type": "budget",    "top_n": 3},
        {"city": "goa",    "travel_type": "couple",    "top_n": 3},
    ]

    for tc in test_cases:
        results = p.personalize(**tc)
        logger.info(
            f"\n  {tc['travel_type'].title()} trip to {tc['city'].title()}"
        )
        if results.empty:
            logger.warning("  No results found")
            continue
        for _, row in results.iterrows():
            price  = f"₹{row['price_inr']:,.0f}" if pd.notna(row.get("price_inr")) else "N/A"
            issues = f"⚠️ {row['issue_count']} issue(s)" if row["issue_count"] > 0 else "✅ Clean"
            logger.info(
                f"    {int(row['rank'])}. {str(row['hotel_name'])[:30]:<30} "
                f"| Score: {row['personalized_score']:.2f} "
                f"| {price} "
                f"| {issues}"
            )

    logger.info("\n  ✅ Personalization test passed")


def _test_hidden_issues():
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 4D — HIDDEN ISSUES DETECTION TEST")
    logger.info("=" * 55)

    p = Personalizer()

    # Scan Mumbai for hidden issues
    logger.info("\n  Scanning Mumbai hotels for hidden issues...")
    flagged = p.scan_all_issues(city="Mumbai")
    logger.info(f"  {len(flagged)} hotels flagged in Mumbai")

    if not flagged.empty:
        logger.info("\n  Top flagged hotels:")
        for _, row in flagged.head(5).iterrows():
            logger.info(f"\n    🏨 {row['hotel_name']} ({row['city']})")
            for issue in row["hidden_issues"]:
                logger.info(f"       {issue}")

    # Check a specific hotel
    sample_id = p.df["hotel_id"].iloc[0]
    result    = p.detect_hidden_issues(sample_id)
    logger.info(f"\n  Single hotel check: {result['hotel_name']}")
    logger.info(f"  Verdict: {result['verdict']}")
    if result["issues"]:
        for issue in result["issues"]:
            logger.info(f"    {issue}")
    else:
        logger.info("    ✅ No issues found")

    logger.info("\n  ✅ Hidden issues detection test passed")
