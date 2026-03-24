"""
src/pipeline/runner.py — Orchestrates all 4 pipeline steps
"""
import sys
import pandas as pd
from src.utils      import get_logger
from src.ingestion  import load_all_sources
from src.processing import run_cleaning, run_standardization, run_merge

logger = get_logger("pipeline.runner")


def run_step1() -> pd.DataFrame:
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   STAYWISE AI — STEP 1: DATA PIPELINE           ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    try:
        raw          = load_all_sources()
        if all(df.empty for df in raw.values()):
            logger.error("No data found in data/raw/ — check your files!")
            sys.exit(1)
        cleaned      = run_cleaning(raw)
        standardized = run_standardization(cleaned)
        unified      = run_merge(standardized)
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  ✅  STEP 1 COMPLETE                             ║")
        logger.info("║  data/processed/unified_hotels.csv               ║")
        logger.info("║  data/processed/pipeline_report.txt              ║")
        logger.info("╚══════════════════════════════════════════════════╝")
        return unified
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
