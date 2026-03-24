"""
src/ranking/scorer.py
──────────────────────
STEP 4A — Hotel Scoring

Computes a rich composite score for each hotel combining:
  - Trust score (from Step 1)
  - Aspect ratings (cleanliness, food, location, etc.)
  - Review volume
  - Price-value ratio
  - Recency bonus (if available)

These scores are the foundation for all ranking and personalization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import load_config, get_logger

logger = get_logger("ranking.scorer")


# ── Score weights (must sum to 1.0) ──────────────────────────
WEIGHTS = {
    "trust_score":        0.30,   # overall trust (computed in Step 1)
    "rating_overall":     0.20,   # raw overall rating
    "rating_cleanliness": 0.10,
    "rating_food":        0.10,
    "rating_location":    0.10,
    "rating_service":     0.10,
    "rating_value":       0.10,
}


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to 0–1. NaNs stay NaN."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s.apply(lambda x: 0.5 if pd.notna(x) else np.nan)
    return (s - mn) / (mx - mn)


def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'composite_score' column (0–10) to the dataframe.

    Strategy:
    - Normalize each rating dimension to 0–1
    - Apply weights
    - Scale result to 0–10
    - Fill missing dimensions with column median
    """
    df = df.copy()

    # Normalize trust_score (already 0–10, scale to 0–1)
    df["_trust_norm"] = _normalize_series(df["trust_score"].clip(0, 10) / 10)

    # Normalize rating_overall (0–5, scale to 0–1)
    df["_rating_norm"] = _normalize_series(df["rating_overall"].clip(0, 5) / 5)

    # Normalize aspect ratings (0–5, scale to 0–1)
    for aspect in ["rating_cleanliness", "rating_food",
                   "rating_location", "rating_service", "rating_value"]:
        df[f"_{aspect}_norm"] = _normalize_series(
            df[aspect].clip(0, 5) / 5
        )

    # Fill NaN norms with median so missing data doesn't zero-out score
    norm_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_norm")]
    for col in norm_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else 0.5)

    # Weighted sum
    score = (
        df["_trust_norm"]              * WEIGHTS["trust_score"]        +
        df["_rating_norm"]             * WEIGHTS["rating_overall"]      +
        df["_rating_cleanliness_norm"] * WEIGHTS["rating_cleanliness"]  +
        df["_rating_food_norm"]        * WEIGHTS["rating_food"]         +
        df["_rating_location_norm"]    * WEIGHTS["rating_location"]     +
        df["_rating_service_norm"]     * WEIGHTS["rating_service"]      +
        df["_rating_value_norm"]       * WEIGHTS["rating_value"]
    )

    df["composite_score"] = (score * 10).round(3)

    # Drop temp columns
    df.drop(columns=norm_cols, inplace=True)

    logger.info(f"  Composite scores computed for {len(df):,} hotels")
    logger.info(f"  Score range: {df['composite_score'].min():.2f} – "
                f"{df['composite_score'].max():.2f}")

    return df


def load_scored_hotels() -> pd.DataFrame:
    """
    Load unified_hotels.csv, compute composite scores, return DataFrame.
    Called by ranker and personalizer.
    """
    cfg      = load_config()
    csv_path = cfg["paths"]["processed_data"] / "unified_hotels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "unified_hotels.csv not found. Run Step 1 first: python3 main.py --step 1"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    df = compute_composite_scores(df)
    return df
