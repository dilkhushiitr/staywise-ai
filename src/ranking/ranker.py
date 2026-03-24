"""
src/ranking/ranker.py
──────────────────────
STEP 4B — Dynamic Ranking Engine

Ranks hotels dynamically based on what the user is searching for.

Key insight: the same hotel ranks differently for different queries.
  - "best food in Goa"      → food rating weighted 3x
  - "clean rooms Mumbai"    → cleanliness weighted 3x
  - "cheap hotels Delhi"    → value + price weighted 3x

Usage:
    from src.ranking.ranker import HotelRanker
    ranker = HotelRanker()
    results = ranker.search(city="goa", query="best food", top_n=10)
"""

import re
import pandas as pd
import numpy as np
from typing import Optional
from src.utils import get_logger
from .scorer   import load_scored_hotels

logger = get_logger("ranking.ranker")


# ── Query intent → aspect boost map ──────────────────────────
# Each keyword pattern boosts the weight of a specific rating aspect
INTENT_PATTERNS = [
    # Food / Restaurant
    (r"\b(food|restaurant|dining|eat|meal|breakfast|lunch|dinner|cuisine)\b",
     "rating_food", 3.0),

    # Cleanliness
    (r"\b(clean|hygiene|hygienic|neat|tidy|spotless)\b",
     "rating_cleanliness", 3.0),

    # Location
    (r"\b(location|central|near|close to|area|accessible|connectivity)\b",
     "rating_location", 3.0),

    # Service
    (r"\b(service|staff|helpful|friendly|hospitality|reception)\b",
     "rating_service", 3.0),

    # Value / Budget
    (r"\b(budget|cheap|affordable|value|economical|low.?cost|inexpensive)\b",
     "rating_value", 3.0),

    # Luxury
    (r"\b(luxury|5.?star|premium|high.?end|lavish|opulent)\b",
     "trust_score", 2.0),

    # Family
    (r"\b(family|kids|children|child.?friendly|playground)\b",
     "rating_service", 2.0),
]


def _detect_intent_boosts(query: str) -> dict[str, float]:
    """
    Parse a query string and return a dict of {aspect: boost_multiplier}.
    Multiple intents can be detected simultaneously.
    """
    boosts = {}
    q = query.lower()
    for pattern, aspect, multiplier in INTENT_PATTERNS:
        if re.search(pattern, q):
            boosts[aspect] = max(boosts.get(aspect, 1.0), multiplier)
    return boosts


def _apply_dynamic_score(
    df:     pd.DataFrame,
    boosts: dict[str, float],
) -> pd.Series:
    """
    Re-compute a dynamic score by applying intent boosts to aspect ratings.
    Returns a Series of dynamic scores (0–10).
    """
    if not boosts:
        return df["composite_score"]

    # Start from composite_score base
    dynamic = df["composite_score"].copy()

    for aspect, multiplier in boosts.items():
        if aspect in df.columns:
            # Normalize aspect to 0–1 then boost
            col = df[aspect].clip(0, 5) / 5
            col = col.fillna(col.median() if pd.notna(col.median()) else 0.5)
            dynamic = dynamic + (col * multiplier * 2)   # 2 = scale factor

    # Re-normalize to 0–10
    mn, mx = dynamic.min(), dynamic.max()
    if mx > mn:
        dynamic = ((dynamic - mn) / (mx - mn)) * 10

    return dynamic.round(3)


class HotelRanker:
    """
    Dynamic hotel ranking engine.
    Adjusts ranking based on query intent keywords.
    """

    def __init__(self):
        logger.info("  Loading hotel data for ranking...")
        self.df = load_scored_hotels()
        logger.info(f"  HotelRanker ready — {len(self.df):,} hotels loaded")

    def search(
        self,
        city:       Optional[str]   = None,
        query:      str             = "",
        min_price:  Optional[float] = None,
        max_price:  Optional[float] = None,
        min_stars:  Optional[int]   = None,
        min_rating: Optional[float] = None,
        top_n:      int             = 20,
    ) -> pd.DataFrame:
        """
        Search and rank hotels dynamically.

        Args:
            city       : filter by city (case insensitive)
            query      : natural language e.g. "best food hotel"
            min_price  : minimum price INR
            max_price  : maximum price INR
            min_stars  : minimum star rating (1–5)
            min_rating : minimum overall rating (0–5)
            top_n      : number of results

        Returns:
            DataFrame of top_n hotels sorted by dynamic score
        """
        df = self.df.copy()

        # ── Filters ───────────────────────────────────────────
        if city:
            df = df[df["city"].str.lower().str.strip() == city.lower().strip()]

        if min_price is not None:
            df = df[df["price_inr"].isna() | (df["price_inr"] >= min_price)]

        if max_price is not None:
            df = df[df["price_inr"].isna() | (df["price_inr"] <= max_price)]

        if min_stars is not None:
            df = df[df["star_rating"].isna() | (df["star_rating"] >= min_stars)]

        if min_rating is not None:
            df = df[df["rating_overall"].isna() | (df["rating_overall"] >= min_rating)]

        if df.empty:
            logger.warning(f"  No hotels found for city='{city}' with given filters")
            return pd.DataFrame()

        # ── Dynamic scoring ───────────────────────────────────
        boosts = _detect_intent_boosts(query) if query else {}
        if boosts:
            logger.info(f"  Intent detected: {boosts}")

        df["dynamic_score"] = _apply_dynamic_score(df, boosts)

        # ── Sort & return top N ───────────────────────────────
        df = df.sort_values("dynamic_score", ascending=False)
        df = df.head(top_n).reset_index(drop=True)
        df["rank"] = df.index + 1

        logger.info(
            f"  Ranked {top_n} hotels"
            f" | city={city or 'all'}"
            f" | query='{query}'"
            f" | filters: price={min_price}–{max_price}, stars≥{min_stars}"
        )

        return df[[
            "rank", "hotel_id", "hotel_name", "city", "area",
            "star_rating", "price_inr", "total_price_inr",
            "rating_overall", "rating_cleanliness", "rating_food",
            "rating_location", "rating_service", "rating_value",
            "trust_score", "composite_score", "dynamic_score",
            "review_count", "source", "amenities",
        ]]

    def get_hotel_by_id(self, hotel_id: str) -> pd.Series | None:
        """Fetch a single hotel by hotel_id."""
        matches = self.df[self.df["hotel_id"] == hotel_id]
        return matches.iloc[0] if not matches.empty else None
