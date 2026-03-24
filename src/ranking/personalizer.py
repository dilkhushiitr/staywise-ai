"""
src/ranking/personalizer.py
────────────────────────────
STEP 4C — Personalization + Hidden Issues Detection

Two features:
1. PERSONALIZATION: Re-ranks hotels based on travel type
   - family   → boosts service + value + cleanliness
   - couple   → boosts location + cleanliness + food
   - solo     → boosts value + location
   - business → boosts service + location + amenities
   - luxury   → boosts trust + rating + star rating
   - budget   → boosts value, penalizes high price

2. HIDDEN ISSUES: Flags hotels with patterns suggesting problems
   - Very low cleanliness scores
   - Very low service scores
   - High negative review ratio
   - Large gap between overall rating and aspect ratings

Usage:
    from src.ranking.personalizer import Personalizer
    p = Personalizer()

    # Personalized top 20
    results = p.personalize(city="goa", travel_type="family", top_n=10)

    # Hidden issues for a hotel
    issues = p.detect_hidden_issues("GBB_000042")
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils import get_logger
from .scorer   import load_scored_hotels

logger = get_logger("ranking.personalizer")


# ── Travel type → aspect weight boosts ───────────────────────
TRAVEL_TYPE_PROFILES = {
    "family": {
        "rating_service":     2.5,
        "rating_cleanliness": 2.5,
        "rating_value":       2.0,
        "rating_food":        1.5,
    },
    "couple": {
        "rating_location":    2.5,
        "rating_cleanliness": 2.0,
        "rating_food":        2.0,
        "rating_service":     1.5,
    },
    "solo": {
        "rating_value":       3.0,
        "rating_location":    2.0,
        "rating_cleanliness": 1.5,
    },
    "business": {
        "rating_service":     3.0,
        "rating_location":    2.5,
        "rating_cleanliness": 1.5,
    },
    "luxury": {
        "trust_score":        3.0,
        "rating_overall":     2.5,
        "rating_service":     2.0,
        "rating_cleanliness": 2.0,
    },
    "budget": {
        "rating_value":       3.5,
        "rating_cleanliness": 1.5,
    },
}

# ── Hidden issue thresholds ───────────────────────────────────
ISSUE_THRESHOLDS = {
    "low_cleanliness":   2.5,   # cleanliness below this is a red flag
    "low_service":       2.5,   # service below this is a red flag
    "low_food":          2.0,   # food below this is a red flag
    "low_value":         2.0,   # value below this is a red flag
    "high_negative_ratio": 0.4, # >40% negative reviews = concern
    "rating_gap":        1.5,   # gap between overall & aspect avg
}


def _personalized_score(
    row:         pd.Series,
    profile:     dict[str, float],
    base_score:  float,
) -> float:
    """Compute personalized score by boosting profile-relevant aspects."""
    boost_sum   = 0.0
    boost_count = 0

    for aspect, weight in profile.items():
        if aspect == "trust_score":
            val = row.get("trust_score")
            if pd.notna(val):
                boost_sum   += (float(val) / 10) * weight
                boost_count += weight
        elif aspect == "rating_overall":
            val = row.get("rating_overall")
            if pd.notna(val):
                boost_sum   += (float(val) / 5) * weight
                boost_count += weight
        else:
            val = row.get(aspect)
            if pd.notna(val):
                boost_sum   += (float(val) / 5) * weight
                boost_count += weight

    if boost_count == 0:
        return base_score

    boost      = (boost_sum / boost_count) * 10
    final      = (base_score * 0.5) + (boost * 0.5)
    return round(min(10.0, final), 3)


def _detect_issues(row: pd.Series) -> list[str]:
    """Return list of hidden issue strings for a hotel row."""
    issues = []
    t      = ISSUE_THRESHOLDS

    clean = row.get("rating_cleanliness")
    if pd.notna(clean) and float(clean) < t["low_cleanliness"]:
        issues.append(f"⚠️  Low cleanliness score ({clean}/5)")

    service = row.get("rating_service")
    if pd.notna(service) and float(service) < t["low_service"]:
        issues.append(f"⚠️  Low service score ({service}/5)")

    food = row.get("rating_food")
    if pd.notna(food) and float(food) < t["low_food"]:
        issues.append(f"⚠️  Low food rating ({food}/5)")

    value = row.get("rating_value")
    if pd.notna(value) and float(value) < t["low_value"]:
        issues.append(f"⚠️  Poor value for money ({value}/5)")

    # Negative review ratio
    pos = row.get("positive_reviews")
    neg = row.get("negative_reviews")
    if pd.notna(pos) and pd.notna(neg) and (pos + neg) > 0:
        neg_ratio = float(neg) / (float(pos) + float(neg))
        if neg_ratio > t["high_negative_ratio"]:
            issues.append(
                f"⚠️  High negative review ratio "
                f"({int(neg)} negative out of {int(pos+neg)} total)"
            )

    # Rating gap: overall much higher than aspect average
    aspects = ["rating_cleanliness", "rating_service",
               "rating_food", "rating_value", "rating_location"]
    aspect_vals = [float(row[a]) for a in aspects if pd.notna(row.get(a))]
    overall = row.get("rating_overall")
    if aspect_vals and pd.notna(overall):
        avg_aspect = np.mean(aspect_vals)
        gap = float(overall) - avg_aspect
        if gap > t["rating_gap"]:
            issues.append(
                f"⚠️  Overall rating ({overall}/5) much higher than "
                f"aspect average ({avg_aspect:.1f}/5) — possibly inflated"
            )

    return issues


class Personalizer:
    """
    Personalized hotel ranking + hidden issue detection.
    """

    def __init__(self):
        logger.info("  Loading hotel data for personalization...")
        self.df = load_scored_hotels()
        logger.info(f"  Personalizer ready — {len(self.df):,} hotels")

    def personalize(
        self,
        city:        Optional[str] = None,
        travel_type: str           = "family",
        max_price:   Optional[float] = None,
        min_stars:   Optional[int]   = None,
        top_n:       int             = 20,
    ) -> pd.DataFrame:
        """
        Return top_n hotels personalized for a travel type.

        Args:
            city        : filter by city
            travel_type : family / couple / solo / business / luxury / budget
            max_price   : max price per night INR
            min_stars   : minimum star rating
            top_n       : number of results

        Returns:
            DataFrame with 'personalized_score' and 'hidden_issues' columns
        """
        df      = self.df.copy()
        profile = TRAVEL_TYPE_PROFILES.get(
            travel_type.lower(),
            TRAVEL_TYPE_PROFILES["family"]
        )

        # ── Filters ───────────────────────────────────────────
        if city:
            df = df[df["city"].str.lower().str.strip() == city.lower().strip()]
        if max_price:
            df = df[df["price_inr"].isna() | (df["price_inr"] <= max_price)]
        if min_stars:
            df = df[df["star_rating"].isna() | (df["star_rating"] >= min_stars)]

        if df.empty:
            logger.warning(f"  No hotels found for city='{city}'")
            return pd.DataFrame()

        # ── Personalized scoring ──────────────────────────────
        df["personalized_score"] = df.apply(
            lambda row: _personalized_score(row, profile, row["composite_score"]),
            axis=1,
        )

        # ── Hidden issues ─────────────────────────────────────
        df["hidden_issues"] = df.apply(
            lambda row: _detect_issues(row), axis=1
        )
        df["issue_count"] = df["hidden_issues"].apply(len)

        # ── Sort: personalized_score desc, issue_count asc ────
        df = df.sort_values(
            ["personalized_score", "issue_count"],
            ascending=[False, True],
        )
        df = df.head(top_n).reset_index(drop=True)
        df["rank"] = df.index + 1

        logger.info(
            f"  Personalized top {top_n} for '{travel_type}' in '{city or 'all'}'"
        )

        return df[[
            "rank", "hotel_id", "hotel_name", "city", "area",
            "star_rating", "price_inr",
            "rating_overall", "rating_cleanliness", "rating_food",
            "rating_location", "rating_service", "rating_value",
            "trust_score", "composite_score", "personalized_score",
            "review_count", "issue_count", "hidden_issues", "source",
        ]]

    def detect_hidden_issues(self, hotel_id: str) -> dict:
        """
        Run hidden issue detection for a specific hotel.

        Returns:
            {
                "hotel_name": str,
                "city":       str,
                "issues":     list[str],
                "issue_count": int,
                "verdict":    "clean" | "minor_concerns" | "major_concerns"
            }
        """
        matches = self.df[self.df["hotel_id"] == hotel_id]
        if matches.empty:
            return {"error": f"Hotel ID '{hotel_id}' not found"}

        row    = matches.iloc[0]
        issues = _detect_issues(row)
        n      = len(issues)
        verdict = "clean" if n == 0 else "minor_concerns" if n <= 2 else "major_concerns"

        return {
            "hotel_id":   hotel_id,
            "hotel_name": row.get("hotel_name", ""),
            "city":       row.get("city", ""),
            "issues":     issues,
            "issue_count": n,
            "verdict":    verdict,
        }

    def scan_all_issues(self, city: Optional[str] = None) -> pd.DataFrame:
        """
        Scan all hotels (or hotels in a city) for hidden issues.
        Returns DataFrame sorted by issue_count descending.
        """
        df = self.df.copy()
        if city:
            df = df[df["city"].str.lower().str.strip() == city.lower().strip()]

        df["hidden_issues"] = df.apply(lambda r: _detect_issues(r), axis=1)
        df["issue_count"]   = df["hidden_issues"].apply(len)

        flagged = df[df["issue_count"] > 0].sort_values(
            "issue_count", ascending=False
        ).reset_index(drop=True)

        logger.info(
            f"  Hidden issue scan: {len(flagged):,} hotels flagged "
            f"out of {len(df):,}"
        )
        return flagged[["hotel_id", "hotel_name", "city",
                        "rating_overall", "trust_score",
                        "issue_count", "hidden_issues"]]
