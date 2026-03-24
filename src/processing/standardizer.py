"""
src/processing/standardizer.py — STEP 1C: Unified 35-column schema
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import load_config, get_logger

logger = get_logger("processing.standardizer")

UNIFIED_COLS = [
    "hotel_id", "hotel_name", "city", "area", "state", "country",
    "address", "latitude", "longitude",
    "star_rating", "price_inr", "tax_inr", "total_price_inr",
    "rating_overall", "review_count", "positive_reviews", "negative_reviews",
    "rating_service", "rating_amenities", "rating_food",
    "rating_value", "rating_location", "rating_cleanliness",
    "hotel_type", "amenities", "facilities", "room_facilities",
    "description", "landmark", "distance_landmark", "rating_label", "source",
]


def standardize(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    def col(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["hotel_id"]          = [f"{prefix}_{i:06d}" for i in range(len(df))]
    out["hotel_name"]        = col("hotel_name", "")
    out["city"]              = col("city", "")
    out["area"]              = col("area", "").fillna("")
    out["state"]             = col("state", "").fillna("")
    out["country"]           = "India"
    out["address"]           = col("address", "").fillna("")
    out["latitude"]          = pd.to_numeric(col("latitude"), errors="coerce")
    out["longitude"]         = pd.to_numeric(col("longitude"), errors="coerce")
    out["star_rating"]       = pd.to_numeric(col("star_rating"), errors="coerce")
    out["price_inr"]         = pd.to_numeric(col("price_inr"), errors="coerce")
    out["tax_inr"]           = pd.to_numeric(col("tax_inr"), errors="coerce")
    price = out["price_inr"].fillna(0)
    tax   = out["tax_inr"].fillna(0)
    total = price + tax
    out["total_price_inr"]   = total.where(total > 0, np.nan)
    out["rating_overall"]    = pd.to_numeric(col("rating"), errors="coerce")
    out["review_count"]      = pd.to_numeric(col("review_count"), errors="coerce")
    out["positive_reviews"]  = pd.to_numeric(col("positive_reviews"), errors="coerce")
    out["negative_reviews"]  = pd.to_numeric(col("negative_reviews"), errors="coerce")
    for asp in ["rating_service","rating_amenities","rating_food",
                "rating_value","rating_location","rating_cleanliness"]:
        out[asp] = pd.to_numeric(col(asp), errors="coerce")
    out["hotel_type"]        = col("hotel_type", "Hotel").fillna("Hotel")
    out["amenities"]         = col("amenities", "").fillna("")
    out["facilities"]        = col("facilities", "").fillna("")
    out["room_facilities"]   = col("room_facilities", "").fillna("")
    out["description"]       = col("description", "").fillna("")
    out["landmark"]          = col("landmark", "").fillna("")
    out["distance_landmark"] = col("distance_landmark", "").fillna("")
    out["rating_label"]      = col("rating_label", "").fillna("")
    out["source"]            = col("source", prefix).fillna(prefix)
    for c in UNIFIED_COLS:
        if c not in out.columns: out[c] = np.nan
    return out[UNIFIED_COLS]


def run_standardization(cleaned: dict) -> dict:
    cfg = load_config()
    prefix_map = {
        "makemytrip": cfg["sources"]["makemytrip"]["id_prefix"],
        "goibibo":    cfg["sources"]["goibibo"]["id_prefix"],
        "google":     cfg["sources"]["google"]["id_prefix"],
    }
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 1C — STANDARDIZING SCHEMA")
    logger.info(f"  Target: {len(UNIFIED_COLS)} unified columns")
    logger.info("=" * 55)
    standardized = {}
    for src, df in cleaned.items():
        if df.empty: continue
        prefix = prefix_map.get(src, src.upper()[:3])
        std    = standardize(df, prefix)
        standardized[src] = std
        out_path = cfg["paths"]["processed_data"] / f"{src}_standard.csv"
        std.to_csv(out_path, index=False)
        logger.info(f"  {src:<15} → {len(std):>6,} rows  saved: {out_path.name}")
    return standardized
