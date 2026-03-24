"""
src/processing/cleaner.py — STEP 1B: Clean each source individually
Saves: data/processed/mmt_cleaned.csv, goibibo_cleaned.csv, google_cleaned.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import (load_config, get_logger, clean_text, clean_price,
                       clean_rating, extract_star, parse_goibibo_aspects,
                       parse_review_counts, parse_google_features)

logger = get_logger("processing.cleaner")


def clean_makemytrip(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw
    logger.info("  Cleaning MakeMyTrip...")
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    def pick(candidates):
        for c in candidates:
            if c in df.columns: return df[c]
        return pd.Series("", index=df.index)

    out = pd.DataFrame(index=df.index)
    out["hotel_name"]        = pick(["hotel_name", "name"]).apply(clean_text)
    out["city"]              = pick(["city"]).apply(clean_text).str.title()
    out["area"]              = pick(["location", "area", "locality"]).apply(clean_text)
    out["price_inr"]         = pick(["price", "hotel_price"]).apply(clean_price)
    out["tax_inr"]           = pick(["tax"]).apply(clean_price)
    out["star_rating"]       = pick(["star_rating", "stars"]).apply(extract_star)
    out["rating"]            = pick(["rating", "mmt_rating"]).apply(clean_rating)
    out["review_count"]      = pd.to_numeric(pick(["reviews", "review_count", "mmt_review_count"]), errors="coerce")
    out["landmark"]          = pick(["nearest_landmark", "landmark"]).apply(clean_text)
    out["distance_landmark"] = pick(["distance_to_landmark", "distance_landmark"]).apply(clean_text)
    out["rating_label"]      = pick(["rating_description", "rating_label"]).apply(clean_text)
    out["description"]       = pick(["description", "hotel_description"]).apply(clean_text)
    out["amenities"]         = pick(["amenities", "hotel_amenities"]).apply(clean_text)
    out["source"]            = "MakeMyTrip"
    out = _drop_invalid(out, "MakeMyTrip")
    logger.info(f"  MakeMyTrip cleaned: {len(out):,} rows")
    return out


def clean_goibibo(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw
    logger.info("  Cleaning Goibibo...")
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    def g(col):
        return df[col] if col in df.columns else pd.Series("", index=df.index)

    out = pd.DataFrame(index=df.index)
    out["hotel_name"]      = g("property_name").apply(clean_text)
    out["city"]            = g("city").apply(clean_text).str.title()
    out["area"]            = g("area").apply(clean_text)
    out["state"]           = g("state").apply(clean_text)
    out["address"]         = g("address").apply(clean_text)
    out["latitude"]        = pd.to_numeric(g("latitude"), errors="coerce")
    out["longitude"]       = pd.to_numeric(g("longitude"), errors="coerce")
    out["star_rating"]     = g("hotel_star_rating").apply(extract_star)
    out["rating"]          = g("site_review_rating").apply(lambda x: clean_rating(x, 5))
    out["review_count"]    = pd.to_numeric(g("site_review_count"), errors="coerce")
    out["hotel_type"]      = g("property_type").apply(clean_text)
    out["description"]     = g("hotel_description").apply(clean_text)
    out["facilities"]      = g("hotel_facilities").apply(clean_text)
    out["room_facilities"] = g("room_facilities").apply(clean_text)
    out["amenities"]       = g("additional_info").apply(clean_text)

    # Aspect ratings
    aspect_col = "site_stay_review_rating" if "site_stay_review_rating" in df.columns else "review_count_by_category"
    aspects = g(aspect_col).apply(parse_goibibo_aspects).apply(pd.Series)
    out = pd.concat([out, aspects], axis=1)

    # Review counts
    rev_counts = g("review_count_by_category").apply(parse_review_counts).apply(pd.Series)
    out = pd.concat([out, rev_counts], axis=1)
    out["source"] = "Goibibo"
    out = _drop_invalid(out, "Goibibo")
    logger.info(f"  Goibibo cleaned: {len(out):,} rows")
    return out


def clean_google(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw
    logger.info("  Cleaning Google Reviews...")
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    def g(col):
        return df[col] if col in df.columns else pd.Series("", index=df.index)

    out = pd.DataFrame(index=df.index)
    out["hotel_name"] = g("hotel_name").apply(clean_text)
    out["city"]       = g("city").apply(clean_text).str.title()
    out["rating"]     = g("hotel_rating").apply(lambda x: clean_rating(x, 5))
    out["price_inr"]  = g("hotel_price").apply(clean_price)
    out["amenities"]  = df.apply(parse_google_features, axis=1)
    out["source"]     = "Google"
    out = _drop_invalid(out, "Google")
    logger.info(f"  Google cleaned: {len(out):,} rows")
    return out


def _drop_invalid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cfg    = load_config()
    min_ln = cfg["processing"]["min_hotel_name_length"]
    before = len(df)
    df = df.dropna(subset=["hotel_name", "city"])
    df = df[df["hotel_name"].str.strip().str.len() >= min_ln]
    dropped = before - len(df)
    if dropped: logger.info(f"  {label}: dropped {dropped} invalid rows")
    return df.reset_index(drop=True)


def save_cleaned(df: pd.DataFrame, name: str) -> Path:
    cfg  = load_config()
    path = cfg["paths"]["processed_data"] / f"{name}_cleaned.csv"
    df.to_csv(path, index=False)
    logger.info(f"  Saved → {path.name}  ({len(df):,} rows)")
    return path


def run_cleaning(raw_sources: dict) -> dict:
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 1B — CLEANING EACH SOURCE")
    logger.info("=" * 55)
    cleaned = {}
    if not raw_sources.get("makemytrip", pd.DataFrame()).empty:
        cleaned["makemytrip"] = clean_makemytrip(raw_sources["makemytrip"])
        save_cleaned(cleaned["makemytrip"], "mmt")
    if not raw_sources.get("goibibo", pd.DataFrame()).empty:
        cleaned["goibibo"] = clean_goibibo(raw_sources["goibibo"])
        save_cleaned(cleaned["goibibo"], "goibibo")
    if not raw_sources.get("google", pd.DataFrame()).empty:
        cleaned["google"] = clean_google(raw_sources["google"])
        save_cleaned(cleaned["google"], "google")
    logger.info("\n  Cleaning Summary:")
    for src, df in cleaned.items():
        logger.info(f"    {src:<15} → {len(df):>6,} rows")
    return cleaned
