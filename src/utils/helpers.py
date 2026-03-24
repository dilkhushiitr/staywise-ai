"""
src/utils/helpers.py — Shared cleaning functions
"""
import re
import pandas as pd
import numpy as np


def clean_text(val) -> str:
    if pd.isna(val): return ""
    s = str(val).strip()
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", s)


def clean_price(val):
    if pd.isna(val): return None
    s = re.sub(r"[₹,\s]", "", str(val)).strip()
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


def clean_rating(val, max_val: float = 5.0):
    if pd.isna(val): return None
    try:
        r = float(str(val).strip())
        if r > 5: r = r / 2
        return round(min(max(r, 0.0), 5.0), 2)
    except ValueError:
        return None


def extract_star(val):
    if pd.isna(val): return None
    try:
        return int(float(re.sub(r"[^\d.]", "", str(val))))
    except (ValueError, TypeError):
        return None


def read_csv_safe(path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)


def parse_goibibo_aspects(val) -> dict:
    """'Service Quality::3.9|Amenities::3.7|...' → dict of floats"""
    out = {"rating_service": None, "rating_amenities": None, "rating_food": None,
           "rating_value": None, "rating_location": None, "rating_cleanliness": None}
    if pd.isna(val) or not str(val).strip(): return out
    key_map = {"service quality": "rating_service", "amenities": "rating_amenities",
               "food and drinks": "rating_food", "value for money": "rating_value",
               "location": "rating_location", "cleanliness": "rating_cleanliness"}
    for part in str(val).split("|"):
        if "::" in part:
            k, _, v = part.partition("::")
            k = k.strip().lower()
            if k in key_map:
                try: out[key_map[k]] = float(v.strip()) if v.strip() else None
                except ValueError: pass
    return out


def parse_review_counts(val) -> dict:
    """'positive reviews::74|critical reviews::13' → dict"""
    pos, neg = None, None
    if not pd.isna(val):
        for part in str(val).split("|"):
            pl = part.lower()
            if "positive" in pl and "::" in pl:
                try: pos = int(pl.split("::")[1].strip())
                except: pass
            elif "critical" in pl and "::" in pl:
                try: neg = int(pl.split("::")[1].strip())
                except: pass
    return {"positive_reviews": pos, "negative_reviews": neg}


def parse_google_features(row: pd.Series) -> str:
    """Combine Feature_1…Feature_N into pipe-separated string"""
    parts = [str(row[c]).strip() for c in row.index
             if re.match(r"feature_\d+", str(c), re.IGNORECASE)
             and str(row[c]).strip().lower() not in ("nan", "none", "")]
    return " | ".join(parts)
