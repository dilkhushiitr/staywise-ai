"""
src/rag/chunker.py
───────────────────
STEP 2A — Document Chunking

Reads unified_hotels.csv and converts each hotel into
one or more text chunks ready for embedding.

Chunking strategy:
  - Each hotel gets 1 primary chunk (core identity + ratings)
  - Hotels with long descriptions get a 2nd chunk (description only)
  - Each chunk carries metadata for filtering (city, price, rating, source)

Output: data/processed/chunks.jsonl  (one JSON object per line)
"""

import json
import pandas as pd
from pathlib import Path
from src.utils import load_config, get_logger

logger = get_logger("rag.chunker")

# A chunk longer than this gets a 2nd description-only chunk
DESCRIPTION_CHUNK_THRESHOLD = 200


def _make_primary_chunk(row: pd.Series) -> str:
    """
    Core chunk: identity + amenities + aspect ratings + trust score.
    Always created for every hotel.
    """
    parts = []

    if row.get("hotel_name"):   parts.append(f"Hotel: {row['hotel_name']}")
    if row.get("city"):         parts.append(f"City: {row['city']}")
    if row.get("area"):         parts.append(f"Area: {row['area']}")
    if row.get("hotel_type"):   parts.append(f"Type: {row['hotel_type']}")
    if pd.notna(row.get("star_rating")):
        parts.append(f"Star rating: {int(row['star_rating'])} stars")
    if pd.notna(row.get("price_inr")):
        parts.append(f"Price: ₹{row['price_inr']:,.0f} per night")
    if row.get("amenities"):    parts.append(f"Amenities: {row['amenities']}")
    if row.get("facilities"):   parts.append(f"Facilities: {row['facilities']}")
    if row.get("room_facilities"): parts.append(f"Room amenities: {row['room_facilities']}")
    if row.get("landmark"):     parts.append(f"Near: {row['landmark']}")

    # Aspect ratings
    aspects = {
        "Service":     row.get("rating_service"),
        "Amenities":   row.get("rating_amenities"),
        "Food":        row.get("rating_food"),
        "Value":       row.get("rating_value"),
        "Location":    row.get("rating_location"),
        "Cleanliness": row.get("rating_cleanliness"),
    }
    asp_str = [f"{k} {v}/5" for k, v in aspects.items() if pd.notna(v)]
    if asp_str:
        parts.append("Guest ratings — " + ", ".join(asp_str))

    if pd.notna(row.get("rating_overall")):
        parts.append(f"Overall rating: {row['rating_overall']}/5")
    if pd.notna(row.get("trust_score")):
        parts.append(f"Trust score: {row['trust_score']}/10")
    if pd.notna(row.get("review_count")):
        parts.append(f"Total reviews: {int(row['review_count'])}")

    return " | ".join(parts)


def _make_description_chunk(row: pd.Series) -> str:
    """
    Secondary chunk: hotel name + city + full description text.
    Only created when description is long enough to be meaningful.
    """
    parts = []
    if row.get("hotel_name"): parts.append(f"Hotel: {row['hotel_name']}")
    if row.get("city"):       parts.append(f"City: {row['city']}")
    if row.get("description"):parts.append(f"About: {row['description']}")
    return " | ".join(parts)


def _build_metadata(row: pd.Series) -> dict:
    """
    Metadata stored alongside each chunk in ChromaDB.
    Used for filtering (e.g. city=Mumbai, price < 3000).
    """
    def safe_float(val):
        try: return float(val) if pd.notna(val) else None
        except: return None

    def safe_int(val):
        try: return int(val) if pd.notna(val) else None
        except: return None

    return {
        "hotel_id":     str(row.get("hotel_id", "")),
        "hotel_name":   str(row.get("hotel_name", "")),
        "city":         str(row.get("city", "")).lower().strip(),
        "area":         str(row.get("area", "")),
        "state":        str(row.get("state", "")),
        "source":       str(row.get("source", "")),
        "hotel_type":   str(row.get("hotel_type", "")),
        "star_rating":  safe_int(row.get("star_rating")),
        "price_inr":    safe_float(row.get("price_inr")),
        "rating_overall": safe_float(row.get("rating_overall")),
        "trust_score":  safe_float(row.get("trust_score")),
        "review_count": safe_int(row.get("review_count")),
        "rating_cleanliness": safe_float(row.get("rating_cleanliness")),
        "rating_food":  safe_float(row.get("rating_food")),
        "rating_location": safe_float(row.get("rating_location")),
        "rating_service":  safe_float(row.get("rating_service")),
        "rating_value":    safe_float(row.get("rating_value")),
    }


def run_chunking() -> list[dict]:
    """
    Main entry: reads unified_hotels.csv → builds chunks → saves chunks.jsonl
    Returns list of chunk dicts.
    """
    cfg       = load_config()
    proc_dir  = cfg["paths"]["processed_data"]
    csv_path  = proc_dir / "unified_hotels.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"unified_hotels.csv not found at {csv_path}\n"
            "Run Step 1 first: python main.py --step 1"
        )

    logger.info("=" * 55)
    logger.info("  STEP 2A — CHUNKING HOTEL DATA")
    logger.info("=" * 55)

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  Loaded unified_hotels.csv — {len(df):,} hotels")

    chunks      = []
    chunk_id    = 0
    desc_chunks = 0

    for _, row in df.iterrows():
        metadata = _build_metadata(row)

        # ── Primary chunk (always) ────────────────────────────
        primary_text = _make_primary_chunk(row)
        chunks.append({
            "chunk_id":   f"chunk_{chunk_id:06d}",
            "hotel_id":   metadata["hotel_id"],
            "chunk_type": "primary",
            "text":       primary_text,
            "metadata":   metadata,
        })
        chunk_id += 1

        # ── Description chunk (only if description is long) ───
        desc = str(row.get("description", "")).strip()
        if len(desc) >= DESCRIPTION_CHUNK_THRESHOLD:
            desc_text = _make_description_chunk(row)
            chunks.append({
                "chunk_id":   f"chunk_{chunk_id:06d}",
                "hotel_id":   metadata["hotel_id"],
                "chunk_type": "description",
                "text":       desc_text,
                "metadata":   metadata,
            })
            chunk_id    += 1
            desc_chunks += 1

    # Save to JSONL
    out_path = proc_dir / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"  Primary chunks   : {len(df):,}")
    logger.info(f"  Description chunks: {desc_chunks:,}")
    logger.info(f"  Total chunks     : {len(chunks):,}")
    logger.info(f"  Saved → chunks.jsonl")

    return chunks
