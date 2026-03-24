"""
src/ingestion/loader.py
────────────────────────
STEP 1A — Data Loader

Routes between two data source modes based on config.yaml:

  data_source_mode: "csv"  → loads from local CSV files (current behaviour)
  data_source_mode: "api"  → fetches live data from Google Places API

To switch to API mode:
  1. Open configs/config.yaml
  2. Change data_source_mode: "csv" → "api"
  3. Add GOOGLE_PLACES_API_KEY to env.example
  4. Run python3 main.py --step 1
"""

import pandas as pd
from pathlib import Path
from src.utils import load_config, get_logger, read_csv_safe

logger = get_logger("ingestion.loader")


# ─────────────────────────────────────────────────────────────
# CSV LOADERS  (data_source_mode = "csv")
# ─────────────────────────────────────────────────────────────

def _load_goibibo(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    path = raw_dir / cfg["sources"]["goibibo"]["filename"]
    if not path.exists():
        logger.warning(f"NOT FOUND: {path}")
        return pd.DataFrame()
    df = read_csv_safe(path)
    logger.info(f"Loaded Goibibo — {len(df):,} rows")
    _log_schema(df, "Goibibo")
    return df


def _load_google_csv(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    path = raw_dir / cfg["sources"]["google"]["filename"]
    if not path.exists():
        logger.warning(f"NOT FOUND: {path}")
        return pd.DataFrame()
    df = read_csv_safe(path)
    logger.info(f"Loaded Google Reviews CSV — {len(df):,} rows")
    _log_schema(df, "Google")
    return df


def _load_makemytrip(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    folder     = raw_dir / cfg["sources"]["makemytrip"]["folder"]
    if not folder.is_dir():
        logger.warning(f"NOT FOUND: {folder}")
        return pd.DataFrame()
    city_files = sorted(folder.glob("**/*.csv"))
    if not city_files:
        logger.warning(f"No CSVs inside {folder}")
        return pd.DataFrame()
    frames = []
    for f in city_files:
        df = read_csv_safe(f)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        if "city" not in df.columns:
            df["city"] = f.stem.strip().title()
        frames.append(df)
        logger.info(f"  Loaded MakeMyTrip/{f.name}: {len(df):,} rows")
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded MakeMyTrip total — {len(combined):,} rows from {len(city_files)} files")
    _log_schema(combined, "MakeMyTrip")
    return combined


def _load_from_csv(cfg: dict) -> dict:
    """Load all sources from local CSV files."""
    raw_dir = cfg["paths"]["raw_data"]
    return {
        "goibibo":    _load_goibibo(raw_dir, cfg),
        "google":     _load_google_csv(raw_dir, cfg),
        "makemytrip": _load_makemytrip(raw_dir, cfg),
    }


# ─────────────────────────────────────────────────────────────
# API LOADER  (data_source_mode = "api")
# ─────────────────────────────────────────────────────────────

def _load_from_api(cfg: dict) -> dict:
    """
    Fetch live hotel data from Google Places API.
    Returns a single 'google_places' source dict.
    """
    from .google_places import fetch_all_cities

    api_cfg      = cfg.get("google_places_api", {})
    cities       = api_cfg.get("cities", ["Mumbai", "Delhi", "Goa"])
    max_per_city = api_cfg.get("max_per_city", 20)
    fetch_details= api_cfg.get("fetch_details", False)

    logger.info(f"  Mode      : Google Places API")
    logger.info(f"  Cities    : {cities}")
    logger.info(f"  Max/city  : {max_per_city}")

    df = fetch_all_cities(
        cities        = cities,
        max_per_city  = max_per_city,
        fetch_details = fetch_details,
        save_cache    = True,
    )

    return {
        "google_places": df,
        "goibibo":       pd.DataFrame(),   # empty — not used in API mode
        "makemytrip":    pd.DataFrame(),   # empty — not used in API mode
    }


# ─────────────────────────────────────────────────────────────
# SHARED HELPER
# ─────────────────────────────────────────────────────────────

def _log_schema(df: pd.DataFrame, label: str):
    logger.info(f"  [{label}] {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"  [{label}] Columns: {list(df.columns)}")
    null_pct = (df.isnull().sum() / len(df) * 100).round(1)
    for col, pct in null_pct[null_pct > 0].items():
        logger.info(f"  [{label}]   null: {col} = {pct}%")


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def load_all_sources() -> dict:
    """
    Load hotel data from configured source.
    Reads data_source_mode from configs/config.yaml:
      "csv" → local CSV files
      "api" → Google Places API (live data)
    """
    cfg  = load_config()
    mode = cfg.get("data_source_mode", "csv").lower().strip()

    logger.info("=" * 55)
    logger.info("  STEP 1A — LOADING DATA SOURCES")
    logger.info(f"  Mode: {mode.upper()}")
    logger.info(f"  Folder: {cfg['paths']['raw_data']}")
    logger.info("=" * 55)

    if mode == "api":
        sources = _load_from_api(cfg)
    else:
        sources = _load_from_csv(cfg)

    loaded  = [k for k, v in sources.items() if not v.empty]
    skipped = [k for k, v in sources.items() if v.empty]

    logger.info(f"\n  Loaded  : {loaded}")
    if skipped:
        logger.warning(f"  Skipped : {skipped}")

    return sources
