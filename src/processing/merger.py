"""
src/processing/merger.py — STEP 1D: Merge + Trust Score + Export
Outputs: unified_hotels.csv, pipeline_report.txt
"""
import numpy as np
import pandas as pd
from datetime import datetime
from src.utils import load_config, get_logger

logger = get_logger("processing.merger")


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    key = (df["hotel_name"].str.lower().str.strip() + "|" +
           df["city"].str.lower().str.strip()       + "|" +
           df["source"].str.lower())
    df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    logger.info(f"  Dedup: {before:,} → {len(df):,}  ({before - len(df)} removed)")
    return df


def add_trust_scores(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()["trust_score"]

    def score(row):
        wsum, wtot = 0.0, 0.0
        r = row.get("rating_overall")
        if pd.notna(r) and float(r) > 0:
            wsum += float(r) * cfg["overall_rating_weight"]
            wtot += cfg["overall_rating_weight"]
        vals = [float(row[a]) for a in
                ["rating_service","rating_amenities","rating_food",
                 "rating_value","rating_location","rating_cleanliness"]
                if pd.notna(row.get(a)) and row.get(a) > 0]
        if vals:
            wsum += np.mean(vals) * cfg["aspect_rating_weight"]
            wtot += cfg["aspect_rating_weight"]
        if wtot == 0: return np.nan
        rc    = row.get("review_count")
        boost = min(cfg["review_volume_boost"],
                    (np.log10(float(rc)) / np.log10(1000)) * cfg["review_volume_boost"]) \
                if pd.notna(rc) and float(rc) > 1 else 0.0
        return round(min(10.0, (wsum / wtot + boost) * 2), 2)

    df = df.copy()
    df["trust_score"] = df.apply(score, axis=1)
    logger.info(f"  Trust scores: {df['trust_score'].notna().sum():,}/{len(df):,} scored")
    return df


def add_review_text(df: pd.DataFrame) -> pd.DataFrame:
    def build(row):
        parts = []
        for lbl, c in [("Hotel","hotel_name"),("City","city"),("Area","area"),
                       ("Type","hotel_type"),("Stars","star_rating"),
                       ("Description","description"),("Amenities","amenities"),
                       ("Facilities","facilities"),("Rooms","room_facilities")]:
            if row.get(c): parts.append(f"{lbl}: {row[c]}")
        asp = [(k, row.get(v)) for k, v in
               [("Service","rating_service"),("Amenities","rating_amenities"),
                ("Food","rating_food"),("Value","rating_value"),
                ("Location","rating_location"),("Cleanliness","rating_cleanliness")]]
        asp_str = [f"{k} {v}/5" for k, v in asp if pd.notna(v)]
        if asp_str: parts.append("Ratings — " + ", ".join(asp_str))
        if pd.notna(row.get("trust_score")): parts.append(f"Trust: {row['trust_score']}/10")
        return " | ".join(parts)
    df = df.copy()
    df["review_text"] = df.apply(build, axis=1)
    return df


def run_merge(standardized: dict) -> pd.DataFrame:
    logger.info("\n" + "=" * 55)
    logger.info("  STEP 1D — MERGE + TRUST SCORE + EXPORT")
    logger.info("=" * 55)

    frames = [df for df in standardized.values() if not df.empty]
    if not frames:
        logger.error("No data to merge.")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"  Merged: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    merged = deduplicate(merged)
    merged = add_trust_scores(merged)
    merged = add_review_text(merged)
    merged.sort_values("trust_score", ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    cfg     = load_config()
    out_dir = cfg["paths"]["processed_data"]

    # Save unified CSV
    csv_path = out_dir / "unified_hotels.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"  ✅ Saved unified_hotels.csv ({merged.shape[0]:,} rows × {merged.shape[1]} cols)")

    # Save report
    report = _make_report(merged)
    rpt_path = out_dir / "pipeline_report.txt"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("\n" + report)
    return merged


def _make_report(df: pd.DataFrame) -> str:
    lines = [
        "=" * 58,
        "  STAYWISE AI — PIPELINE REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 58,
        f"\n  Total hotels  : {len(df):,}",
        f"  Total columns : {df.shape[1]}",
        f"  Unique cities : {df['city'].nunique()}",
        "\n── By Source ──────────────────────────────────────",
    ]
    for src, grp in df.groupby("source"):
        lines.append(f"   {src:<16}: {len(grp):>6,} hotels")
    lines += ["\n── Top 10 Cities ──────────────────────────────────"]
    for city, cnt in df.groupby("city").size().sort_values(ascending=False).head(10).items():
        lines.append(f"   {city:<22}: {cnt:>5,}")
    lines += ["\n── Data Coverage ──────────────────────────────────"]
    for col in ["rating_overall","price_inr","star_rating","rating_cleanliness","trust_score"]:
        n = df[col].notna().sum()
        lines.append(f"   {col:<24}: {n:>6,}/{len(df):,} ({n/len(df)*100:.0f}%)")
    if df["price_inr"].notna().any():
        lines += ["\n── Price (INR) ─────────────────────────────────────",
                  f"   Min    : ₹{df['price_inr'].min():>10,.0f}",
                  f"   Median : ₹{df['price_inr'].median():>10,.0f}",
                  f"   Max    : ₹{df['price_inr'].max():>10,.0f}"]
    lines += ["\n" + "=" * 58,
              "  ✅  STEP 1 COMPLETE — Ready for Step 2: RAG",
              "=" * 58]
    return "\n".join(lines)
