"""
Provides Parquet-native I/O for c9r.

All persistence goes through this module.  Parquet files live under PQ_DIR
(default ``PQ/`` relative to project root) and use zstd compression.
Application-layer deduplication is performed before every append.
"""
from __future__ import annotations
import logging
import os
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from typing import Sequence
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
if "__file__" in globals():
    _HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = _HERE.parent
else:
    PROJECT_ROOT = Path.cwd()
PQ_DIR = Path(os.getenv("PQ_DIR", str(PROJECT_ROOT / "PQ")))
PQ_DIR.mkdir(exist_ok=True)
SCROBBLE_PQ = PQ_DIR / "scrobble.parquet"
ARTIST_INFO_PQ = PQ_DIR / "artist_info.parquet"
AVC_PQ = PQ_DIR / "avc.parquet"
C_PQ = PQ_DIR / "c.parquet"
UC_PQ = PQ_DIR / "uc.parquet"
QA_REPORT_PQ = PQ_DIR / "qa_report.parquet"

# ── PyArrow schemas ───────────────────────────────────────────────────────────
SCROBBLE_SCHEMA = pa.schema([
    ("artist_name", pa.string()),
    ("album_title", pa.string()),
    ("track_title", pa.string()),
    ("artist_mbid", pa.string()),
    ("play_time", pa.timestamp("us", tz="UTC")),
])
ARTIST_INFO_SCHEMA = pa.schema([
    ("artist_name", pa.string()),
    ("mbid", pa.string()),
    ("country", pa.string()),
    ("disambiguation_comment", pa.string()),
    ("aliases", pa.string()),
])
AVC_SCHEMA = pa.schema([
    ("artist_variants_hash", pa.string()),
    ("artist_variants_text", pa.string()),
    ("canonical_name", pa.string()),
    ("to_link", pa.bool_()),
    ("comment", pa.string()),
    ("stamp", pa.timestamp("us", tz="UTC")),
])

# ── Column aliases (Last.fm API → c9r canonical names) ────────────────────────
_COLUMN_ALIASES = {
    "Artist": "artist_name",
    "Album": "album_title",
    "Song": "track_title",
    "artist mbid": "artist_mbid",
    "mbid": "artist_mbid",
}
SCROBBLE_COLS = ["artist_name", "album_title", "track_title", "artist_mbid", "play_time"]

# Operator tokens for sanitising column names
OP_TOKENS = {
    r"\s\-\s": "_minus_",
    r"\s\+\s": "_plus_",
    r"\s\*\s": "_mul_",
    r"\s\/\s": "_div_",
}

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_UUID_RE = UUID_RE  # to keep backward compat for internal references


# ── Core I/O helpers ──────────────────────────────────────────────────────────
def read_parquet(path: Path) -> pd.DataFrame | None:
    """Reads a Parquet file or returns None when missing."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def dump_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str = "zstd",
) -> Path:
    """Overwrites *path* with *df* as a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)
    logger.info("Wrote %d rows → %s", len(df), path)
    return path


def append_to_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    dedup_cols: Sequence[str] | None = None,
    compression: str = "zstd",
) -> Path:
    """
    Appends *df* to an existing Parquet file with deduplication.

    If *path* does not exist yet, creates it.  When *dedup_cols* is given,
    duplicates (keeping last) are dropped after concatenation.
    """
    if df.empty:
        logger.info("Empty DataFrame — nothing to append to %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        if dedup_cols:
            combined = combined.drop_duplicates(subset=list(dedup_cols), keep="last")
        combined.to_parquet(path, index=False, compression=compression)
        logger.info(
            "Appended %d rows → %s (total: %d rows)",
            len(df), path, len(combined),
        )
    else:
        if dedup_cols:
            df = df.drop_duplicates(subset=list(dedup_cols), keep="last")
        df.to_parquet(path, index=False, compression=compression)
        logger.info("Created %s with %d rows", path, len(df))
    return path


# ── Scrobble-specific helpers ─────────────────────────────────────────────────
def normalise_scrobble_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises a raw scrobble DataFrame into the canonical schema.

    Renames columns, converts UTS → UTC datetime, cleans MBIDs,
    deduplicates, and selects only the standard columns.
    """
    df = df.rename(columns=_COLUMN_ALIASES, errors="ignore")
    # Handling timestamp conversion
    if "uts" in df.columns:
        df["play_time"] = pd.to_datetime(df["uts"].astype(int), unit="s", utc=True)
    elif "Timestamp" in df.columns:
        df["play_time"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("UTC")
    # Cleaning artist_mbid
    if "artist_mbid" not in df.columns:
        df["artist_mbid"] = None
    else:
        df["artist_mbid"] = (
            df["artist_mbid"]
            .astype(str)
            .str.strip()
            .where(lambda s: s.str.match(_UUID_RE))
        )
    # Ensuring all expected columns exist
    for col in SCROBBLE_COLS:
        if col not in df.columns:
            df[col] = None
    # Deduplicating and selecting final columns
    df = df.drop_duplicates(subset=["artist_name", "track_title", "play_time"])
    return df[SCROBBLE_COLS].copy()


def ingest_scrobbles(df: pd.DataFrame) -> int:
    """
    Normalises *df* and appends to scrobble.parquet with deduplication.

    Returns the number of rows in the incoming DataFrame.
    """
    normalised = normalise_scrobble_df(df)
    append_to_parquet(
        normalised,
        SCROBBLE_PQ,
        dedup_cols=["artist_name", "track_title", "play_time"],
    )
    return len(normalised)


def latest_scrobble_ts() -> int | None:
    """Returns the Unix timestamp of the newest scrobble, or None if empty."""
    existing = read_parquet(SCROBBLE_PQ)
    if existing is None or existing.empty:
        return None
    return int(existing["play_time"].max().timestamp())


# ── Palette registration (kept for BI frontend) ──────────────────────────────
def register_custom_palette(palette_name: str, palettes: list[dict]) -> list[str]:
    """Registers a custom Seaborn palette from a palettes.json structure."""
    import seaborn as sns
    palette = next((p for p in palettes if p["paletteName"] == palette_name), None)
    if not palette:
        raise ValueError(f"Palette {palette_name} not found in the JSON file.")
    colors = [
        f"#{color['hex']}" if not color["hex"].startswith("#") else color["hex"]
        for color in sorted(palette["colors"], key=lambda x: x["position"])
    ]
    sns.set_palette(sns.color_palette(colors))
    return colors


# ── Column-name sanitiser (for ML feature columns) ───────────────────────────
def sanitize(col: str, seen: Counter) -> str:
    """
    Turns 'partial_ratio - QRatio' → 'partial_ratio_minus_QRatio'.

    Guarantees each result is a valid Python identifier and unique.
    """
    safe = col
    for pat, tok in OP_TOKENS.items():
        safe = re.sub(pat, tok, safe)
    safe = re.sub(r"\W+", "_", safe).strip("_")
    count = seen[safe]
    seen[safe] += 1
    if count:
        safe = f"{safe}_{count}"
    return safe
