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
SCROBBLE_PQ = PQ_DIR / "scrobble.parquet"  # legacy single-file path
SCROBBLE_PQ_DIR = PQ_DIR / "scrobble"  # year-partitioned directory
ARTIST_INFO_PQ = PQ_DIR / "artist_info.parquet"
AVC_PQ = PQ_DIR / "avc.parquet"
C_PQ = PQ_DIR / "c.parquet"
UC_PQ = PQ_DIR / "uc.parquet"
QA_REPORT_PQ = PQ_DIR / "qa_report.parquet"
GS_MB_PQ = PQ_DIR / "gs_mb.parquet"
ALIAS_SEP = "{"

# ── PyArrow schemas (canonical definitions live in helpers.schema) ─────────────
from helpers.schema import (  # noqa: E402 — re-exported for backward compat
    stamp_metadata as _stamp_metadata,
    table_name_for_path as _table_name_for_path,
    read_file_version as _read_file_version,
    current_version as _current_version,
)

# ── Column aliases (Last.fm API → c9r canonical names) ────────────────────────
_COLUMN_ALIASES = {
    "Artist": "artist_name",
    "Album": "album_title",
    "Song": "track_title",
    "artist mbid": "artist_mbid",
    "mbid": "artist_mbid",
}
SCROBBLE_COLS = ["artist_name", "album_title", "track_title", "artist_mbid", "play_time"]
ARTIST_INFO_COLS = ["artist_name", "mbid", "country", "disambiguation_comment", "aliases"]
AVC_COLS = ["artist_variants_hash", "artist_variants_text", "canonical_name", "to_link", "comment", "stamp"]
UC_COLS = ["country_code", "start_date", "end_date"]
GS_MB_COLS = ["variant_a", "variant_b", "to_link", "source"]

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
    """Reads a Parquet file or returns None when missing.

    Checks embedded schema version: warns when stale, raises
    ``RuntimeError`` when the file is newer than the running code.
    """
    if not path.exists():
        return None
    # Checking schema version metadata
    tbl_name, file_ver = _read_file_version(path)
    if tbl_name is None:
        tbl_name = _table_name_for_path(path)
    if tbl_name is not None:
        cur = _current_version(tbl_name)
        if cur and file_ver > cur:
            raise RuntimeError(
                f"{path.name} has schema v{file_ver} but this code only "
                f"supports {tbl_name} up to v{cur}. Please upgrade c9r."
            )
        if cur and file_ver < cur:
            logger.warning(
                "%s is at schema v%d (current: v%d) — run "
                "'c9r schema migrate' to update.",
                path.name, file_ver, cur,
            )
    return pd.read_parquet(path)


def dump_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str = "zstd",
) -> Path:
    """Overwrites *path* with *df* as a Parquet file.

    Automatically stamps c9r schema version metadata for recognised tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    tbl_name = _table_name_for_path(path)
    if tbl_name:
        table = _stamp_metadata(table, tbl_name)
    pq.write_table(table, path, compression=compression)
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
        # Dropping all-NA columns before concat to avoid FutureWarning
        parts = [d.dropna(axis=1, how="all") for d in (existing, df) if not d.empty]
        combined = pd.concat(parts, ignore_index=True)
        if dedup_cols:
            combined = combined.drop_duplicates(subset=list(dedup_cols), keep="last")
        # Stamping schema version metadata for recognised tables
        table = pa.Table.from_pandas(combined, preserve_index=False)
        tbl_name = _table_name_for_path(path)
        if tbl_name:
            table = _stamp_metadata(table, tbl_name)
        pq.write_table(table, path, compression=compression)
        logger.info(
            "Appended %d rows → %s (total: %d rows)",
            len(df), path, len(combined),
        )
    else:
        if dedup_cols:
            df = df.drop_duplicates(subset=list(dedup_cols), keep="last")
        table = pa.Table.from_pandas(df, preserve_index=False)
        tbl_name = _table_name_for_path(path)
        if tbl_name:
            table = _stamp_metadata(table, tbl_name)
        pq.write_table(table, path, compression=compression)
        logger.info("Created %s with %d rows", path, len(df))
    return path


# ── Scrobble year-partitioned helpers ─────────────────────────────────────────
def scrobble_data_exists() -> bool:
    """Returns True when scrobble data is present (partitioned or legacy)."""
    if SCROBBLE_PQ_DIR.exists() and any(SCROBBLE_PQ_DIR.rglob("*.parquet")):
        return True
    return SCROBBLE_PQ.exists()


def read_scrobble_df() -> pd.DataFrame | None:
    """Reads scrobble data from partitioned dir, falling back to legacy file.

    Drops the synthetic ``year`` partition column so callers always
    receive the canonical 5-column schema.
    """
    if SCROBBLE_PQ_DIR.exists() and any(SCROBBLE_PQ_DIR.rglob("*.parquet")):
        df = pd.read_parquet(SCROBBLE_PQ_DIR)
        if "year" in df.columns:
            df = df.drop(columns=["year"])
        return df if not df.empty else None
    return read_parquet(SCROBBLE_PQ)


def scrobble_duckdb_from() -> str:
    """Returns a DuckDB-compatible source expression for scrobble data.

    Produces a ``read_parquet(…, hive_partitioning=true)`` call for the
    partitioned layout, or a plain quoted path for the legacy file.
    """
    if SCROBBLE_PQ_DIR.exists() and any(SCROBBLE_PQ_DIR.rglob("*.parquet")):
        return (
            f"read_parquet('{SCROBBLE_PQ_DIR.as_posix()}/**/*.parquet',"
            f" hive_partitioning=true)"
        )
    return f"'{SCROBBLE_PQ.as_posix()}'"


def dump_scrobble_df(df: pd.DataFrame) -> Path:
    """Overwrites the year-partitioned scrobble store with *df*.

    Used by write-back operations (encoding fix, MBID backfill) that
    modify the full DataFrame in memory and need to persist it.
    """
    if df.empty:
        logger.info("Empty DataFrame — nothing to write to %s", SCROBBLE_PQ_DIR)
        return SCROBBLE_PQ_DIR
    df = df.copy()
    df["year"] = df["play_time"].dt.year
    SCROBBLE_PQ_DIR.mkdir(parents=True, exist_ok=True)
    for year, group in df.groupby("year"):
        year_dir = SCROBBLE_PQ_DIR / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part_path = year_dir / "part.parquet"
        table = pa.Table.from_pandas(
            group.drop(columns=["year"]), preserve_index=False,
        )
        table = _stamp_metadata(table, "scrobble")
        pq.write_table(table, part_path, compression="zstd")
    logger.info("Wrote %d rows → %s (partitioned by year)", len(df), SCROBBLE_PQ_DIR)
    return SCROBBLE_PQ_DIR


def migrate_scrobble_to_partitioned() -> int:
    """Converts the legacy single-file scrobble.parquet to year-partitioned layout.

    Leaves the legacy file in place (caller may remove it afterward).
    Returns the number of rows migrated.
    """
    if not SCROBBLE_PQ.exists():
        logger.info("No legacy scrobble.parquet to migrate.")
        return 0
    df = pd.read_parquet(SCROBBLE_PQ)
    if df.empty:
        return 0
    dump_scrobble_df(df)
    logger.info(
        "Migrated %d scrobbles from %s → %s",
        len(df), SCROBBLE_PQ, SCROBBLE_PQ_DIR,
    )
    return len(df)


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
    """Normalises *df* and appends to year-partitioned scrobble store.

    Each year partition is deduped independently to avoid loading
    the entire history into memory.
    """
    normalised = normalise_scrobble_df(df)
    if normalised.empty:
        return 0
    normalised["year"] = normalised["play_time"].dt.year
    SCROBBLE_PQ_DIR.mkdir(parents=True, exist_ok=True)
    dedup_cols = ["artist_name", "track_title", "play_time"]
    for year, group in normalised.groupby("year"):
        year_dir = SCROBBLE_PQ_DIR / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part_file = year_dir / "part.parquet"
        data = group.drop(columns=["year"])
        if part_file.exists():
            existing = pd.read_parquet(part_file)
            combined = pd.concat([existing, data], ignore_index=True)
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            table = pa.Table.from_pandas(combined, preserve_index=False)
            table = _stamp_metadata(table, "scrobble")
            pq.write_table(table, part_file, compression="zstd")
        else:
            data = data.drop_duplicates(subset=dedup_cols, keep="last")
            table = pa.Table.from_pandas(data, preserve_index=False)
            table = _stamp_metadata(table, "scrobble")
            pq.write_table(table, part_file, compression="zstd")
    logger.info("Ingested %d scrobbles (partitioned by year)", len(normalised))
    return len(normalised)


def latest_scrobble_ts() -> int | None:
    """Returns the Unix timestamp of the newest scrobble, or None if empty."""
    existing = read_scrobble_df()
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
