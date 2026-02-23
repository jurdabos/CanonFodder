"""
Provides post-ingestion quality-assurance checks for scrobble data.

Covers schema conformance, null/empty rates, timestamp integrity,
duplicate detection, MBID validation, row-count reconciliation,
and character-encoding sanity.
"""
from __future__ import annotations
import logging
import re
import unicodedata
from datetime import datetime, UTC
from typing import Any
import pandas as pd
from helpers.io import (
    SCROBBLE_COLS, SCROBBLE_PQ, QA_REPORT_PQ,
    UUID_RE, read_parquet, append_to_parquet,
)

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
ALBUM_NULL_THRESHOLD = 0.30        # up to 30 % album missingness is normal for LB
DUPLICATE_RATE_THRESHOLD = 0.05    # > 5 % duplicates warrants a warning
MIN_PLAUSIBLE_YEAR = 2000
# Matching Unicode replacement char or C0/C1 control chars (except \n \r \t)
_BAD_CHAR_RE = re.compile(r"[\uFFFD\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def _null_and_empty(series: pd.Series) -> dict[str, Any]:
    """Returns null count, empty-string count, and their rates for a single column."""
    total = len(series)
    null_count = int(series.isna().sum())
    empty_count = int((series.fillna("").astype(str) == "").sum()) - null_count
    empty_count = max(empty_count, 0)
    return {
        "null_count": null_count,
        "null_pct": round(null_count / total * 100, 2) if total else 0.0,
        "empty_count": empty_count,
        "empty_pct": round(empty_count / total * 100, 2) if total else 0.0,
    }


def _check_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Verifies that the DataFrame has exactly the expected scrobble columns."""
    expected = set(SCROBBLE_COLS)
    actual = set(df.columns)
    return {
        "pass": actual == expected,
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
    }


def _check_nulls(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Reports per-column null and empty-string rates."""
    return {col: _null_and_empty(df[col]) for col in SCROBBLE_COLS if col in df.columns}


def _check_timestamps(df: pd.DataFrame) -> dict[str, Any]:
    """Validates play_time is tz-aware UTC within a plausible range."""
    result: dict[str, Any] = {"pass": True, "issues": []}
    if "play_time" not in df.columns or df.empty:
        result["pass"] = False
        result["issues"].append("play_time column missing or DataFrame empty")
        return result
    col = df["play_time"]
    # Checking timezone awareness
    if col.dt.tz is None:
        result["pass"] = False
        result["issues"].append("play_time is not timezone-aware")
    # Checking plausible range
    now = pd.Timestamp.now(tz="UTC")
    min_ts = pd.Timestamp(f"{MIN_PLAUSIBLE_YEAR}-01-01", tz="UTC")
    before_min = int((col < min_ts).sum())
    after_now = int((col > now).sum())
    if before_min:
        result["pass"] = False
        result["issues"].append(f"{before_min} timestamps before {MIN_PLAUSIBLE_YEAR}")
    if after_now:
        result["pass"] = False
        result["issues"].append(f"{after_now} timestamps in the future")
    result["before_min_count"] = before_min
    result["after_now_count"] = after_now
    # Checking monotonicity after sort
    sorted_ts = col.sort_values()
    non_monotonic = int((sorted_ts.diff().dt.total_seconds().dropna() < 0).sum())
    result["non_monotonic_count"] = non_monotonic
    if non_monotonic:
        result["issues"].append(f"{non_monotonic} non-monotonic gaps after sort")
    return result


def _check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Counts rows sharing the dedup key (artist_name, track_title, play_time)."""
    dedup_cols = ["artist_name", "track_title", "play_time"]
    present = [c for c in dedup_cols if c in df.columns]
    if len(present) < len(dedup_cols):
        return {"duplicate_count": 0, "duplicate_pct": 0.0, "pass": True}
    dup_mask = df.duplicated(subset=dedup_cols, keep="first")
    dup_count = int(dup_mask.sum())
    total = len(df)
    dup_pct = round(dup_count / total * 100, 2) if total else 0.0
    return {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "pass": (dup_pct / 100) <= DUPLICATE_RATE_THRESHOLD,
    }


def _check_mbids(df: pd.DataFrame) -> dict[str, Any]:
    """Reports MBID fill rate and UUID-validity rate."""
    if "artist_mbid" not in df.columns or df.empty:
        return {"fill_rate": 0.0, "valid_rate": 0.0, "total": 0}
    total = len(df)
    non_null = df["artist_mbid"].dropna()
    non_empty = non_null[non_null.astype(str).str.strip() != ""]
    filled = len(non_empty)
    valid = int(non_empty.astype(str).str.match(UUID_RE).sum())
    return {
        "total": total,
        "filled": filled,
        "fill_rate": round(filled / total * 100, 2) if total else 0.0,
        "valid": valid,
        "valid_rate": round(valid / filled * 100, 2) if filled else 0.0,
    }


def _check_encoding(df: pd.DataFrame) -> dict[str, Any]:
    """Scans string columns for replacement chars and control characters."""
    text_cols = ["artist_name", "album_title", "track_title"]
    bad_rows = 0
    bad_examples: list[str] = []
    for col in text_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        mask = series.str.contains(_BAD_CHAR_RE, na=False)
        n = int(mask.sum())
        bad_rows += n
        if n and len(bad_examples) < 5:
            bad_examples.extend(series[mask].head(3).tolist())
    return {
        "bad_char_rows": bad_rows,
        "pass": bad_rows == 0,
        "examples": bad_examples[:5],
    }


def _reconcile_rows(df: pd.DataFrame, fetched_count: int | None) -> dict[str, Any]:
    """Compares fetched count against rows in the DataFrame."""
    stored = len(df)
    if fetched_count is None:
        return {"fetched": None, "stored": stored, "pass": True}
    diff = fetched_count - stored
    diff_pct = round(abs(diff) / fetched_count * 100, 2) if fetched_count else 0.0
    return {
        "fetched": fetched_count,
        "stored": stored,
        "diff": diff,
        "diff_pct": diff_pct,
        "pass": diff_pct <= DUPLICATE_RATE_THRESHOLD * 100,
    }


# ── Main entry point ─────────────────────────────────────────────────────────
def qa_lb_ingest(
    *,
    fetched_count: int | None = None,
    last_n_hours: int | None = None,
) -> dict[str, Any]:
    """
    Runs all QA checks on scrobble.parquet and persists results.

    Parameters
    ----------
    fetched_count : int, optional
        Number of rows the API reported fetching (for reconciliation).
    last_n_hours : int, optional
        When set, restricts the check to scrobbles ingested in the
        last *n* hours.  Otherwise checks the full file.

    Returns
    -------
    dict
        Nested QA results keyed by check name.
    """
    df = read_parquet(SCROBBLE_PQ)
    if df is None or df.empty:
        log.warning("scrobble.parquet is empty or missing — QA skipped.")
        return {"status": "skipped", "reason": "no data"}
    # Optionally filtering to recent window
    if last_n_hours is not None and "play_time" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=last_n_hours)
        df = df[df["play_time"] >= cutoff]
        if df.empty:
            log.info("No scrobbles in the last %d hours — QA skipped.", last_n_hours)
            return {"status": "skipped", "reason": f"no data in last {last_n_hours}h"}
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "row_count": len(df),
        "schema": _check_schema(df),
        "nulls": _check_nulls(df),
        "timestamps": _check_timestamps(df),
        "duplicates": _check_duplicates(df),
        "mbids": _check_mbids(df),
        "reconciliation": _reconcile_rows(df, fetched_count),
        "encoding": _check_encoding(df),
    }
    # Deriving overall pass/fail
    report["passed"] = all([
        report["schema"]["pass"],
        report["timestamps"]["pass"],
        report["duplicates"]["pass"],
        report["encoding"]["pass"],
        report["nulls"].get("artist_name", {}).get("null_pct", 0) == 0,
        report["nulls"].get("track_title", {}).get("null_pct", 0) == 0,
    ])
    # Persisting a flat summary row to qa_report.parquet
    _persist_report(report)
    return report


def _persist_report(report: dict[str, Any]) -> None:
    """Appends a single summary row to PQ/qa_report.parquet."""
    nulls = report.get("nulls", {})
    row = pd.DataFrame([{
        "timestamp": report["timestamp"],
        "row_count": report["row_count"],
        "passed": report["passed"],
        "schema_ok": report["schema"]["pass"],
        "artist_null_pct": nulls.get("artist_name", {}).get("null_pct", 0),
        "track_null_pct": nulls.get("track_title", {}).get("null_pct", 0),
        "album_null_pct": nulls.get("album_title", {}).get("null_pct", 0),
        "mbid_fill_rate": report["mbids"].get("fill_rate", 0),
        "mbid_valid_rate": report["mbids"].get("valid_rate", 0),
        "duplicate_count": report["duplicates"].get("duplicate_count", 0),
        "duplicate_pct": report["duplicates"].get("duplicate_pct", 0),
        "ts_before_min": report["timestamps"].get("before_min_count", 0),
        "ts_after_now": report["timestamps"].get("after_now_count", 0),
        "bad_char_rows": report["encoding"].get("bad_char_rows", 0),
        "fetched": report["reconciliation"].get("fetched"),
        "stored": report["reconciliation"].get("stored"),
    }])
    append_to_parquet(row, QA_REPORT_PQ)
    log.info("QA report appended to %s", QA_REPORT_PQ)
