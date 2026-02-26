"""
Provides quality-assurance checks for all c9r Parquet stores.

Covers schema conformance, null/empty rates, timestamp integrity,
duplicate detection, MBID validation, row-count reconciliation,
and character-encoding sanity.
"""
from __future__ import annotations
import logging
import re
from datetime import datetime, UTC
from typing import Any
import pandas as pd
from helpers.io import (
    SCROBBLE_COLS, ARTIST_INFO_COLS, AVC_COLS, GS_MB_COLS,
    SCROBBLE_PQ, ARTIST_INFO_PQ, AVC_PQ, UC_PQ, GS_MB_PQ, QA_REPORT_PQ,
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


def _check_schema(
    df: pd.DataFrame,
    expected_cols: list[str] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Verifies that the DataFrame has the expected columns.

    When *strict* is True (the default), extra columns also cause a failure.
    When False, only missing columns are flagged.
    """
    expected = set(expected_cols or SCROBBLE_COLS)
    actual = set(df.columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if strict:
        ok = actual == expected
    else:
        ok = not missing
    return {"pass": ok, "missing": missing, "unexpected": unexpected}


def _check_nulls(df: pd.DataFrame, cols: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Reports per-column null and empty-string rates."""
    cols = cols or SCROBBLE_COLS
    return {col: _null_and_empty(df[col]) for col in cols if col in df.columns}


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
        # Localizing to UTC so the remaining range checks can proceed
        col = col.dt.tz_localize("UTC")
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


def _check_mbids(df: pd.DataFrame, col: str = "artist_mbid") -> dict[str, Any]:
    """Reports MBID fill rate and UUID-validity rate for *col*."""
    if col not in df.columns or df.empty:
        return {"fill_rate": 0.0, "valid_rate": 0.0, "total": 0}
    total = len(df)
    non_null = df[col].dropna()
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


def _check_encoding(df: pd.DataFrame, text_cols: list[str] | None = None) -> dict[str, Any]:
    """Scans string columns for replacement chars and control characters."""
    if text_cols is None:
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
    source: str | None = None,
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
    source : str, optional
        Data origin label, e.g. ``"lastfm"`` or ``"listenbrainz"``.

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
        "source": source,
        "target": "scrobble",
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
    enrichment = report.get("enrichment", {})
    row = pd.DataFrame([{
        "timestamp": report["timestamp"],
        "source": report.get("source"),
        "target": report.get("target", "scrobble"),
        "row_count": report["row_count"],
        "passed": report["passed"],
        "schema_ok": report.get("schema", {}).get("pass", True),
        "artist_null_pct": nulls.get("artist_name", {}).get("null_pct", 0),
        "track_null_pct": nulls.get("track_title", {}).get("null_pct", 0),
        "album_null_pct": nulls.get("album_title", {}).get("null_pct", 0),
        "mbid_fill_rate": report.get("mbids", {}).get("fill_rate", 0),
        "mbid_valid_rate": report.get("mbids", {}).get("valid_rate", 0),
        "hash_fill_rate": report.get("hash_fill", {}).get("fill_rate", 0),
        "unique_countries": report.get("unique_countries"),
        "duplicate_count": report.get("duplicates", {}).get("duplicate_count", 0),
        "duplicate_pct": report.get("duplicates", {}).get("duplicate_pct", 0),
        "ts_before_min": report.get("timestamps", {}).get("before_min_count", 0),
        "ts_after_now": report.get("timestamps", {}).get("after_now_count", 0),
        "bad_char_rows": report.get("encoding", {}).get("bad_char_rows", 0),
        "fetched": report.get("reconciliation", {}).get("fetched"),
        "stored": report.get("reconciliation", {}).get("stored"),
        "country_fill_rate": enrichment.get("country", {}).get("fill_rate"),
        "disambiguation_fill_rate": enrichment.get("disambiguation", {}).get("fill_rate"),
        "aliases_fill_rate": enrichment.get("aliases", {}).get("fill_rate"),
    }])
    append_to_parquet(row, QA_REPORT_PQ)
    log.info("QA report appended to %s", QA_REPORT_PQ)


# ── Real-fill helpers ─────────────────────────────────────────────────────────
def _real_fill(series: pd.Series, total: int) -> dict[str, Any]:
    """
    Counts values that are genuinely filled — not None, NaN, empty, or
    the literal string ``"None"``.
    """
    s = series.fillna("").astype(str).str.strip()
    mask = (s != "") & (s.str.lower() != "none")
    filled = int(mask.sum())
    return {
        "filled": filled,
        "fill_rate": round(filled / total * 100, 2) if total else 0.0,
    }


# ── artist_info QA ────────────────────────────────────────────────────────────
def qa_artist_info(*, source: str | None = None) -> dict[str, Any]:
    """
    Runs QA checks on artist_info.parquet.

    Checks schema, null/empty rates, MBID validity, duplicates,
    encoding on text columns, and real-fill enrichment rates for
    country, disambiguation_comment, and aliases.
    """
    df = read_parquet(ARTIST_INFO_PQ)
    if df is None or df.empty:
        log.warning("artist_info.parquet is empty or missing — QA skipped.")
        return {"status": "skipped", "reason": "no data"}
    total = len(df)
    schema = _check_schema(df, ARTIST_INFO_COLS, strict=False)
    nulls = _check_nulls(df, ARTIST_INFO_COLS)
    mbids = _check_mbids(df, "mbid")
    # Duplicating on artist_name
    dup_cols = ["artist_name"]
    dup_mask = df.duplicated(subset=dup_cols, keep="first")
    dup_count = int(dup_mask.sum())
    dup_pct = round(dup_count / total * 100, 2) if total else 0.0
    duplicates = {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "pass": (dup_pct / 100) <= DUPLICATE_RATE_THRESHOLD,
    }
    encoding = _check_encoding(df, ["artist_name", "disambiguation_comment", "aliases"])
    # Computing real-fill enrichment rates
    enrichment = {
        "country": _real_fill(df["country"], total) if "country" in df.columns else {"filled": 0, "fill_rate": 0.0},
        "disambiguation": _real_fill(df["disambiguation_comment"], total) if "disambiguation_comment" in df.columns else {"filled": 0, "fill_rate": 0.0},
        "aliases": _real_fill(df["aliases"], total) if "aliases" in df.columns else {"filled": 0, "fill_rate": 0.0},
    }
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "target": "artist_info",
        "row_count": total,
        "schema": schema,
        "nulls": nulls,
        "duplicates": duplicates,
        "mbids": mbids,
        "encoding": encoding,
        "enrichment": enrichment,
    }
    report["passed"] = all([
        schema["pass"],
        duplicates["pass"],
        encoding["pass"],
        nulls.get("artist_name", {}).get("null_pct", 0) == 0,
    ])
    _persist_report(report)
    return report


# ── avc QA ────────────────────────────────────────────────────────────────────
def qa_avc(*, source: str | None = None) -> dict[str, Any]:
    """
    Runs QA checks on avc.parquet.

    Checks schema, null/empty rates, duplicate hashes,
    timestamp validity, and encoding on text columns.
    """
    df = read_parquet(AVC_PQ)
    if df is None or df.empty:
        log.warning("avc.parquet is empty or missing — QA skipped.")
        return {"status": "skipped", "reason": "no data"}
    total = len(df)
    schema = _check_schema(df, AVC_COLS, strict=False)
    nulls = _check_nulls(df, AVC_COLS)
    # Duplicating on artist_variants_hash
    dup_cols = ["artist_variants_hash"]
    present = [c for c in dup_cols if c in df.columns]
    if present:
        dup_mask = df.duplicated(subset=present, keep="first")
        dup_count = int(dup_mask.sum())
        dup_pct = round(dup_count / total * 100, 2) if total else 0.0
    else:
        dup_count, dup_pct = 0, 0.0
    duplicates = {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "pass": (dup_pct / 100) <= DUPLICATE_RATE_THRESHOLD,
    }
    # Validating stamp timestamps
    timestamps = {"pass": True, "issues": [], "before_min_count": 0, "after_now_count": 0}
    if "stamp" in df.columns and not df["stamp"].dropna().empty:
        col = df["stamp"].dropna()
        if pd.api.types.is_datetime64_any_dtype(col):
            if col.dt.tz is None:
                timestamps["pass"] = False
                timestamps["issues"].append("stamp is not timezone-aware")
        else:
            timestamps["pass"] = False
            timestamps["issues"].append(f"stamp has non-datetime dtype: {col.dtype}")
    encoding = _check_encoding(df, ["artist_variants_text", "canonical_name", "comment"])
    # Computing hash fill rate
    if "artist_variants_hash" in df.columns:
        hash_filled = int(df["artist_variants_hash"].notna().sum())
        hash_fill_rate = round(hash_filled / total * 100, 2) if total else 0.0
    else:
        hash_filled, hash_fill_rate = 0, 0.0
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "target": "artist_variants_canonized",
        "row_count": total,
        "schema": schema,
        "nulls": nulls,
        "duplicates": duplicates,
        "timestamps": timestamps,
        "encoding": encoding,
        "hash_fill": {"filled": hash_filled, "fill_rate": hash_fill_rate},
    }
    report["passed"] = all([
        schema["pass"],
        duplicates["pass"],
        timestamps["pass"],
        encoding["pass"],
    ])
    _persist_report(report)
    return report


# ── gs_mb QA ──────────────────────────────────────────────────────────────────
def qa_gs_mb(*, source: str | None = None) -> dict[str, Any]:
    """
    Runs QA checks on gs_mb.parquet (MusicBrainz gold-standard pairs).

    Checks schema, null/empty rates, duplicate pairs, encoding on
    text columns, and to_link / source distribution.
    """
    df = read_parquet(GS_MB_PQ)
    if df is None or df.empty:
        log.warning("gs_mb.parquet is empty or missing — QA skipped.")
        return {"status": "skipped", "reason": "no data"}
    total = len(df)
    schema = _check_schema(df, GS_MB_COLS, strict=False)
    nulls = _check_nulls(df, GS_MB_COLS)
    # Duplicating on (variant_a, variant_b)
    dup_cols = ["variant_a", "variant_b"]
    present = [c for c in dup_cols if c in df.columns]
    if present:
        dup_mask = df.duplicated(subset=present, keep="first")
        dup_count = int(dup_mask.sum())
        dup_pct = round(dup_count / total * 100, 2) if total else 0.0
    else:
        dup_count, dup_pct = 0, 0.0
    duplicates = {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "pass": (dup_pct / 100) <= DUPLICATE_RATE_THRESHOLD,
    }
    encoding = _check_encoding(df, ["variant_a", "variant_b"])
    # Computing to_link distribution
    if "to_link" in df.columns:
        pos = int((df["to_link"] == True).sum())   # noqa: E712
        neg = int((df["to_link"] == False).sum())   # noqa: E712
        null_link = int(df["to_link"].isna().sum())
    else:
        pos, neg, null_link = 0, 0, 0
    label_dist = {"positive": pos, "negative": neg, "null": null_link}
    # Computing source breakdown
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()
    else:
        source_counts = {}
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "target": "gs_mb",
        "row_count": total,
        "schema": schema,
        "nulls": nulls,
        "duplicates": duplicates,
        "encoding": encoding,
        "label_distribution": label_dist,
        "source_breakdown": source_counts,
    }
    report["passed"] = all([
        schema["pass"],
        duplicates["pass"],
        encoding["pass"],
        nulls.get("variant_a", {}).get("null_pct", 0) == 0,
        nulls.get("variant_b", {}).get("null_pct", 0) == 0,
    ])
    _persist_report(report)
    return report


# ── uc QA ─────────────────────────────────────────────────────────────────────
def qa_uc(*, source: str | None = None) -> dict[str, Any]:
    """
    Produces a summary for uc.parquet (user-country history).

    Reports total entries and distinct country codes.
    """
    df = read_parquet(UC_PQ)
    if df is None or df.empty:
        log.warning("uc.parquet is empty or missing — QA skipped.")
        return {"status": "skipped", "reason": "no data"}
    total = len(df)
    unique_countries = int(df["country_code"].nunique()) if "country_code" in df.columns else 0
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "target": "user_country",
        "row_count": total,
        "unique_countries": unique_countries,
        "passed": True,
    }
    _persist_report(report)
    return report
