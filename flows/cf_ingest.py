"""
Prefect flow for the c9r data pipeline.

Orchestrates:
  1. Scrobble ingestion   (FR-01 → FR-03)
  2. Artist enrichment    (FR-04)
  3. Artist-info cleanup  (dedup)

Retries with exponential back-off are configured per task (FR-08).
"""
from __future__ import annotations
import logging
import os
from datetime import UTC, datetime
from typing import Optional
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
load_dotenv()
log = logging.getLogger(__name__)


@task(
    name="fetch_scrobbles",
    description="Fetches new scrobbles and persists to Parquet.",
    retries=8,
    retry_delay_seconds=30,
    retry_jitter_factor=1.5,
    log_prints=True,
)
def fetch_scrobbles(username: str, *, full: bool = False, source: str = "lastfm") -> int:
    """Ingests scrobbles via the workflow helper; returns count."""
    from corefunc.workflow import run_data_gathering_workflow
    from corefunc.qa import qa_lb_ingest
    logger = get_run_logger()
    logger.info("Fetching scrobbles for %s from %s", username, source)
    n = run_data_gathering_workflow(username, full=full, source=source)
    logger.info("Ingested %d scrobbles.", n)
    # Running post-ingestion QA checks
    if n > 0:
        logger.info("Running post-ingestion QA checks …")
        report = qa_lb_ingest(fetched_count=n, last_n_hours=24, source=source)
        if report.get("status") != "skipped":
            if report.get("passed"):
                logger.info("QA passed (%d rows checked).", report["row_count"])
            else:
                logger.warning("QA issues detected:")
                if not report["schema"]["pass"]:
                    logger.warning("  Schema: missing=%s, unexpected=%s", report["schema"]["missing"], report["schema"]["unexpected"])
                ts = report["timestamps"]
                for issue in ts.get("issues", []):
                    logger.warning("  Timestamp: %s", issue)
                dup = report["duplicates"]
                if not dup["pass"]:
                    logger.warning("  Duplicates: %d (%.1f%%)", dup["duplicate_count"], dup["duplicate_pct"])
                enc = report["encoding"]
                if not enc["pass"]:
                    logger.warning("  Encoding: %d rows with bad characters", enc["bad_char_rows"])
    return n


@task(
    name="enrich_artists",
    description="Enriches unknown artists from MusicBrainz.",
    retries=8,
    retry_delay_seconds=30,
    retry_jitter_factor=1.5,
    log_prints=True,
)
def enrich_artists() -> int:
    """Looks up country/MBID for unresolved artists; returns count."""
    from corefunc.enrich import enrich_artist_country
    logger = get_run_logger()
    n = enrich_artist_country()
    logger.info("Enriched %d artists.", n)
    return n


@task(
    name="fix_encoding",
    description="Repairs encoding-corrupted strings in scrobble & artist_info.",
    retries=3,
    retry_delay_seconds=30,
    log_prints=True,
)
def fix_encoding_task() -> dict[str, tuple[int, int]]:
    """Repairs encoding issues; returns per-file (fixed, total) dict."""
    from corefunc.data_cleaning import fix_encoding
    logger = get_run_logger()
    results = fix_encoding()
    for label, (fixed, total) in results.items():
        logger.info("%s: %d rows repaired out of %d.", label, fixed, total)
    return results


@task(
    name="clean_artist_info",
    description="Deduplicates artist_info.parquet.",
    retries=3,
    retry_delay_seconds=30,
    log_prints=True,
)
def clean_artists() -> tuple[int, int]:
    """Deduplicates artist_info; returns (removed, remaining)."""
    from corefunc.data_cleaning import clean_artist_info
    logger = get_run_logger()
    removed, remaining = clean_artist_info()
    logger.info("Cleaned artist_info: removed=%d, remaining=%d.", removed, remaining)
    return removed, remaining


@task(
    name="canonise_batch",
    description="Discovers and flags artist name variant candidates using the ML model.",
    retries=0,
    log_prints=True,
)
def canonise_batch() -> dict[str, int]:
    """Runs batch canonisation; returns counts.

    Loads the persisted LightGBM pipeline in-process (no HTTP dependency).
    Exits gracefully when the model pickle is missing.
    """
    from corefunc.canon.workflow import discover_candidates, write_new_candidates
    logger = get_run_logger()
    try:
        from helpers.inference import load_model
        model = load_model()
    except FileNotFoundError as exc:
        logger.warning("Canonisation skipped — model not available: %s", exc)
        return {"flagged_for_review": 0, "skipped": 0}
    logger.info("Running batch canonisation (%d features).", len(model.feature_names_in_))
    candidates = discover_candidates(model)
    if not candidates:
        logger.info("No new variant candidates found.")
        return {"flagged_for_review": 0, "skipped": 0}
    written = write_new_candidates(candidates)
    logger.info("Flagged %d candidate group(s) for human review.", written)
    return {"flagged_for_review": written, "skipped": 0}


@flow(
    name="c9r_ingest",
    description="Weekly c9r scrobble-ingestion pipeline.",
    retries=0,
    log_prints=True,
)
def weekly_ingest_flow(*, full: bool = False, source: str | None = None) -> dict:
    """
    Orchestrates the weekly ingestion pipeline.

    Steps: fetch → fix-encoding → enrich → clean → canonise.
    """
    logger = get_run_logger()
    start = datetime.now(UTC)
    if source is None:
        source = os.getenv("C9R_SOURCE", "lastfm")
    # Resolving username from the appropriate env var
    if source == "lastfm":
        username = os.getenv("LASTFM_USER")
        if not username:
            raise ValueError("LASTFM_USER not set.")
    else:
        username = os.getenv("LB_USER")
        if not username:
            raise ValueError("LB_USER not set.")
    logger.info("c9r ingest flow started at %s for user '%s' (source=%s).", start, username, source)
    new_scrobbles = fetch_scrobbles(username, full=full, source=source)
    enc_results = fix_encoding_task()
    encoding_fixed = sum(r[0] for r in enc_results.values())
    enriched = enrich_artists()
    removed, remaining = clean_artists()
    # Appending canonisation as the 5th step
    canon_result = canonise_batch()
    end = datetime.now(UTC)
    duration = end - start
    logger.info("Flow finished in %s.", duration)
    return {
        "new_scrobbles": new_scrobbles,
        "encoding_fixed": encoding_fixed,
        "enriched_artists": enriched,
        "cleaned": removed,
        "remaining": remaining,
        "flagged_for_review": canon_result["flagged_for_review"],
        "skipped": canon_result["skipped"],
        "duration": str(duration),
    }


if __name__ == "__main__":
    weekly_ingest_flow()
