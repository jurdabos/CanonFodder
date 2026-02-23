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
    logger = get_run_logger()
    logger.info("Fetching scrobbles for %s from %s", username, source)
    n = run_data_gathering_workflow(username, full=full, source=source)
    logger.info("Ingested %d scrobbles.", n)
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


@flow(
    name="c9r_ingest",
    description="Weekly c9r scrobble-ingestion pipeline.",
    retries=0,
    log_prints=True,
)
def weekly_ingest_flow(*, full: bool = False, source: str | None = None) -> dict:
    """
    Orchestrates the weekly ingestion pipeline.

    Steps: fetch → enrich → clean.
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
    enriched = enrich_artists()
    removed, remaining = clean_artists()
    end = datetime.now(UTC)
    duration = end - start
    logger.info("Flow finished in %s.", duration)
    return {
        "new_scrobbles": new_scrobbles,
        "enriched_artists": enriched,
        "cleaned": removed,
        "remaining": remaining,
        "duration": str(duration),
    }


if __name__ == "__main__":
    weekly_ingest_flow()
