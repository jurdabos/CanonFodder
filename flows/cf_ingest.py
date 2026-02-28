"""
Prefect flow for the c9r data pipeline.

Orchestrates:
  1. Scrobble ingestion       (FR-01 → FR-03)
  2. Artist enrichment        (FR-04)
  3. Artist-info cleanup      (dedup)
  4. Canonisation + propagate (FR-05)
  5. Gold-standard refresh    (FR-06)  — optional, requires local MB mirror
  6. Model retraining         (FR-07)  — optional, requires gold-standard data

Retries with exponential back-off are configured per task (FR-10).
"""
from __future__ import annotations
import logging
import os
from datetime import UTC, datetime
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from prefect.tasks import exponential_backoff
load_dotenv()
log = logging.getLogger(__name__)


@task(
    name="fetch_scrobbles",
    description="Fetches new scrobbles and persists to Parquet.",
    retries=8,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
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
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
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


@task(
    name="propagate_avc",
    description="Applies decided AVC mappings to artist_info and scrobble history (FR-05).",
    retries=0,
    log_prints=True,
)
def propagate_avc_task() -> dict[str, int]:
    """Propagates decided canonisation to artist_info; returns summary."""
    from corefunc.canon.workflow import propagate_avc
    logger = get_run_logger()
    result = propagate_avc()
    logger.info("Propagated AVC: %d updated, %d aliases added.", result["updated"], result["aliases_added"])
    return result


@task(
    name="augment_gold_standard",
    description="Refreshes gold-standard pairs from local MB mirror (FR-06).",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
    log_prints=True,
)
def augment_gs_task() -> dict[str, int]:
    """Extracts positive/negative pairs from MBDB into gs_mb.parquet.

    Skips gracefully when the local MusicBrainz mirror is unreachable.
    """
    from corefunc.canon.augment import augment_gold_standard
    from corefunc.mb_local import check_local_mb
    logger = get_run_logger()
    if not check_local_mb():
        logger.warning("Gold-standard augmentation skipped — local MB mirror unreachable.")
        return {"rows_written": 0, "skipped": True}
    n = augment_gold_standard()
    logger.info("Gold-standard refresh: %d rows written to gs_mb.parquet.", n)
    return {"rows_written": n, "skipped": False}


@task(
    name="retrain_model",
    description="Retrains the canonisation model on current gold-standard data (FR-07).",
    retries=0,
    log_prints=True,
)
def retrain_model_task() -> dict[str, int]:
    """Runs the unified training pipeline with default settings.

    Skips gracefully when gold-standard data is missing.
    Persists the best model to ML/ for inference after successful retraining.
    """
    from corefunc.canon.trainer import run_training
    from corefunc.canon.tuner import save_best_historical_models
    from helpers.io import AVC_PQ, read_parquet
    logger = get_run_logger()
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        logger.warning("Retraining skipped — no AVC data available.")
        return {"models_trained": 0, "skipped": True}
    decided = avc[avc["to_link"].notna()]
    if len(decided) < 20:
        logger.warning("Retraining skipped — only %d decided AVC rows (need ≥20).", len(decided))
        return {"models_trained": 0, "skipped": True}
    logger.info("Retraining on %d decided AVC rows.", len(decided))
    results = run_training(run_name="flow_retrain")
    logger.info("Retraining complete — %d model(s) evaluated.", len(results))
    # Exporting best model per type from MLflow → ML/*.pkl for inference
    saved = save_best_historical_models()
    logger.info("Exported %d model pickle(s) to ML/.", len(saved))
    return {"models_trained": len(results), "skipped": False}


@flow(
    name="c9r_ingest",
    description="Weekly c9r scrobble-ingestion pipeline.",
    retries=0,
    log_prints=True,
)
def weekly_ingest_flow(*, full: bool = False, source: str | None = None) -> dict:
    """
    Orchestrates the weekly ingestion pipeline.

    Steps: fetch → fix-encoding → enrich → clean → canonise → propagate
           → augment gold standard → retrain model.
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
    # FR-01 → FR-03: Fetching, normalising, and persisting scrobbles
    new_scrobbles = fetch_scrobbles(username, full=full, source=source)
    enc_results = fix_encoding_task()
    encoding_fixed = sum(r[0] for r in enc_results.values())
    # FR-04: Enriching unknown artists from MusicBrainz
    enriched = enrich_artists()
    removed, remaining = clean_artists()
    # FR-05: Canonising — discovering variants and applying decided mappings
    canon_result = canonise_batch()
    propagate_result = propagate_avc_task()
    # FR-06: Refreshing gold-standard pairs from local MB mirror
    gs_result = augment_gs_task()
    # FR-07: Retraining the canonisation model on current data
    retrain_result = retrain_model_task()
    end = datetime.now(UTC)
    duration = end - start
    logger.info("Flow finished in %s.", duration)
    return {
        "new_scrobbles": new_scrobbles,
        "encoding_fixed": encoding_fixed,
        "enriched_artists": enriched,
        "cleaned": removed,
        "remaining": remaining,
        "auto_linked": 0,
        "flagged_for_review": canon_result["flagged_for_review"],
        "skipped": canon_result["skipped"],
        "avc_propagated": propagate_result["updated"],
        "avc_aliases_added": propagate_result["aliases_added"],
        "gs_rows_written": gs_result["rows_written"],
        "models_trained": retrain_result["models_trained"],
        "duration": str(duration),
    }


if __name__ == "__main__":
    weekly_ingest_flow.serve(
        name="c9r-weekly-ingest",
        cron="0 3 * * 1",  # to run every Monday at 03:00 UTC
    )
