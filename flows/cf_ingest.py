"""
Prefect flow for CanonFodder data pipeline.
This flow orchestrates the complete data pipeline for CanonFodder, including:
- Fetching new scrobbles from Last.fm (FR-01)
- Normalizing scrobbles (FR-02)
- Bulk inserting into database (FR-03)
- Artist enrichment from MusicBrainz (FR-04)
- Canonization of artist name variants (FR-05)
- Parquet export (FR-06)
- Data profiling for BI frontend (FR-07)

The flow includes proper retry and back-off mechanisms (FR-08) and
is designed to complete within 15 minutes for 1 million scrobbles (FR-09).
"""

import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Dict, Optional, Union

from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
# from prefect.schedules import CronSchedule  # Not needed for direct execution

# Load environment variables
load_dotenv()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corefunc.pipeline import clean_artist_data as clean_artist_data_func
from corefunc.pipeline import enrich_artist_data
from corefunc.pipeline import export_to_parquet as export_to_parquet_func
from corefunc.pipeline import fetch_new_data
from corefunc.pipeline import run_canonization as run_canonization_func
from corefunc.pipeline import run_data_profiling as run_data_profiling_func

# Import CanonFodder modules
from HTTP import mbAPI

# Initialize MusicBrainz API
mbAPI.init()


@task(
    name="fetch_new_scrobbles",
    description="Fetch new scrobbles from Last.fm",
    retries=8,  # Up to 8 retries as per FR-08
    retry_delay_seconds=30,  # Start with 30 seconds
    retry_jitter_factor=1.5,  # Exponential back-off
    log_prints=True,
)
def fetch_new_scrobbles() -> Optional[int]:
    """
    Fetch new scrobbles from last.fm since the last run (FR-01, FR-02, FR-03).
    This task:
    - Pulls recent tracks for a user since the last stored timestamp
    - Persists raw JSON
    - Normalizes scrobbles (rename columns, convert UTS→UTC datetime)
    - Removes duplicates
    - Bulk inserts into database with conflict handling
    """
    logger = get_run_logger()
    logger.info("Starting fetch of new scrobbles")

    # Get username from environment variable
    username = os.getenv("LASTFM_USER")
    if not username:
        raise ValueError("No Last.fm username provided. Set LASTFM_USER environment variable.")

    result = fetch_new_data(username)

    if result["status"] == "error":
        raise ValueError(f"Error fetching new data: {result['message']}")

    if result["new_scrobbles"] == 0:
        logger.info("No new scrobbles since last run – nothing to do.")
        return None

    logger.info(f"Fetched {result['new_scrobbles']} new scrobbles")
    return result["new_scrobbles"]


@task(
    name="enrich_artist_info",
    description="Enrich artist information from MusicBrainz",
    retries=8,
    retry_delay_seconds=30,
    retry_jitter_factor=1.5,
    log_prints=True,
)
def enrich_artist_info(new_scrobbles: Optional[int]) -> Dict[str, int]:
    """
    Enrich artist information from MusicBrainz (FR-04).

    This task:
    - For new MBIDs, fetches country & aliases
    - Caches results in artist_info table
    """
    logger = get_run_logger()
    logger.info("Starting artist info enrichment")

    if new_scrobbles is None:
        logger.info("No new scrobbles to process - continuing with artist enrichment anyway")
    else:
        logger.info(f"Processing artist info for {new_scrobbles} new scrobbles")

    result = enrich_artist_data()

    if result["status"] == "error":
        raise ValueError(f"Error enriching artist data: {result['message']}")

    logger.info(
        f"Artist info enrichment complete: processed={result['processed']}, "
        f"created={result['created']}, updated={result['updated']}"
    )

    return {"processed": result["processed"], "created": result["created"], "updated": result["updated"]}


@task(name="clean_artist_data", description="Clean up artist data", retries=3, retry_delay_seconds=30, log_prints=True)
def clean_artist_data(enrichment_result: Dict[str, int]) -> Dict[str, int]:
    """
    Clean up artist data by removing duplicates and orphaned records.
    """
    logger = get_run_logger()
    logger.info("Starting artist data cleanup")

    result = clean_artist_data_func()

    if result["status"] == "error":
        raise ValueError(f"Error cleaning artist data: {result['message']}")

    logger.info(f"Artist data cleanup complete: removed {result['cleaned']} records, " f"{result['remaining']} remain")

    return {"cleaned": result["cleaned"], "remaining": result["remaining"]}


@task(
    name="run_canonization",
    description="Run canonization to group artist name variants",
    retries=3,
    retry_delay_seconds=30,
    log_prints=True,
)
def run_canonization(cleanup_result: Dict[str, int]) -> Dict[str, Union[int, str]]:
    """
    Run canonization to group artist name variants (FR-05).

    This task:
    - Groups artist name variants
    - Stores mapping in ArtistVariantsCanonized table
    - Applies canonization to scrobble history
    """
    logger = get_run_logger()
    logger.info("Starting artist name canonization")

    result = run_canonization_func()

    if result["status"] == "error":
        raise ValueError(f"Error running canonization: {result['message']}")

    logger.info(
        f"Canonization complete: processed {result['row_count']} rows with " f"{result['artist_count']} unique artists"
    )

    return {
        "row_count": result["row_count"],
        "artist_count": result["artist_count"],
        "data_source": result["data_source"],
    }


@task(
    name="export_to_parquet",
    description="Export data to parquet files",
    retries=3,
    retry_delay_seconds=30,
    log_prints=True,
)
def export_to_parquet(canonization_result: Dict[str, Union[int, str]]) -> Dict[str, str]:
    """
    Export data to parquet files (FR-06).

    This task:
    - Dumps star schema to parquet files
    - Used for analytics and BI dashboards
    """
    logger = get_run_logger()
    logger.info("Starting parquet export")

    result = export_to_parquet_func()

    if result["status"] == "error":
        raise ValueError(f"Error exporting to parquet: {result['message']}")

    logger.info(f"Parquet export complete: {result['parquet_path']}")

    return {"parquet_path": str(result["parquet_path"])}


@task(
    name="run_data_profiling",
    description="Run data profiling to generate analytics",
    retries=3,
    retry_delay_seconds=30,
    log_prints=True,
)
def run_data_profiling(parquet_result: Dict[str, str]) -> bool:
    """
    Run data profiling to generate analytics (FR-07).

    This task:
    - Generates analytics for BI dashboards
    - Prepares data for interactive exploration
    """
    logger = get_run_logger()
    logger.info("Starting data profiling")

    result = run_data_profiling_func()

    if result["status"] == "error":
        raise ValueError(f"Error running data profiling: {result['message']}")

    logger.info("Data profiling complete")
    return True


@flow(
    name="cf_ingest",
    description="Complete CanonFodder data pipeline covering FR-01 to FR-07",
    retries=0,  # Don't retry the entire flow, let individual tasks handle retries
    log_prints=True,
)
def weekly_ingest_flow():
    """
    Weekly data ingestion flow for CanonFodder.

    This flow orchestrates the entire data pipeline from fetching new scrobbles
    to generating analytics for BI dashboards. It's designed to run weekly
    and includes proper error handling and retry mechanisms.
    """
    logger = get_run_logger()
    logger.info("Starting CanonFodder weekly ingestion flow")

    # Get the current time for logging
    start_time = datetime.now(UTC)
    logger.info(f"Flow started at: {start_time}")

    # Execute the pipeline tasks in order
    new_scrobbles = fetch_new_scrobbles()
    enrichment_result = enrich_artist_info(new_scrobbles)
    cleanup_result = clean_artist_data(enrichment_result)
    canonization_result = run_canonization(cleanup_result)
    parquet_result = export_to_parquet(canonization_result)
    profiling_result = run_data_profiling(parquet_result)

    # Log completion
    end_time = datetime.now(UTC)
    duration = end_time - start_time
    logger.info(f"Flow completed at: {end_time}")
    logger.info(f"Total duration: {duration}")

    return {
        "new_scrobbles": new_scrobbles,
        "enrichment": enrichment_result,
        "cleanup": cleanup_result,
        "canonization": canonization_result,
        "parquet_path": parquet_result["parquet_path"],
        "profiling_complete": profiling_result,
        "duration": str(duration),
    }


# Create a deployment with a weekly schedule
if __name__ == "__main__":
    # For local testing, run the flow immediately
    weekly_ingest_flow()

    # To create a deployment with a schedule, uncomment the following:
    # from prefect.deployments import Deployment
    #
    # deployment = Deployment.build_from_flow(
    #     flow=weekly_ingest_flow,
    #     name="weekly-ingest",
    #     # schedule=CronSchedule(cron="0 0 * * 0", timezone="UTC"),  # Weekly on Sunday at midnight UTC
    #     work_queue_name="default",
    #     tags=["canonfodder", "lastfm", "musicbrainz", "data-pipeline"]
    # )
    # deployment.apply()
