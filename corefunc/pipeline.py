"""
Pull-based pipeline for CanonFodder.

This module provides a pull-based pipeline for fetching new data from Last.fm,
enriching artist information from MusicBrainz, and cleaning up the database.
It can be triggered manually or via Airflow.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Optional, Tuple, Union, Callable

import pandas as pd

from DB import engine, SessionLocal
from DB.ops import bulk_insert_scrobbles, load_scrobble_table_from_db_to_df, populate_artist_info_from_scrobbles
from HTTP import lfAPI, mbAPI
from corefunc.data_cleaning import clean_artist_info_table
from helpers.io import dump_parquet
from helpers.progress import ProgressCallback, null_progress_callback
from corefunc import dataprofiler as dp

# Initialize logging
logger = logging.getLogger(__name__)


def fetch_new_data(
        username: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[int, str, None]]:
    """
    Fetch new data from Last.fm since the last run.

    Parameters
    ----------
    username : str, optional
        Last.fm username to fetch data for. If None, will use the LASTFM_USER environment variable.
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[int, str, None]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'new_scrobbles': Number of new scrobbles fetched
        - 'latest_timestamp': Latest timestamp in the database
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    # Get username from parameter or environment variable
    username = username or os.getenv("LASTFM_USER")
    if not username:
        error_msg = "No Last.fm username provided. Set LASTFM_USER environment variable or pass username parameter."
        logger.error(error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'new_scrobbles': 0,
            'latest_timestamp': None
        }

    try:
        # Initialize MusicBrainz API if not already initialized
        mbAPI.init()

        # Report progress
        progress_callback("Initializing", 5, "Setting up environment")

        # Find newest scrobble timestamp already in DB
        progress_callback("Checking database", 10, "Looking for existing scrobbles")
        df_db, _tbl = load_scrobble_table_from_db_to_df(engine)
        latest_ts = None
        if df_db is not None and not df_db.empty:
            latest_ts = int(df_db["play_time"].max().timestamp())
            logger.info(f"DB already holds {len(df_db)} scrobbles – newest at {df_db['play_time'].max()}")
            progress_callback("Checking database", 15, f"Found {len(df_db)} scrobbles in database")
        else:
            progress_callback("Checking database", 15, "No existing scrobbles found")

        # Fetch new scrobbles
        logger.info(f"Fetching scrobbles from Last.fm API since {latest_ts}")
        progress_callback("Fetching from Last.fm API", 20, "Connecting to Last.fm")

        df_recent = lfAPI.fetch_scrobbles_since(username, latest_ts)

        if df_recent.empty:
            logger.info("No new scrobbles since last run – nothing to do.")
            progress_callback("Complete", 100, "No new scrobbles to process")
            return {
                'status': 'success',
                'message': "No new scrobbles since last run",
                'new_scrobbles': 0,
                'latest_timestamp': latest_ts
            }

        # Insert into database
        logger.info(f"Fetched {len(df_recent)} new scrobbles")
        progress_callback("Processing data", 50, f"Processing {len(df_recent)} scrobbles")

        progress_callback("Storing results", 60, "Inserting into database")
        bulk_insert_scrobbles(df_recent, engine)

        # Update country information
        progress_callback("Storing results", 70, "Updating country information")
        with SessionLocal() as session:
            try:
                updated = lfAPI.sync_user_country(session, username, ask=False)
                if updated:
                    logger.info("Country information updated.")
                    progress_callback("Storing results", 75, "Country information updated")
                else:
                    logger.info("Country information is already up-to-date.")
                    progress_callback("Storing results", 75, "Country information already up-to-date")
            except Exception as e:
                error_msg = f"Error updating country information: {str(e)}"
                logger.error(error_msg)
                progress_callback("Warning", 75, error_msg)

        # Save to parquet
        progress_callback("Finalizing", 80, "Saving to parquet files")
        dump_parquet(df_recent, constant=True)

        return {
            'status': 'success',
            'message': f"Successfully fetched {len(df_recent)} new scrobbles",
            'new_scrobbles': len(df_recent),
            'latest_timestamp': latest_ts
        }

    except Exception as e:
        error_msg = f"Error fetching new data: {str(e)}"
        logger.exception(error_msg)
        progress_callback("Error", 100, error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'new_scrobbles': 0,
            'latest_timestamp': None
        }


def enrich_artist_data(
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[int, str]]:
    """
    Enrich artist data with information from MusicBrainz.

    Parameters
    ----------
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[int, str]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'processed': Number of artists processed
        - 'created': Number of new artist records created
        - 'updated': Number of artist records updated
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    try:
        # Initialize MusicBrainz API if not already initialized
        mbAPI.init()

        # Report progress
        progress_callback("Enriching", 0, "Starting artist data enrichment")

        # Populate artist info from scrobbles
        processed, created, updated = populate_artist_info_from_scrobbles(progress_cb=progress_callback)

        logger.info(f"Artist info enrichment complete: processed={processed}, created={created}, updated={updated}")
        progress_callback("Complete", 100, f"Processed {processed} artists, created {created}, updated {updated}")

        return {
            'status': 'success',
            'message': f"Successfully enriched artist data: processed={processed}, created={created}, updated={updated}",
            'processed': processed,
            'created': created,
            'updated': updated
        }

    except Exception as e:
        error_msg = f"Error enriching artist data: {str(e)}"
        logger.exception(error_msg)
        progress_callback("Error", 100, error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'processed': 0,
            'created': 0,
            'updated': 0
        }


def clean_artist_data(
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[int, str]]:
    """
    Clean up artist data by removing duplicates and orphaned records.

    Parameters
    ----------
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[int, str]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'cleaned': Number of records cleaned
        - 'remaining': Number of records remaining
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    try:
        # Report progress
        progress_callback("Cleaning", 0, "Starting artist data cleanup")

        # Clean up the ArtistInfo table
        cleaned, remaining = clean_artist_info_table()

        logger.info(f"Artist data cleanup complete: removed {cleaned} records, {remaining} remain")
        progress_callback("Complete", 100, f"Removed {cleaned} records, {remaining} remain")

        return {
            'status': 'success',
            'message': f"Successfully cleaned artist data: removed {cleaned} records, {remaining} remain",
            'cleaned': cleaned,
            'remaining': remaining
        }

    except Exception as e:
        error_msg = f"Error cleaning artist data: {str(e)}"
        logger.exception(error_msg)
        progress_callback("Error", 100, error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'cleaned': 0,
            'remaining': 0
        }


def run_data_profiling(
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, str]:
    """
    Run data profiling to generate analytics.

    Parameters
    ----------
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, str]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    try:
        # Report progress
        progress_callback("Profiling", 0, "Starting data profiling")

        # Run data profiling
        dp.run_profiling()

        logger.info("Data profiling complete")
        progress_callback("Complete", 100, "Data profiling complete")

        return {
            'status': 'success',
            'message': "Successfully ran data profiling"
        }

    except Exception as e:
        error_msg = f"Error running data profiling: {str(e)}"
        logger.exception(error_msg)
        progress_callback("Error", 100, error_msg)
        return {
            'status': 'error',
            'message': error_msg
        }


def run_incremental_pipeline(
        username: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Run the incremental pipeline: fetch new data, enrich artist data, and clean artist data.

    Parameters
    ----------
    username : str, optional
        Last.fm username to fetch data for. If None, will use the LASTFM_USER environment variable.
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[str, Dict]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'fetch_result': Result of fetch_new_data
        - 'enrich_result': Result of enrich_artist_data
        - 'clean_result': Result of clean_artist_data
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    # Initialize result dictionary
    result = {
        'status': 'success',
        'message': "Pipeline completed successfully",
        'fetch_result': None,
        'enrich_result': None,
        'clean_result': None
    }

    # Step 1: Fetch new data
    progress_callback("Pipeline", 0, "Starting data fetch")
    fetch_result = fetch_new_data(username, progress_callback)
    result['fetch_result'] = fetch_result

    if fetch_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at data fetch step: {fetch_result['message']}"
        return result

    # If no new scrobbles, we can still run the other steps
    # Step 2: Enrich artist data
    progress_callback("Pipeline", 33, "Starting artist data enrichment")
    enrich_result = enrich_artist_data(progress_callback)
    result['enrich_result'] = enrich_result

    if enrich_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at artist enrichment step: {enrich_result['message']}"
        return result

    # Step 3: Clean artist data
    progress_callback("Pipeline", 66, "Starting artist data cleanup")
    clean_result = clean_artist_data(progress_callback)
    result['clean_result'] = clean_result

    if clean_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at artist cleanup step: {clean_result['message']}"
        return result

    # All steps completed successfully
    progress_callback("Pipeline", 100, "Pipeline completed successfully")
    return result


def run_full_pipeline(
        username: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Run the full pipeline: fetch new data, enrich artist data, clean artist data, and run data profiling.

    Parameters
    ----------
    username : str, optional
        Last.fm username to fetch data for. If None, will use the LASTFM_USER environment variable.
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[str, Dict]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'fetch_result': Result of fetch_new_data
        - 'enrich_result': Result of enrich_artist_data
        - 'clean_result': Result of clean_artist_data
        - 'profile_result': Result of run_data_profiling
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    # Initialize result dictionary
    result = {
        'status': 'success',
        'message': "Pipeline completed successfully",
        'fetch_result': None,
        'enrich_result': None,
        'clean_result': None,
        'profile_result': None
    }

    # Step 1: Fetch new data
    progress_callback("Pipeline", 0, "Starting data fetch")
    fetch_result = fetch_new_data(username, progress_callback)
    result['fetch_result'] = fetch_result

    if fetch_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at data fetch step: {fetch_result['message']}"
        return result

    # If no new scrobbles, we can still run the other steps
    # Step 2: Enrich artist data
    progress_callback("Pipeline", 25, "Starting artist data enrichment")
    enrich_result = enrich_artist_data(progress_callback)
    result['enrich_result'] = enrich_result

    if enrich_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at artist enrichment step: {enrich_result['message']}"
        return result

    # Step 3: Clean artist data
    progress_callback("Pipeline", 50, "Starting artist data cleanup")
    clean_result = clean_artist_data(progress_callback)
    result['clean_result'] = clean_result

    if clean_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at artist cleanup step: {clean_result['message']}"
        return result

    # Step 4: Run data profiling
    progress_callback("Pipeline", 75, "Starting data profiling")
    profile_result = run_data_profiling(progress_callback)
    result['profile_result'] = profile_result

    if profile_result['status'] == 'error':
        result['status'] = 'error'
        result['message'] = f"Pipeline failed at data profiling step: {profile_result['message']}"
        return result

    # All steps completed successfully
    progress_callback("Pipeline", 100, "Pipeline completed successfully")
    return result
