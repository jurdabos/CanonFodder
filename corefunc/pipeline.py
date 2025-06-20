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
from corefunc.canonizer import apply_previous
from corefunc.model_server import start_server, stop_server, get_server_status
from helpers.io import dump_parquet, latest_parquet
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
    # Getting username from parameter or environment variable
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
        # Inserting into database
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


def run_canonization(
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[int, str, bool]]:
    """
    Run the canonization process to group artist name variants and store mapping in AVC.

    Parameters
    ----------
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[int, str, bool]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'row_count': Number of rows processed
        - 'artist_count': Number of unique artists
        - 'data_source': Source of the data ('parquet' or 'database')
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    try:
        # Report progress
        progress_callback("Canonization", 0, "Starting artist name canonization")

        # Check model server status and start if not running
        status = get_server_status()
        if not status["is_running"]:
            progress_callback("Canonization", 10, "Starting model server")
            success = start_server()
            if success:
                logger.info("Model server started successfully")
                progress_callback("Canonization", 20, "Model server started successfully")
            else:
                logger.warning("Failed to start model server. Continuing without it.")
                progress_callback("Canonization", 20, "Failed to start model server. Continuing without it.")
        else:
            logger.info("Model server is already running")
            progress_callback("Canonization", 20, "Model server is already running")

        # Load data
        progress_callback("Canonization", 30, "Loading data")

        # Try to load from parquet first
        data_source = "parquet"
        try:
            data = pd.read_parquet(latest_parquet())
            logger.info("Loaded data from parquet")
            progress_callback("Canonization", 40, "Loaded data from parquet")
        except Exception as e:
            # Fall back to database
            logger.warning(f"Failed to load from parquet: {str(e)}. Falling back to database.")
            progress_callback("Canonization", 40, "Falling back to database")
            data, _ = load_scrobble_table_from_db_to_df(engine)
            data_source = "database"

        if data is None or data.empty:
            error_msg = "No data found for canonization"
            logger.error(error_msg)
            progress_callback("Error", 100, error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'row_count': 0,
                'artist_count': 0,
                'data_source': data_source
            }

        # Apply previous canonization
        progress_callback("Canonization", 60, "Applying previous canonization")
        data = apply_previous(data)

        # Get statistics
        row_count = len(data)
        artist_count = data["Artist"].nunique()

        logger.info(f"Canonization complete: processed {row_count} rows with {artist_count} unique artists")
        progress_callback("Complete", 100, f"Processed {row_count} rows with {artist_count} unique artists")

        return {
            'status': 'success',
            'message': f"Successfully applied canonization to {row_count} rows with {artist_count} unique artists",
            'row_count': row_count,
            'artist_count': artist_count,
            'data_source': data_source
        }

    except Exception as e:
        error_msg = f"Error during canonization: {str(e)}"
        logger.exception(error_msg)
        progress_callback("Error", 100, error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'row_count': 0,
            'artist_count': 0,
            'data_source': 'unknown'
        }


def export_to_parquet(
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Union[str, Path]]:
    """
    Export data to parquet files.

    Parameters
    ----------
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.

    Returns
    -------
    Dict[str, Union[str, Path]]
        Dictionary with status information:
        - 'status': 'success' or 'error'
        - 'message': Status message
        - 'parquet_path': Path to the exported parquet file
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    try:
        # Report progress
        progress_callback("Exporting", 0, "Starting parquet export")

        # Load data from database
        progress_callback("Exporting", 20, "Loading data from database")
        df, _tbl = load_scrobble_table_from_db_to_df(engine)

        if df is None or df.empty:
            error_msg = "No data found for export"
            logger.error(error_msg)
            progress_callback("Error", 100, error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'parquet_path': None
            }

        # Export to parquet
        progress_callback("Exporting", 60, "Exporting to parquet")
        parquet_path = dump_parquet(df, constant=True)

        logger.info(f"Parquet export complete: {parquet_path}")
        progress_callback("Complete", 100, f"Exported to {parquet_path}")

        return {
            'status': 'success',
            'message': f"Successfully exported data to {parquet_path}",
            'parquet_path': parquet_path
        }

    except Exception as e:
        error_msg = f"Error during parquet export: {str(e)}"
        logger.exception(error_msg)
        progress_callback("Error", 100, error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'parquet_path': None
        }


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
