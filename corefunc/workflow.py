"""
Core workflow functionality for CanonFodder.

This module provides the main data gathering workflow for fetching and processing
scrobble data from Last.fm.
"""

from __future__ import annotations
import os
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib

# Import from helpers
from helpers.progress import ProgressCallback, null_progress_callback
from helpers.io import dump_parquet, latest_parquet, register_custom_palette
from helpers.cli import yes_no, choose_lastfm_user
import helpers.aliases as mb_alias

# Import from DB
from DB import engine, SessionLocal
from DB.models import Base
from DB.ops import (
    ascii_freq,
    bulk_insert_scrobbles,
    load_scrobble_table_from_db_to_df,
    seed_ascii_chars,
    populate_artist_info_from_scrobbles
)

# Import API modules
from HTTP import lfAPI
from HTTP import mbAPI

# Import core functionality
from corefunc import dataprofiler as dp


def run_data_gathering_workflow(
        username: str = None,
        progress_callback: Optional[ProgressCallback] = None
) -> None:
    """
    Run the standard data gathering workflow.

    Parameters
    ----------
    username : str, optional
        Last.fm username to fetch data for. If None, will prompt for input.
    progress_callback : ProgressCallback, optional
        Callback function for progress updates, receiving:
            task_name: str - Current task name
            percentage: float - Progress percentage (0-100)
            message: str - Optional status message
    """
    # Use null callback if none provided
    if progress_callback is None:
        progress_callback = null_progress_callback

    # If no username provided, use CLI prompt (only in CLI mode)
    if username is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        username = choose_lastfm_user()
        if not username:
            print("No username provided, exiting.")
            return None

    try:
        matplotlib.use("TkAgg")

        # Configure logging - normal StreamHandler so logs appear in console too
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Report progress if callback provided
        if progress_callback:
            progress_callback("Initializing", 5, "Setting up environment")

        # ─── Set-up ─────────────────────────────────
        Base.metadata.create_all(engine)
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        pd.set_option("display.width", 200)

        HERE = Path(__file__).resolve().parent
        PROJECT_ROOT = HERE.parent
        JSON_DIR = PROJECT_ROOT / "JSON"
        PALETTES_FILE = JSON_DIR / "palettes.json"

        if progress_callback:
            progress_callback("Initializing", 10, "Loading color palettes")
        else:
            print(f"\nFetching recent scrobbles for {username}...")

        # Load custom palette
        try:
            with PALETTES_FILE.open("r", encoding="utf-8") as fh:
                custom_palettes = json.load(fh)["palettes"]
            custom_colors = register_custom_palette("colorpalette_5", custom_palettes)
            sns.set_style(style="whitegrid")
            sns.set_palette(sns.color_palette(custom_colors))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load custom palette: {e}")
            # Use default style if palettes file can't be loaded
            sns.set_style(style="whitegrid")
            custom_colors = None

        if progress_callback:
            progress_callback("Checking database", 15, "Looking for existing scrobbles")

        # Find newest scrobble timestamp already in DB
        try:
            df_db, _tbl = load_scrobble_table_from_db_to_df(engine)
            latest_ts: int | None = None
            if df_db is not None and not df_db.empty:
                latest_ts = int(df_db["play_time"].max().timestamp())
                logging.info("DB already holds %s scrobbles – newest at %s",
                             len(df_db), df_db['play_time'].max())
                if progress_callback:
                    progress_callback("Checking database", 25, f"Found {len(df_db)} scrobbles in database")
            else:
                if progress_callback:
                    progress_callback("Checking database", 25, "No existing scrobbles found")
        except Exception as e:
            logging.error(f"Error accessing database: {e}")
            latest_ts = None
            if progress_callback:
                progress_callback("Checking database", 25, "Error querying database")

        logging.info("Fetching scrobbles from Last.fm API%s …",
                     f' since {latest_ts}' if latest_ts else '')
        if progress_callback:
            progress_callback("Fetching from Last.fm API", 30, "Connecting to Last.fm")

        # Fetch scrobbles with progress updates
        class ProgressTracker:
            def __init__(self, callback: ProgressCallback):
                self.total_pages = 0
                self.current_page = 0
                self.callback = callback

            def update(self, current_page, total_pages=None):
                if total_pages is not None:
                    self.total_pages = total_pages
                self.current_page = current_page
                if self.total_pages > 0:
                    percentage = 30 + (self.current_page / self.total_pages) * 40
                    msg = f"Page {self.current_page}/{self.total_pages}"
                    self.callback("Fetching from Last.fm API", percentage, msg)

        tracker = ProgressTracker(progress_callback)

        df_recent = lfAPI.fetch_scrobbles_since(username, latest_ts)
        # ─── API stage ──────────────
        if df_recent.empty:
            logging.info("No NEW scrobbles since last run – nothing to do.")
            if progress_callback:
                progress_callback("Complete", 100, "No new scrobbles to process")
            else:
                print("No NEW scrobbles since last run – nothing to do.")
            return 0 if username is None else None  # For CLI compatibility

        if progress_callback:
            progress_callback("Processing data", 75, f"Processing {len(df_recent)} scrobbles")
        else:
            print(f"Added {len(df_recent)} new scrobbles to the database.")

        # Insert into database
        if progress_callback:
            progress_callback("Storing results", 80, "Inserting into database")
        bulk_insert_scrobbles(df_recent, engine)

        # Update country information
        if not progress_callback:
            print("\nUpdating country information...")
            try:
                with SessionLocal() as session:
                    updated = lfAPI.sync_user_country(session, username)
                    if updated:
                        print("Country information updated.")
                    else:
                        print("Country information is already up-to-date.")
            except Exception as e:
                print(f"Error updating country information: {str(e)}")
        elif progress_callback:
            try:
                with SessionLocal() as session:
                    updated = lfAPI.sync_user_country(session, username)
                    progress_callback("Storing results", 85, "Country information updated")
            except Exception as e:
                progress_callback("Storing results", 85, f"Error updating country: {str(e)}")

        if progress_callback:
            progress_callback("Storing results", 90, "Updating statistics")
        else:
            print("\nRunning data profiling...")

        progress_callback("Enriching", 92, "Populating artist metadata…")
        try:
            populate_artist_info_from_scrobbles(progress_cb=progress_callback)
        except Exception as exc:
            progress_callback("Warning", 92, f"Artist enrichment failed: {exc}")

        progress_callback("Complete", 94, "Full refresh done")

        if progress_callback:
            progress_callback("Finalizing", 96, "Saving to parquet files")

        # Save to single consolidated parquet file
        dump_parquet(df_recent, constant=True)

        # Run data profiling
        if not progress_callback:
            try:
                # Run profiling and generate report
                profile_result = dp.run_profiling(df_recent)

                # Generate HTML report
                report_path = Path("docs/reports")
                report_path.mkdir(exist_ok=True, parents=True)
                html_path = report_path / "profiling_report.html"

                try:
                    dp.generate_html_report(profile_result, html_path)
                    print(f"Data profiling report saved to {html_path}")
                except ImportError:
                    print("Showdown not available. HTML report not generated.")

                print("Data profiling completed.")
            except Exception as e:
                print(f"Error during data profiling: {str(e)}")

        logging.info("Data gathering finished successfully.")
        if progress_callback:
            progress_callback("Complete", 100, "Data gathering completed successfully")
        else:
            print("\nWorkflow completed successfully.")

        return 0 if username is None else None  # Return value for CLI compatibility

    except KeyError as e:
        # Specific handling for common errors
        error_msg = f"Data format error: {str(e)}"
        logging.exception(f"Data structure error in data gathering workflow: {error_msg}")
        if progress_callback:
            progress_callback("Error", 100, error_msg)
        else:
            print(error_msg)
            print("This may indicate an issue with the Last.fm API response format.")
        raise  # Re-raise to be caught by the calling function

    except Exception as e:
        # More detailed error information for general exceptions
        error_details = traceback.format_exc()
        error_msg = f"Error: {str(e)}"
        logging.exception(f"Unexpected error in data gathering workflow: {error_msg}\n{error_details}")
        if progress_callback:
            progress_callback("Error", 100, error_msg)
        else:
            print(error_msg)
            print("See logs for detailed error information.")
        raise  # Re-raise to be caught by the calling function
