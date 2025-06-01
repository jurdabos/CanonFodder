"""
Core data gathering functionality for CanonFodder.

This module provides the main data gathering workflow for fetching and processing
scrobble data from Last.fm, as well as functions for cleaning and maintaining the database.
"""

from __future__ import annotations
import os
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple
from collections import Counter

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
from DB.models import Base, ArtistInfo, Scrobble
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

from sqlalchemy import select, func, delete, text


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
                dp.run_profiling()
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


def run_full_refresh(
        username: str = None,
        progress_callback: Optional[ProgressCallback] = None
) -> None:
    """
    Perform a full refresh of the scrobble data.
    This will delete all existing scrobbles and fetch a complete history from Last.fm.

    Parameters
    ----------
    username : str, optional
        Last.fm username to fetch data for. If None, will prompt for input.
    progress_callback : ProgressCallback, optional
        Callback function for progress updates.
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
        # 1) Hard-delete every row (TRUNCATE preferred where available)
        progress_callback("Clearing table", 5, "Dropping existing rows…")
        try:
            with SessionLocal() as sess:
                dialect = sess.bind.dialect.name
                if dialect in ("mysql", "postgresql"):
                    sess.execute(text("TRUNCATE TABLE scrobble"))
                else:  # SQLite has no TRUNCATE
                    sess.execute(delete(Scrobble))
                sess.commit()
        except Exception as exc:
            progress_callback("Error", 100, f"Purging failed: {exc}")
            return

        # 2) Vacuum / reset autoinc if SQLite
        with SessionLocal() as sess:
            if sess.bind.dialect.name == "sqlite":
                sess.execute(text("VACUUM"))
                sess.commit()

        progress_callback("Fetching", 10, "Requesting full history from API")
        
        # 3) Call existing pipeline – empty table → fetches everything
        run_data_gathering_workflow(username, progress_callback)
        
        # 4) Enrich freshly fetched scrobbles with MB artist meta
        progress_callback("Enriching", 90, "Populating artist metadata…")
        try:
            populate_artist_info_from_scrobbles(progress_cb=progress_callback)
        except Exception as exc:  # fail-safe; don't abort
            progress_callback("Warning", 95, f"Artist enrichment failed: {exc}")
        
        progress_callback("Complete", 100, "Full refresh done")

    except Exception as e:
        # More detailed error information for general exceptions
        error_details = traceback.format_exc()
        error_msg = f"Error: {str(e)}"
        logging.exception(f"Unexpected error in full refresh: {error_msg}\n{error_details}")
        if progress_callback:
            progress_callback("Error", 100, error_msg)
        else:
            print(error_msg)
            print("See logs for detailed error information.")
        raise  # Re-raise to be caught by the calling function


def clean_artist_info_table() -> Tuple[int, int]:
    """
    Clean up the ArtistInfo table by removing duplicates and unnecessary entries.
    This function:
    1. Identifies duplicate artist names
    2. Keeps only the most complete record for each artist
    3. Removes any orphaned artists (not referenced in the Scrobble table)
    
    Returns:
        tuple: (cleaned_count, total_count) - number of records cleaned and total remaining
    """
    print("Starting ArtistInfo table cleanup...")
    
    with SessionLocal() as session:
        # Get all artists from the ArtistInfo table
        artists_query = select(ArtistInfo)
        artists = session.execute(artists_query).scalars().all()
        
        if not artists:
            print("No artists found in the ArtistInfo table.")
            return 0, 0
        
        total_before = len(artists)
        print(f"Found {total_before} artists in the ArtistInfo table.")
        
        # Find artists with duplicate names
        artist_names = [a.artist_name for a in artists]
        name_counts = Counter(artist_names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        
        if duplicates:
            print(f"Found {len(duplicates)} artist names with duplicates.")
            
            # For each duplicate, keep only the most complete record
            for name, count in duplicates.items():
                dupes = session.execute(
                    select(ArtistInfo).where(ArtistInfo.artist_name == name)
                ).scalars().all()
                
                # Sort by completeness (non-null fields)
                def completeness_score(artist):
                    score = 0
                    if artist.mbid:
                        score += 2  # MBID is most important
                    if artist.country:
                        score += 1
                    if artist.disambiguation_comment:
                        score += 1
                    if artist.aliases:
                        score += 1
                    return score
                
                sorted_dupes = sorted(dupes, key=completeness_score, reverse=True)
                
                # Keep the most complete record, delete the rest
                for dupe in sorted_dupes[1:]:
                    session.delete(dupe)
            
            session.commit()
        
        # Find orphaned artists (not referenced in Scrobble table)
        # This is optional and might be slow on large databases
        
        # Get all artist names from the Scrobble table
        scrobble_artists = session.execute(
            select(Scrobble.artist_name).distinct()
        ).scalars().all()
        
        scrobble_artist_set = set(scrobble_artists)
        
        # Find artists in ArtistInfo that aren't in Scrobble
        orphaned = []
        for artist in artists:
            if artist.artist_name not in scrobble_artist_set:
                orphaned.append(artist)
        
        if orphaned:
            print(f"Found {len(orphaned)} orphaned artists (not referenced in Scrobble table).")
            # Uncomment to actually delete orphaned artists
            # for artist in orphaned:
            #     session.delete(artist)
            # session.commit()
        
        # Get final count
        final_count = session.execute(select(func.count()).select_from(ArtistInfo)).scalar_one()
        cleaned_count = total_before - final_count
        
        print(f"Cleanup complete. Removed {cleaned_count} records, {final_count} artists remain.")
        return cleaned_count, final_count