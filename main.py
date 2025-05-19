"""
CLI-based entry-point for CanonFodder
A hybrid CLI/graphical approach with animated typing and steampunk-inspired aesthetics.
The application maintains all functionality of the previous GUI while presenting
a more console-like experience.
"""

from __future__ import annotations
import os
import sys
import time
import random
import threading
import logging
import curses
import locale
from pathlib import Path
from typing import Callable
import argparse
import platform
import signal
from sqlalchemy import delete
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────
# side-effect imports that need to run once at start-up
# ────────────────────────────────────────────────────────────────
load_dotenv()

import lfAPI  # noqa: E402
from DB import engine  # noqa: E402
from DB.models import Base  # noqa: E402
from corefunc import dataprofiler as dp  # noqa: E402
from corefunc import canonizer as cz  # noqa: E402
from helpers.cli import choose_lastfm_user, unify_artist_names_cli  # CLI integration  # noqa: E402

# ────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ────────────────────────────────────────────────────────────────
APP_TITLE = "CanonFodder"
WELCOME_TEXT = (
    "CanonFodder is a SINGLE-USER installation – one database, one Last.fm user.\n\n"
    "To analyse another account you currently need to install CanonFodder again "
    "into a separate directory (or virtualenv) so the databases do not overlap.\n\n"
    "Please enter the Last.fm user name that shall be analysed in THIS installation."
)

# Color definitions for the steampunk-inspired CLI interface
COLORS = {
    # Original colors
    "WHITE": "#FFFFFF",
    "BLACK": "#000000",
    "TEAL": "#16697A",  # Caribbean Current
    "LIME": "#DBF4A7",  # Mindaro
    "RUST": "#A24936",  # Chestnut
    "BLUE": "#7EBCE6",  # Maya Blue
    "PEACH": "#E6BEAE",  # Pale Dogwood

    # New colors from Coolors palette
    "SAGE": "#79AF91",  # Cambridge Blue
    "SAND": "#BF9F6F",  # Lion
    "MAUVE": "#996662",  # Rose Taupe
    "SLATE": "#90838E",  # Taupe Gray
    "SILVER": "#B2BDCA"  # French Gray
}


# Terminal color codes (ANSI escape sequences)
# Not all terminals support true color, so we use approximations
class Colors:
    RESET = "\033[0m"
    WHITE = "\033[97m"
    BLACK = "\033[30m"
    TEAL = "\033[36m"  # Caribbean Current
    LIME = "\033[92m"  # Mindaro
    RUST = "\033[31m"  # Chestnut
    BLUE = "\033[94m"  # Maya Blue
    PEACH = "\033[93m"  # Pale Dogwood
    SAGE = "\033[32m"  # Cambridge Blue
    SAND = "\033[33m"  # Lion
    MAUVE = "\033[35m"  # Rose Taupe
    SLATE = "\033[90m"  # Taupe Gray
    SILVER = "\033[37m"  # French Gray

    # Background colors
    BG_BLACK = "\033[40m"
    BG_WHITE = "\033[107m"
    BG_TEAL = "\033[46m"
    BG_LIME = "\033[102m"
    BG_RUST = "\033[41m"
    BG_BLUE = "\033[104m"
    BG_PEACH = "\033[103m"
    BG_SAGE = "\033[42m"
    BG_SAND = "\033[43m"
    BG_MAUVE = "\033[45m"
    BG_SLATE = "\033[100m"
    BG_SILVER = "\033[47m"

    # Text styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"


# Initialize color support based on platform
def init_colors():
    if platform.system() == "Windows":
        # Enable ANSI colors on Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass  # Fail silently if Windows color initialization fails


def run_data_gathering_workflow(
        username: str = None,
        progress_callback: Callable[[str, float, str], None] = None
) -> None:
    """
    Run the standard data gathering workflow.
    Parameters
    ----------
    username : str, optional
        Last.fm username to fetch data for. If None, will prompt for input.
    progress_callback : callable, optional
        Callback function for progress updates, receiving:
            task_name: str - Current task name
            percentage: float - Progress percentage (0-100)
            message: str - Optional status message
    """
    # If no username provided, use CLI prompt (only in CLI mode)
    if username is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        username = choose_lastfm_user()
        if not username:
            print("No username provided, exiting.")
            return
    try:
        import os
        import json
        import pandas as pd
        import seaborn as sns
        import matplotlib
        matplotlib.use("TkAgg")
        from DB import engine, SessionLocal
        from DB.ops import (
            ascii_freq,
            bulk_insert_scrobbles,
            load_scrobble_table_from_db_to_df,
            seed_ascii_chars
        )
        from helpers.io import (
            dump_parquet,
            latest_parquet,
            register_custom_palette
        )
        from helpers.cli import (
            yes_no
        )
        import helpers.aliases as mb_alias
        import mbAPI
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
        JSON_DIR = HERE / "JSON"
        PALETTES_FILE = JSON_DIR / "palettes.json"
        os.environ.pop("FLASK_APP", None)
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
            def __init__(self):
                self.total_pages = 0
                self.current_page = 0

            def update(self, current_page, total_pages=None):
                if total_pages is not None:
                    self.total_pages = total_pages
                self.current_page = current_page
                if self.total_pages > 0 and progress_callback:
                    percentage = 30 + (self.current_page / self.total_pages) * 40
                    msg = f"Page {self.current_page}/{self.total_pages}"
                    progress_callback("Fetching from Last.fm API", percentage, msg)

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
        # ascii table stats
        seed_ascii_chars(engine)
        logging.info(ascii_freq(engine))
        if progress_callback:
            progress_callback("Finalizing", 90, "Saving to parquet files")
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
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"Error: {str(e)}"
        logging.exception(f"Unexpected error in data gathering workflow: {error_msg}\n{error_details}")
        if progress_callback:
            progress_callback("Error", 100, error_msg)
        else:
            print(error_msg)
            print("Check the log file for more details.")
        raise  # Re-raise to be caught by the calling function


# ────────────────────────────────────────────────────────────────
# CLI-based interface starter
# ────────────────────────────────────────────────────────────────
def get_db_statistics():
    """Fetch and format database statistics."""
    try:
        from DB import engine
        from sqlalchemy import text
        import pandas as pd
        # SQL query similar to the one in DB/testerqueries.sql
        stats_query = """
        SELECT
          'artist_mbid' AS column_name,
          COUNT(*) AS total_rows,
          COUNT(CASE WHEN artist_mbid IS NOT NULL AND TRIM(artist_mbid) != '' THEN 1 END) AS non_null_count,
          COUNT(CASE WHEN artist_mbid IS NULL OR TRIM(artist_mbid) = '' THEN 1 END) AS null_count,
          ROUND(COUNT(CASE WHEN artist_mbid IS NOT NULL AND TRIM(artist_mbid) != '' THEN 1 END) / COUNT(*) * 100, 2) AS non_null_percentage,
          ROUND(COUNT(CASE WHEN artist_mbid IS NULL OR TRIM(artist_mbid) = '' THEN 1 END) / COUNT(*) * 100, 2) AS null_percentage
        FROM scrobble
        UNION ALL
        SELECT
          'album_title',
          COUNT(*),
          COUNT(CASE WHEN album_title IS NOT NULL AND TRIM(album_title) != '' THEN 1 END),
          COUNT(CASE WHEN album_title IS NULL OR TRIM(album_title) = '' THEN 1 END),
          ROUND(COUNT(CASE WHEN album_title IS NOT NULL AND TRIM(album_title) != '' THEN 1 END) / COUNT(*) * 100, 2),
          ROUND(COUNT(CASE WHEN album_title IS NULL OR TRIM(album_title) = '' THEN 1 END) / COUNT(*) * 100, 2)
        FROM scrobble
        UNION ALL
        SELECT
          'track_title',
          COUNT(*),
          COUNT(CASE WHEN track_title IS NOT NULL AND TRIM(track_title) != '' THEN 1 END),
          COUNT(CASE WHEN track_title IS NULL OR TRIM(track_title) = '' THEN 1 END),
          ROUND(COUNT(CASE WHEN track_title IS NOT NULL AND TRIM(track_title) != '' THEN 1 END) / COUNT(*) * 100, 2),
          ROUND(COUNT(CASE WHEN track_title IS NULL OR TRIM(track_title) = '' THEN 1 END) / COUNT(*) * 100, 2)
        FROM scrobble
        UNION ALL
        SELECT
          'artist_name',
          COUNT(*),
          COUNT(CASE WHEN artist_name IS NOT NULL AND TRIM(artist_name) != '' THEN 1 END),
          COUNT(CASE WHEN artist_name IS NULL OR TRIM(artist_name) = '' THEN 1 END),
          ROUND(COUNT(CASE WHEN artist_name IS NOT NULL AND TRIM(artist_name) != '' THEN 1 END) / COUNT(*) * 100, 2),
          ROUND(COUNT(CASE WHEN artist_name IS NULL OR TRIM(artist_name) = '' THEN 1 END) / COUNT(*) * 100, 2)
        FROM scrobble
        UNION ALL
        SELECT
          'play_time',
          COUNT(*),
          COUNT(CASE WHEN play_time IS NOT NULL THEN 1 END),
          COUNT(CASE WHEN play_time IS NULL THEN 1 END),
          ROUND(COUNT(CASE WHEN play_time IS NOT NULL THEN 1 END) / COUNT(*) * 100, 2),
          ROUND(COUNT(CASE WHEN play_time IS NULL THEN 1 END) / COUNT(*) * 100, 2)
        FROM scrobble
        """
        with engine.connect() as connection:
            result = connection.execute(text(stats_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception as e:
        logging.error(f"Error fetching database statistics: {e}")
        return None


def start_gui():
    """Start the CanonFodder CLI-based interface."""
    # Initialize and start the CLI interface
    cli = CliInterface()
    cli.start()


# ────────────────────────────────────────────────────────────────
# CLI interface implementation
# ────────────────────────────────────────────────────────────────

class CliInterface:
    """
    CLI-based interface for CanonFodder with animated typing and 
    steampunk-inspired terminal aesthetics.
    """

    def __init__(self):
        """Initialize the CLI interface."""
        self.username = None
        self.stdscr = None
        # Check for saved username
        self._load_username()

    def _load_username(self):
        """Load username from environment or config file."""
        self.username = os.getenv("LASTFM_USER")

        # Check for config file if not found in environment
        if not self.username:
            config_path = Path.home() / ".canonrc"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        for line in f:
                            if line.startswith("username="):
                                self.username = line.split("=")[1].strip()
                                break
                except Exception as e:
                    logging.warning(f"Failed to read config file: {e}")

    @staticmethod
    def _save_username(username):
        """Save username to environment and config file."""
        # Save to environment for current session
        os.environ["LASTFM_USER"] = username

        # Try to save to config file for persistence
        try:
            config_path = Path.home() / ".canonrc"
            if config_path.exists():
                with open(config_path, "r") as f:
                    lines = f.readlines()

                # Update existing username line or add new one
                username_line_found = False
                for i, line in enumerate(lines):
                    if line.startswith("username="):
                        lines[i] = f"username={username}\n"
                        username_line_found = True
                        break

                if not username_line_found:
                    lines.append(f"username={username}\n")

                with open(config_path, "w") as f:
                    f.writelines(lines)
            else:
                # Create new config file
                with open(config_path, "w") as f:
                    f.write(f"username={username}\n")
        except Exception as e:
            logging.warning(f"Failed to save username to config: {e}")

    def start(self):
        """Start the CLI interface with curses."""
        # Configure locale for proper UTF-8 support
        locale.setlocale(locale.LC_ALL, '')

        # Start the curses application
        curses.wrapper(self._main_loop)

    def _main_loop(self, stdscr):
        """Main application loop with curses screen."""
        self.stdscr = stdscr

        # Configure curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()

        # Initialize color pairs
        self._init_color_pairs()

        # Clear screen
        self.stdscr.clear()

        # Show welcome screen or main menu based on username
        if self.username:
            self._show_main_menu()
        else:
            self._show_welcome_screen()

    @staticmethod
    def _init_color_pairs():
        """Initialize curses color pairs for the interface."""
        # Basic colors - these numbers need to be between 0-7 for base colors
        # and 8-15 for bright variants in most terminals
        curses.init_pair(1, curses.COLOR_WHITE, -1)    # White on default
        curses.init_pair(2, curses.COLOR_CYAN, -1)     # Teal/Cyan on default
        curses.init_pair(3, curses.COLOR_GREEN, -1)    # Green/Lime on default
        curses.init_pair(4, curses.COLOR_RED, -1)      # Rust/Red on default
        curses.init_pair(5, curses.COLOR_BLUE, -1)     # Blue on default
        curses.init_pair(6, curses.COLOR_YELLOW, -1)   # Peach/Yellow on default
        curses.init_pair(11, curses.COLOR_MAGENTA, -1) # Mauve/Rose on default
        curses.init_pair(12, curses.COLOR_BLACK, -1)   # Black on default
        
        # Additional colors - using extended color pairs where available
        # Not all terminals support these extended colors, so these are a best-effort
        try:
            # Dark green for Sage
            curses.init_pair(13, 28, -1)  # Dark green for Sage
            # Brown/gold for Sand
            curses.init_pair(14, 130, -1) # Brown/gold for Sand
            # Gray for Slate
            curses.init_pair(15, 244, -1) # Gray for Slate
            # Light gray for Silver
            curses.init_pair(16, 250, -1) # Light gray for Silver
        except:
            # Fallback to basic colors if extended colors aren't available
            curses.init_pair(13, curses.COLOR_GREEN, -1)   # Sage (fallback)
            curses.init_pair(14, curses.COLOR_YELLOW, -1)  # Sand (fallback)
            curses.init_pair(15, curses.COLOR_WHITE, -1)   # Slate (fallback)
            curses.init_pair(16, curses.COLOR_WHITE, -1)   # Silver (fallback)
    
        # Highlight pairs
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)   # White highlight
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)    # Cyan highlight
        curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_GREEN)   # Green highlight
        curses.init_pair(10, curses.COLOR_BLACK, curses.COLOR_RED)    # Red highlight
        
        # Additional highlight pairs
        try:
            curses.init_pair(17, curses.COLOR_BLACK, curses.COLOR_MAGENTA)  # Mauve highlight
            curses.init_pair(18, curses.COLOR_BLACK, curses.COLOR_YELLOW)   # Sand highlight
            curses.init_pair(19, curses.COLOR_BLACK, 244)                   # Slate highlight
            curses.init_pair(20, curses.COLOR_BLACK, 250)                   # Silver highlight
        except:
            # Fallbacks
            curses.init_pair(17, curses.COLOR_BLACK, curses.COLOR_MAGENTA)  # Mauve highlight
            curses.init_pair(18, curses.COLOR_BLACK, curses.COLOR_YELLOW)   # Sand highlight
            curses.init_pair(19, curses.COLOR_BLACK, curses.COLOR_WHITE)    # Slate highlight
            curses.init_pair(20, curses.COLOR_BLACK, curses.COLOR_WHITE)    # Silver highlight

    def _animated_type(self, y, x, text, color_pair=0, delay=0.03, highlight=False):
        """Type text with animation effect."""
        if DISABLE_ANIMATIONS:
            # If animations disabled, just print the text immediately
            attr = curses.A_BOLD if highlight else 0
            self.stdscr.addstr(y, x, text, curses.color_pair(color_pair) | attr)
            self.stdscr.refresh()
            return

        for i, char in enumerate(text):
            attr = curses.A_BOLD if highlight else 0
            self.stdscr.addstr(y, x + i, char, curses.color_pair(color_pair) | attr)
            self.stdscr.refresh()
            time.sleep(delay * random.uniform(0.5, 1.5))

    def _show_welcome_screen(self):
        """Display the welcome screen and username prompt."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw ASCII art title
        title_lines = ["CanonFodder Terminal Interface"]

        # Calculate starting position to center the title
        start_y = max(1, (h - len(title_lines) - 10) // 2)

        # Display title with animation
        for i, line in enumerate(title_lines):
            self._animated_type(start_y + i, max(0, (w - len(line)) // 2),
                                line, color_pair=2, delay=0.004)

        # Display welcome text
        welcome_text = WELCOME_TEXT.split('\n')

        # Add a blank line after the title
        start_y += len(title_lines) + 2

        # Display each line of the welcome text
        for i, line in enumerate(welcome_text):
            wrapped_lines = [line[j:j + w - 4] for j in range(0, len(line), w - 4)]
            for j, wrapped in enumerate(wrapped_lines):
                y_pos = start_y + i + j
                if y_pos < h - 3:  # Ensure we don't write past the bottom
                    self._animated_type(y_pos, 2, wrapped, color_pair=1, delay=0.01)

        # Prompt for username
        prompt_y = start_y + len(welcome_text) + len([w for l in welcome_text for w in l.split('\n')]) + 2
        if prompt_y >= h - 3:
            prompt_y = h - 3

        self._animated_type(prompt_y, 2, "Last.fm username: ", color_pair=3, highlight=True)

        # Enable cursor for input
        curses.curs_set(1)
        curses.echo()

        # Get username input
        username = ""
        username_x = 18  # Length of "Last.fm username: "

        while True:
            self.stdscr.move(prompt_y, 2 + username_x + len(username))
            c = self.stdscr.getch()

            if c == ord('\n'):  # Enter key
                break
            elif c == 27:  # ESC key
                # Restore cursor state before returning
                curses.noecho()
                curses.curs_set(0)
                return None  # Explicit return None on ESC
            elif c == curses.KEY_BACKSPACE or c == 127:  # Backspace
                if username:
                    username = username[:-1]
                    self.stdscr.addstr(prompt_y, 2 + username_x + len(username), " ")
                    self.stdscr.refresh()
            elif 32 <= c <= 126:  # Printable ASCII
                username += chr(c)

        # Disable cursor and echo
        curses.noecho()
        curses.curs_set(0)

        # Validate username
        if not username.strip():
            # Show error message
            self.stdscr.addstr(prompt_y + 1, 2, "Username cannot be empty! Press any key to try again.",
                               curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.refresh()
            self.stdscr.getch()
            return self._show_welcome_screen()

        # Save the username
        self.username = username.strip()
        self._save_username(self.username)

        # Show success message and continue to main menu
        self._animated_type(prompt_y + 1, 2, f"Welcome, {self.username}! Loading main menu...",
                            color_pair=3, highlight=True)
        time.sleep(0.5)  # Shorter pause before continuing to menu

        # Show main menu
        return self._show_main_menu()

    def _show_main_menu(self):
        """Display the main menu with options."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw menu header
        header = f"// CANONFODDER TERMINAL INTERFACE //"
        subheader = f"// LOGGED IN AS: {self.username}"

        self._animated_type(1, (w - len(header)) // 2, header, color_pair=2, highlight=True, delay=0.01)
        self._animated_type(2, (w - len(subheader)) // 2, subheader, color_pair=2, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(3, 1, separator, curses.color_pair(2))

        # Menu options
        menu_title = "MAIN MENU"
        self._animated_type(5, (w - len(menu_title)) // 2, menu_title, color_pair=1, highlight=True, delay=0.01)

        # Fetch and display database statistics
        try:
            db_stats = get_db_statistics()
            if db_stats is not None and not db_stats.empty:
                # Display database statistics table
                stats_title = "SCROBBLES IN DB"
                self._animated_type(7, 4, stats_title, color_pair=3, highlight=True, delay=0.01)
                
                # Table headers with padding to ensure alignment
                headers = [
                    "Column".ljust(12),
                    "Total".rjust(8),
                    "Non-Null".rjust(10),
                    "Null".rjust(8),
                    "Non-Null %".rjust(11),
                    "Null %".rjust(8)
                ]
                header_row = " ".join(headers)
                self.stdscr.addstr(8, 4, header_row, curses.color_pair(2) | curses.A_BOLD)
                
                # Add a line separator
                self.stdscr.addstr(9, 4, "─" * (len(header_row)), curses.color_pair(2))
                
                # Display each row of statistics
                for i, row in db_stats.iterrows():
                    if i < 5:  # Limit to 5 rows to save space
                        row_text = (
                            f"{row['column_name']:<12} "
                            f"{row['total_rows']:>8} "
                            f"{row['non_null_count']:>10} "
                            f"{row['null_count']:>8} "
                            f"{row['non_null_percentage']:>11.2f} "
                            f"{row['null_percentage']:>8.2f}"
                        )
                        self.stdscr.addstr(10 + i, 4, row_text, curses.color_pair(1))
                
                # Add spacing after the table
                option_start_y = 16
            else:
                # If no statistics available, display a message
                self.stdscr.addstr(7, 4, "No database statistics available", curses.color_pair(4))
                option_start_y = 9
        except Exception as e:
            # If there's an error, display it and continue
            error_msg = f"Error displaying statistics: {str(e)}"
            if len(error_msg) > w - 8:
                error_msg = error_msg[:w-11] + "..."
            self.stdscr.addstr(7, 4, error_msg, curses.color_pair(4))
            option_start_y = 9
        
        options = [
            "1. Data Gathering - Fetch new scrobbles from Last.fm",
            "2. User Input / Canonization - Clean and normalize artist data",
            "3. Statistics & Visualizations - View insights and reports",
            "4. FULL REFRESH - Drop & re-sync ALL scrobbles",
            "5. Exit"
        ]
    
        for i, option in enumerate(options):
            self._animated_type(option_start_y + i, 4, option, color_pair=1, delay=0.01)

        # Draw instructions
        instructions = "Press a number key to select an option..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Bottom border
        self.stdscr.addstr(h - 2, 1, separator, curses.color_pair(2))
        self.stdscr.refresh()

        # Handle menu selection
        while True:
            # Highlight the menu options to clarify selection
            for i, option in enumerate(options):
                self.stdscr.addstr(option_start_y + i, 4, option, curses.color_pair(1) | curses.A_BOLD)
            self.stdscr.refresh()

            c = self.stdscr.getch()

            if c == ord('1'):
                self._run_data_gathering()
                return
            elif c == ord('2'):
                self._run_canonization()
                return
            elif c == ord('3'):
                self._run_statistics()
                return
            elif c == ord('4'):  # ← flush + refetch
                self._run_full_refresh();
                return
            elif c == ord('5') or c in (ord('q'), 27):
                return

    def _show_progress_screen(self, title):
        """Show a progress screen for long-running operations."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        # Draw title
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))
        # Initial progress message
        self.stdscr.addstr(4, 2, "Initializing...", curses.color_pair(1))
        # Draw progress bar outline
        bar_width = w - 6
        self.stdscr.addstr(6, 2, "┌" + "─" * bar_width + "┐", curses.color_pair(1))
        self.stdscr.addstr(7, 2, "│" + " " * bar_width + "│", curses.color_pair(1))
        self.stdscr.addstr(8, 2, "└" + "─" * bar_width + "┘", curses.color_pair(1))
        # Additional status message area
        self.stdscr.addstr(10, 2, "Status: Starting up...", curses.color_pair(1))
        self.stdscr.refresh()

        # Return the progress update function to be used as a callback
        def update_progress(task, percentage, message=None):
            # Update task name
            self.stdscr.addstr(4, 2, " " * (w - 4))  # Clear the line
            self.stdscr.addstr(4, 2, f"Task: {task}", curses.color_pair(1))
            # Update progress bar
            filled_width = int((percentage / 100) * bar_width)
            self.stdscr.addstr(7, 3, "█" * filled_width + " " * (bar_width - filled_width), curses.color_pair(3))
            # Show percentage
            percent_str = f" {percentage:.1f}% "
            percent_pos = min(3 + filled_width - len(percent_str) // 2, w - len(percent_str) - 3)
            if percent_pos < 3:
                percent_pos = 3
            self.stdscr.addstr(7, percent_pos, percent_str, curses.color_pair(7))
            # Update status message
            if message:
                self.stdscr.addstr(10, 2, " " * (w - 4))  # Clear the line
                status_text = f"Status: {message}"
                # Truncate if too long
                if len(status_text) > w - 4:
                    status_text = status_text[:w - 7] + "..."
                self.stdscr.addstr(10, 2, status_text, curses.color_pair(1))
            self.stdscr.refresh()
        return update_progress

    def _run_data_gathering(self):
        """Run the data gathering workflow with progress updates."""
        # Show progress screen
        progress_callback = self._show_progress_screen("DATA GATHERING")

        # Run in a thread to keep UI responsive
        def run_task():
            try:
                run_data_gathering_workflow(self.username, progress_callback)
                # Show completion message
                haa, duplavee = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, " " * (duplavee - 4))  # Clear any previous message
                self.stdscr.addstr(12, 2, "Data gathering completed successfully!",
                                   curses.color_pair(3) | curses.A_BOLD)
                self.stdscr.addstr(14, 2, "Press any key to return to the menu...", curses.color_pair(1))
                self.stdscr.refresh()
            except Exception as e:
                # Show error message
                haa, duplavee = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, " " * (duplavee - 4))  # Clear any previous message
                error_msg = f"Error: {str(e)}"
                # Truncate if too long
                if len(error_msg) > duplavee - 4:
                    error_msg = error_msg[:duplavee - 7] + "..."
                self.stdscr.addstr(12, 2, error_msg, curses.color_pair(4) | curses.A_BOLD)
                self.stdscr.addstr(14, 2, "Press any key to return to the menu...", curses.color_pair(1))
                self.stdscr.refresh()
                logging.exception("Error during data gathering")
        # Start the task in a separate thread
        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()
        # Wait for the thread to complete
        while thread.is_alive():
            # Check for key press to cancel (ESC key)
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                h, w = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, "Attempting to cancel operation...",
                                   curses.color_pair(4) | curses.A_BOLD)
                self.stdscr.refresh()
                # Can't forcibly stop the thread, but we can signal it's done
                thread.join(timeout=0.1)
                if thread.is_alive():
                    # If thread is still running, wait for it to finish naturally
                    self.stdscr.addstr(13, 2, "Please wait for current operation to complete...",
                                       curses.color_pair(1))
                    self.stdscr.refresh()
        # Wait for a keypress before returning to menu
        self.stdscr.getch()
        # Return to main menu
        self._show_main_menu()

    def _run_canonization(self):
        """Placeholder for canonization workflow."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        # Draw title
        title = "USER INPUT / CANONIZATION"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))
        # Display message
        message = "This feature will be implemented in a future update."
        self._animated_type(4, (w - len(message)) // 2, message, color_pair=1, delay=0.01)
        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)
        # Wait for keypress
        self.stdscr.getch()
        # Return to main menu
        self._show_main_menu()

    def _run_statistics(self):
        """Placeholder for statistics workflow."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        # Draw title
        title = "STATISTICS & VISUALIZATIONS"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))
        # Display message
        message = "This feature will be implemented in a future update."
        self._animated_type(4, (w - len(message)) // 2, message, color_pair=1, delay=0.01)
        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)
        # Wait for keypress
        self.stdscr.getch()
        # Return to main menu
        self._show_main_menu()

    def _run_full_refresh(self):
        """Flush *scrobble* and pull a complete history from Last.fm."""
        # Ask for confirmation – destructive!
        confirm_text = "This will DELETE every scrobble in the database. Proceed? (y/N) "
        self.stdscr.clear()
        self._animated_type(2, 2, confirm_text, color_pair=4, highlight=True)
        self.stdscr.refresh()
        curses.echo(); curses.curs_set(1)
        ch = self.stdscr.getstr(2, 2 + len(confirm_text)).decode().lower()
        curses.noecho(); curses.curs_set(0)
        if ch != 'y':
            self._show_main_menu(); return
        # Re-use the nice progress screen
        progress = self._show_progress_screen("FULL REFRESH")

        def task():
            from DB import SessionLocal
            from DB.models import Scrobble
            from sqlalchemy import text, inspect
            # 1) Hard-delete every row (TRUNCATE preferred where available)
            progress("Clearing table", 5, "Dropping existing rows…")
            try:
                with SessionLocal() as sess:
                    dialect = sess.bind.dialect.name
                    if dialect in ("mysql", "postgresql"):
                        sess.execute(text("TRUNCATE TABLE scrobble"))
                    else:   # SQLite has no TRUNCATE
                        sess.execute(delete(Scrobble))
                    sess.commit()
            except Exception as exc:
                progress("Error", 100, f"Purging failed: {exc}")
                return
            # 2) Vacuum / reset autoinc if SQLite
            if sess.bind.dialect.name == "sqlite":
                sess.execute(text("VACUUM")); sess.commit()
            progress("Fetching", 10, "Requesting full history from API")
            # 3) Call existing pipeline – empty table → fetches everything
            run_data_gathering_workflow(self.username, progress)
            progress("Complete", 100, "Full refresh done")
        t = threading.Thread(target=task, daemon=True); t.start()
        while t.is_alive():
            if self.stdscr.getch() == 27:        # allow ESC cancel display
                pass
        self.stdscr.getch()
        self._show_main_menu()


# ────────────────────────────────────────────────────────────────
# CLI entry point for application
# ────────────────────────────────────────────────────────────────
def _cli_entry() -> int:
    """Handle command-line arguments and start the appropriate interface."""
    parser = argparse.ArgumentParser(description="CanonFodder: Last.fm scrobble analysis tool")
    parser.add_argument(
        "--legacy-cli",
        action="store_true",
        help="Run in legacy CLI mode (direct data gathering).",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Alias for --legacy-cli for backward compatibility.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable typing animations in CLI interface.",
    )
    parser.add_argument(
        "--enrich-artist-mbid",
        action="store_true",
        help="Fetch missing artist_mbid values from Last.fm API.",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Last.fm username to use with --enrich-artist-mbid or --legacy-cli",
    )
    args = parser.parse_args()

    # Check for special operations
    if args.enrich_artist_mbid:
        # Get username from args, env, or config
        username = args.username or os.getenv("LASTFM_USER")
        
        if not username:
            # Try to read from config file
            config_path = Path.home() / ".canonrc"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        for line in f:
                            if line.startswith("username="):
                                username = line.split("=")[1].strip()
                                break
                except Exception:
                    pass

        if not username:
            print("No username provided. Please specify with --username or set LASTFM_USER environment variable.")
            return 1

        print(f"Enriching artist_mbid values for {username}...")
        import lfAPI

        # Setup console progress display
        from tqdm import tqdm
        pbar = tqdm(total=100, desc="Fetching MBIDs")
        
        def progress_callback(task, percentage, message=None):
            pbar.n = percentage
            if message:
                pbar.set_description(f"{task}: {message}")
            else:
                pbar.set_description(task)
            pbar.refresh()
        
        result = lfAPI.enrich_artist_mbids(username, progress_callback)
        
        pbar.close()
        
        if result["status"] == "success":
            print(f"✅ Success: {result['message']}")
        else:
            print(f"❌ Error: {result['message']}")
        return 0

    # For backward compatibility, treat --cli the same as --legacy-cli
    if args.legacy_cli or args.cli:
        # In legacy CLI mode, let the function handle username input directly
        # If username was provided, pass it to the function
        return run_data_gathering_workflow(args.username) or 0
    else:
        # Initialize terminal colors
        init_colors()

        # Set global animation flag
        if args.no_animation:
            global DISABLE_ANIMATIONS
            DISABLE_ANIMATIONS = True

        # Start the CLI interface
        start_gui()
        return 0


if __name__ == "__main__":
    # Define global animation flag
    DISABLE_ANIMATIONS = False

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    sys.exit(_cli_entry())
