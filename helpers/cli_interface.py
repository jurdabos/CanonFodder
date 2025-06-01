"""
CLI interface for CanonFodder.

This module provides a curses-based CLI interface for the application with
animated typing and steampunk-inspired terminal aesthetics.
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
import platform
from pathlib import Path
from typing import Optional
from sqlalchemy import select, func, text, delete
# Import from helpers
from helpers.progress import ProgressCallback
from helpers.cli import choose_lastfm_user
# Import from DB
from DB import SessionLocal
from DB.models import ArtistInfo, Scrobble, Base
from DB.ops import populate_artist_info_from_scrobbles

# Import from corefunc
from corefunc.data_gathering import run_full_refresh
from corefunc.model_server import start_server, stop_server, get_server_status
from corefunc.canonizer import apply_previous
from corefunc.dataprofiler import run_profiling
from corefunc.visualizations import (
    load_data, scrobbles_per_year, scrobbles_per_month, 
    monthly_ridgeline, seasonal_decomposition, top_artists_bar,
    artist_distribution_violin, find_trusted_companions, save_figure
)

# Import API modules
from HTTP import mbAPI
from HTTP import lfAPI

# Constants
APP_TITLE = "CanonFodder"
WELCOME_TEXT = (
    "CanonFodder is a SINGLE-USER installation – one database, one Last.fm user.\n\n"
    "To analyse another account you currently need to install CanonFodder again "
    "into a separate directory (or virtualenv) so the databases do not overlap.\n\n"
    "Please enter the Last.fm user name that shall be analysed in THIS installation."
)

# Global animation flag
DISABLE_ANIMATIONS = False


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


def get_db_statistics():
    """
    Get statistics about the database tables.

    Returns
    -------
    pandas.DataFrame
        DataFrame with statistics about the database tables
    """
    try:
        import pandas as pd
        from sqlalchemy import inspect

        # Connect to the database
        from DB import engine

        # Get the inspector
        inspector = inspect(engine)

        # Get table names
        table_names = inspector.get_table_names()

        # Focus on the scrobble table
        if 'scrobble' not in table_names:
            return None

        # Get column information for the scrobble table
        columns = inspector.get_columns('scrobble')
        column_names = [col['name'] for col in columns]

        # Create a DataFrame to store statistics
        stats_data = []

        # Connect to the database and get statistics
        with engine.connect() as conn:
            # Get total row count
            total_rows = conn.execute(text("SELECT COUNT(*) FROM scrobble")).scalar()

            if total_rows == 0:
                return None

            # Get statistics for each column
            for col_name in column_names:
                # Count non-null values
                non_null_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM scrobble WHERE {col_name} IS NOT NULL")
                ).scalar()

                # Calculate null count and percentages
                null_count = total_rows - non_null_count
                non_null_percentage = (non_null_count / total_rows) * 100
                null_percentage = (null_count / total_rows) * 100

                # Add to statistics data
                stats_data.append({
                    'column_name': col_name,
                    'total_rows': total_rows,
                    'non_null_count': non_null_count,
                    'null_count': null_count,
                    'non_null_percentage': non_null_percentage,
                    'null_percentage': null_percentage
                })

        # Create DataFrame from statistics data
        stats_df = pd.DataFrame(stats_data)

        return stats_df

    except Exception as e:
        logging.error(f"Error getting database statistics: {e}")
        return None


def start_gui():
    """Start the CanonFodder CLI-based interface."""
    # Initialize and start the CLI interface
    cli = CliInterface()
    cli.start()


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
        curses.init_pair(1, curses.COLOR_WHITE, -1)  # White on default
        curses.init_pair(2, curses.COLOR_CYAN, -1)  # Teal/Cyan on default
        curses.init_pair(3, curses.COLOR_GREEN, -1)  # Green/Lime on default
        curses.init_pair(4, curses.COLOR_RED, -1)  # Rust/Red on default
        curses.init_pair(5, curses.COLOR_BLUE, -1)  # Blue on default
        curses.init_pair(6, curses.COLOR_YELLOW, -1)  # Peach/Yellow on default
        curses.init_pair(11, curses.COLOR_MAGENTA, -1)  # Mauve/Rose on default
        curses.init_pair(12, curses.COLOR_BLACK, -1)  # Black on default

        # Additional colors - using extended color pairs where available
        # Not all terminals support these extended colors, so these are a best-effort
        try:
            # Dark green for Sage
            curses.init_pair(13, 28, -1)  # Dark green for Sage
            # Brown/gold for Sand
            curses.init_pair(14, 130, -1)  # Brown/gold for Sand
            # Gray for Slate
            curses.init_pair(15, 244, -1)  # Gray for Slate
            # Light gray for Silver
            curses.init_pair(16, 250, -1)  # Light gray for Silver
        except:
            # Fallback to basic colors if extended colors aren't available
            curses.init_pair(13, curses.COLOR_GREEN, -1)  # Sage (fallback)
            curses.init_pair(14, curses.COLOR_YELLOW, -1)  # Sand (fallback)
            curses.init_pair(15, curses.COLOR_WHITE, -1)  # Slate (fallback)
            curses.init_pair(16, curses.COLOR_WHITE, -1)  # Silver (fallback)

        # Highlight pairs
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)  # White highlight
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)  # Cyan highlight
        curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Green highlight
        curses.init_pair(10, curses.COLOR_BLACK, curses.COLOR_RED)  # Red highlight

        # Additional highlight pairs
        try:
            curses.init_pair(17, curses.COLOR_BLACK, curses.COLOR_MAGENTA)  # Mauve highlight
            curses.init_pair(18, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Sand highlight
            curses.init_pair(19, curses.COLOR_BLACK, 244)  # Slate highlight
            curses.init_pair(20, curses.COLOR_BLACK, 250)  # Silver highlight
        except:
            # Fallbacks
            curses.init_pair(17, curses.COLOR_BLACK, curses.COLOR_MAGENTA)  # Mauve highlight
            curses.init_pair(18, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Sand highlight
            curses.init_pair(19, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Slate highlight
            curses.init_pair(20, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Silver highlight

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
                # Adding a line separator
                self.stdscr.addstr(9, 4, "─" * (len(header_row)), curses.color_pair(2))
                # Displaying each row of statistics
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
                option_start_y = 16
            else:
                # If no statistics available, display a message
                self.stdscr.addstr(7, 4, "No database statistics available", curses.color_pair(4))
                option_start_y = 9
        except Exception as e:
            # If there's an error, display it and continue
            error_msg = f"Error displaying statistics: {str(e)}"
            if len(error_msg) > w - 8:
                error_msg = error_msg[:w - 11] + "..."
            self.stdscr.addstr(7, 4, error_msg, curses.color_pair(4))
            option_start_y = 9
        options = [
            "1. Data Gathering",
            "2. Canonization",
            "3. EDA",
            "4. Vizz",
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
                self._show_data_gathering_menu()
                return
            elif c == ord('2'):
                self._run_canonization()
                return
            elif c == ord('3'):
                self._run_eda()
                return
            elif c == ord('4'):
                self._run_visualizations()
                return
            elif c == ord('5') or c in (ord('q'), 27):
                return

    def _show_data_gathering_menu(self):
        """Display the Data Gathering submenu with options."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw menu header
        header = f"// DATA GATHERING MENU //"
        subheader = f"// LOGGED IN AS: {self.username}"

        self._animated_type(1, (w - len(header)) // 2, header, color_pair=2, highlight=True, delay=0.01)
        self._animated_type(2, (w - len(subheader)) // 2, subheader, color_pair=2, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(3, 1, separator, curses.color_pair(2))

        # Menu options
        menu_title = "DATA GATHERING OPTIONS"
        self._animated_type(5, (w - len(menu_title)) // 2, menu_title, color_pair=1, highlight=True, delay=0.01)

        options = [
            "1. Incremental Pull",
            "2. Full refresh",
            "3. Return to main menu"
        ]

        option_start_y = 7
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
                self._run_full_refresh()
                return
            elif c == ord('3') or c in (ord('q'), 27):
                self._show_main_menu()
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

        # Return a proper progress callback class instance
        class CursesProgressCallback:
            def __init__(self, stdscr, width, bar_width):
                self.stdscr = stdscr
                self.width = width
                self.bar_width = bar_width

            def __call__(self, task: str, percentage: float, message: Optional[str] = None) -> None:
                # Update task name
                self.stdscr.addstr(4, 2, " " * (self.width - 4))  # Clear the line
                self.stdscr.addstr(4, 2, f"Task: {task}", curses.color_pair(1))

                # Update progress bar
                filled_width = int((percentage / 100) * self.bar_width)
                self.stdscr.addstr(7, 3, "█" * filled_width + " " * (self.bar_width - filled_width),
                                   curses.color_pair(3))

                # Show percentage
                percent_str = f" {percentage:.1f}% "
                percent_pos = min(3 + filled_width - len(percent_str) // 2, self.width - len(percent_str) - 3)
                if percent_pos < 3:
                    percent_pos = 3
                self.stdscr.addstr(7, percent_pos, percent_str, curses.color_pair(7))

                # Update status message
                if message:
                    self.stdscr.addstr(10, 2, " " * (self.width - 4))  # Clear the line
                    status_text = f"Status: {message}"
                    # Only truncate if necessary (not enough space on screen)
                    if len(status_text) > self.width - 4:
                        # Check if it's an SQL log or similar that we want to show in full if possible
                        if "[SQL:" in message or message.startswith("SQL:"):
                            # Display as much as possible without scrolling
                            status_text = status_text[:self.width - 4]
                        else:
                            # For other messages, truncate with ellipsis
                            status_text = status_text[:self.width - 7] + "..."
                    self.stdscr.addstr(10, 2, status_text, curses.color_pair(1))

                self.stdscr.refresh()

        return CursesProgressCallback(self.stdscr, w, bar_width)

    def _run_data_gathering(self):
        """Run the data gathering workflow with progress updates."""
        # Show progress screen
        progress_callback = self._show_progress_screen("DATA GATHERING")

        # Run in a thread to keep UI responsive
        def run_task():
            try:
                # Use the new pipeline module
                from corefunc.pipeline import run_incremental_pipeline

                result = run_incremental_pipeline(self.username, progress_callback)

                # Show completion message
                haa, duplavee = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, " " * (duplavee - 4))  # Clear any previous message

                if result['status'] == 'success':
                    fetch_result = result.get('fetch_result', {})
                    enrich_result = result.get('enrich_result', {})
                    clean_result = result.get('clean_result', {})

                    new_scrobbles = fetch_result.get('new_scrobbles', 0)
                    processed = enrich_result.get('processed', 0)
                    created = enrich_result.get('created', 0)
                    updated = enrich_result.get('updated', 0)
                    cleaned = clean_result.get('cleaned', 0)

                    success_msg = f"Data gathering completed successfully!"
                    self.stdscr.addstr(12, 2, success_msg, curses.color_pair(3) | curses.A_BOLD)

                    if new_scrobbles > 0:
                        details_msg = f"Fetched {new_scrobbles} new scrobbles."
                        self.stdscr.addstr(13, 2, details_msg, curses.color_pair(1))

                    artist_msg = f"Artists: processed {processed}, created {created}, updated {updated}, cleaned {cleaned}"
                    self.stdscr.addstr(14, 2, artist_msg, curses.color_pair(1))
                else:
                    error_msg = f"Error: {result['message']}"
                    # Only truncate if necessary (not enough space on screen)
                    if len(error_msg) > duplavee - 4:
                        # Check if it's an SQL log or similar that we want to show in full if possible
                        if "[SQL:" in result['message'] or result['message'].startswith("SQL:"):
                            # Display as much as possible without scrolling
                            error_msg = error_msg[:duplavee - 4]
                        else:
                            # For other messages, truncate with ellipsis
                            error_msg = error_msg[:duplavee - 7] + "..."
                    self.stdscr.addstr(12, 2, error_msg, curses.color_pair(4) | curses.A_BOLD)

                self.stdscr.addstr(16, 2, "Press any key to return to the menu...", curses.color_pair(1))
                self.stdscr.refresh()
            except Exception as e:
                # Show error message
                haa, duplavee = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, " " * (duplavee - 4))  # Clear any previous message
                error_msg = f"Error: {str(e)}"
                # Only truncate if necessary (not enough space on screen)
                if len(error_msg) > duplavee - 4:
                    # Check if it's an SQL log or similar that we want to show in full if possible
                    if "[SQL:" in str(e) or str(e).startswith("SQL:"):
                        # Display as much as possible without scrolling
                        error_msg = error_msg[:duplavee - 4]
                    else:
                        # For other messages, truncate with ellipsis
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
        # Return to data gathering menu
        self._show_data_gathering_menu()

    def _run_canonization(self):
        """Run the canonization workflow with automatic model server start."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "CANONIZATION"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Check model server status
        status = get_server_status()
        is_running = status["is_running"]

        # Start the model server if not running
        if not is_running:
            self._animated_type(4, 4, "Starting model server...", color_pair=3, delay=0.01)
            self.stdscr.refresh()

            success = start_server()

            if success:
                self._animated_type(5, 4, "Model server started successfully!", color_pair=3, delay=0.01)
            else:
                self._animated_type(5, 4, "Failed to start model server. Continuing without it.", color_pair=4, delay=0.01)
        else:
            self._animated_type(4, 4, "Model server is already running.", color_pair=3, delay=0.01)

        self.stdscr.refresh()

        # Loading message
        self._animated_type(7, 4, "Loading data for canonization...", color_pair=1, delay=0.01)
        self.stdscr.refresh()

        # Run canonization in a separate thread
        canonization_results = {}

        def run_canonization():
            try:
                # Load data
                import pandas as pd
                from helpers.io import latest_parquet

                # Try to load from parquet first
                try:
                    data = pd.read_parquet(latest_parquet())
                    canonization_results["data_source"] = "parquet"
                except Exception:
                    # Fall back to database
                    from DB.ops import load_scrobble_table_from_db_to_df
                    from DB import engine
                    data, _ = load_scrobble_table_from_db_to_df(engine)
                    canonization_results["data_source"] = "database"

                if data is None or data.empty:
                    canonization_results["success"] = False
                    canonization_results["error"] = "No data found"
                    return

                # Apply previous canonization
                data = apply_previous(data)

                # Store results
                canonization_results["success"] = True
                canonization_results["row_count"] = len(data)
                canonization_results["artist_count"] = data["Artist"].nunique()

            except Exception as e:
                canonization_results["success"] = False
                canonization_results["error"] = str(e)

        # Run in a thread to keep UI responsive
        thread = threading.Thread(target=run_canonization)
        thread.daemon = True
        thread.start()

        # Show a spinner while waiting
        spinner_chars = ['|', '/', '-', '\\']
        spinner_idx = 0

        while thread.is_alive():
            # Update spinner
            spinner_text = f"Processing data {spinner_chars[spinner_idx]}"
            self.stdscr.addstr(9, 4, spinner_text, curses.color_pair(3))
            self.stdscr.refresh()

            # Increment spinner index
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)

            # Sleep briefly
            time.sleep(0.1)

            # Check for ESC key to cancel
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                self.stdscr.addstr(11, 4, "Cancellation requested. Please wait for current operations to complete...",
                                  curses.color_pair(4))
                self.stdscr.refresh()

        # Clear processing messages
        self.stdscr.addstr(9, 4, " " * (w - 8), curses.color_pair(0))

        # Display results
        if canonization_results.get("success", False):
            result_msg = "Canonization completed successfully!"
            self._animated_type(9, 4, result_msg, color_pair=3, highlight=True, delay=0.01)

            # Show statistics
            stats_y = 11
            self._animated_type(stats_y, 4, "Statistics:", color_pair=1, highlight=True, delay=0.01)
            self._animated_type(stats_y + 1, 6, f"Data source: {canonization_results.get('data_source', 'unknown')}", color_pair=1, delay=0.01)
            self._animated_type(stats_y + 2, 6, f"Total rows: {canonization_results.get('row_count', 0)}", color_pair=1, delay=0.01)
            self._animated_type(stats_y + 3, 6, f"Unique artists: {canonization_results.get('artist_count', 0)}", color_pair=1, delay=0.01)
        else:
            error_msg = "Error during canonization:"
            self._animated_type(9, 4, error_msg, color_pair=4, highlight=True, delay=0.01)

            if "error" in canonization_results:
                error_details = canonization_results["error"]
                # Truncate if too long
                if len(error_details) > w - 8:
                    error_details = error_details[:w - 11] + "..."
                self._animated_type(11, 4, error_details, color_pair=4, delay=0.01)

        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Wait for keypress
        self.stdscr.getch()

        # Return to main menu
        self._show_main_menu()

    def _run_aliases_refresh(self):
        """Refresh artist aliases from MusicBrainz."""
        progress_callback = self._show_progress_screen("REFRESH ARTIST ALIASES")

        def run_task():
            try:
                progress_callback("Initializing", 5, "Checking database...")

                # Get count of artists with MBIDs

                with SessionLocal() as session:
                    total_artists = session.execute(
                        select(func.count()).where(ArtistInfo.mbid.is_not(None))
                    ).scalar_one()

                    progress_callback("Scanning", 10, f"Found {total_artists} artists with MBIDs")

                    # Get all artists with MBIDs
                    artists = session.execute(
                        select(ArtistInfo).where(ArtistInfo.mbid.is_not(None))
                    ).scalars().all()

                    if not artists:
                        progress_callback("Complete", 100, "No artists with MBIDs found")
                        return

                    # Process each artist
                    success_count = 0
                    error_count = 0

                    for i, artist in enumerate(artists):
                        percentage = 10 + (i / total_artists) * 85
                        progress_callback("Updating", percentage,
                                          f"Processing {artist.artist_name} ({i + 1}/{total_artists})")

                        try:
                            # Get aliases from MusicBrainz
                            aliases = mbAPI.get_aliases(artist.mbid)

                            # Update artist record
                            if aliases:
                                aliases_str = ','.join(aliases)
                                artist.aliases = aliases_str
                                session.commit()
                                success_count += 1
                            else:
                                # Empty string to mark as processed
                                artist.aliases = ""
                                session.commit()
                        except Exception as e:
                            error_count += 1
                            progress_callback("Error", percentage, f"Error with {artist.artist_name}: {str(e)}")

                    # Show completion message
                    progress_callback("Complete", 95, f"Updated {success_count} artists ({error_count} errors)")

                    # Get final stats
                    non_empty_count = session.execute(
                        select(func.count()).where(
                            ArtistInfo.aliases.is_not(None),
                            ArtistInfo.aliases != ""
                        )
                    ).scalar_one()

                    progress_callback("Complete", 100, f"Done! {non_empty_count} artists now have aliases")

            except Exception as e:
                progress_callback("Error", 100, f"Operation failed: {str(e)}")

        # Run the task in a thread
        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()

        # Wait for the thread to complete
        while thread.is_alive():
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                h, w = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, "Attempting to cancel operation...",
                                   curses.color_pair(4) | curses.A_BOLD)
                self.stdscr.refresh()
                thread.join(timeout=0.1)

        # Wait for keypress
        self.stdscr.addstr(14, 2, "Press any key to return to the menu...", curses.color_pair(1))
        self.stdscr.refresh()
        self.stdscr.getch()

        # Return to data gathering menu
        self._show_data_gathering_menu()

    def _run_eda(self):
        """Run exploratory data analysis using dataprofiler."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "EXPLORATORY DATA ANALYSIS"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Loading message
        loading_msg = "Loading data for analysis..."
        self._animated_type(4, (w - len(loading_msg)) // 2, loading_msg, color_pair=3, delay=0.01)
        self.stdscr.refresh()

        # Run data profiling in a separate thread
        profiling_results = {}

        def run_eda():
            try:
                # Load data
                import pandas as pd
                from helpers.io import latest_parquet

                # Try to load from parquet first
                try:
                    data = pd.read_parquet(latest_parquet())
                    profiling_results["data_source"] = "parquet"
                except Exception:
                    # Fall back to database
                    from DB.ops import load_scrobble_table_from_db_to_df
                    from DB import engine
                    data, _ = load_scrobble_table_from_db_to_df(engine)
                    profiling_results["data_source"] = "database"

                if data is None or data.empty:
                    profiling_results["success"] = False
                    profiling_results["error"] = "No data found"
                    return

                # Run profiling
                profile_result = run_profiling(data)

                # Also run data profiling from pipeline
                from corefunc.pipeline import run_data_profiling
                from helpers.progress import ProgressCallback

                # Create a simple progress callback for data profiling
                class SimpleProgressCallback(ProgressCallback):
                    def __call__(self, task: str, percentage: float, message: Optional[str] = None) -> None:
                        # We don't need to do anything here since we're already showing a spinner
                        pass

                run_data_profiling(progress_callback=SimpleProgressCallback())

                # Store results
                profiling_results["success"] = True
                profiling_results["row_count"] = len(profile_result.df)
                profiling_results["artist_count"] = len(profile_result.artist_counts)
                profiling_results["top_artists"] = profile_result.artist_counts.head(10).to_dict()

                # Generate a quick visualization
                import matplotlib
                matplotlib.use("Agg")  # Use non-interactive backend
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Create visualization directory if it doesn't exist
                viz_dir = Path("visualizations")
                viz_dir.mkdir(exist_ok=True)

                # Create top artists plot
                plt.figure(figsize=(10, 6))
                top_artists = profile_result.artist_counts.head(10)
                sns.barplot(x=top_artists.values, y=top_artists.index, palette="viridis")
                plt.title("Top 10 Artists by Scrobble Count")
                plt.tight_layout()

                # Save the plot
                viz_path = viz_dir / "top_artists_eda.png"
                plt.savefig(viz_path)
                plt.close()

                profiling_results["viz_path"] = str(viz_path)

            except Exception as e:
                profiling_results["success"] = False
                profiling_results["error"] = str(e)

        # Run in a thread to keep UI responsive
        thread = threading.Thread(target=run_eda)
        thread.daemon = True
        thread.start()

        # Show a spinner while waiting
        spinner_chars = ['|', '/', '-', '\\']
        spinner_idx = 0

        while thread.is_alive():
            # Update spinner
            spinner_text = f"Analyzing data {spinner_chars[spinner_idx]}"
            self.stdscr.addstr(6, (w - len(spinner_text)) // 2, spinner_text, curses.color_pair(3))
            self.stdscr.refresh()

            # Increment spinner index
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)

            # Sleep briefly
            time.sleep(0.1)

            # Check for ESC key to cancel
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                self.stdscr.addstr(8, 4, "Cancellation requested. Please wait for current operations to complete...",
                                 curses.color_pair(4))
                self.stdscr.refresh()

        # Clear the screen
        self.stdscr.clear()

        # Redraw title and separator
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Check if profiling was successful
        if profiling_results.get("success", False):
            # Show success message
            success_msg = "Data analysis completed successfully!"
            self._animated_type(4, (w - len(success_msg)) // 2, success_msg, color_pair=3, highlight=True, delay=0.01)

            # Show statistics
            stats_y = 6
            self._animated_type(stats_y, 4, "Dataset Statistics:", color_pair=1, highlight=True, delay=0.01)
            self._animated_type(stats_y + 1, 6, f"Data source: {profiling_results.get('data_source', 'unknown')}", color_pair=1, delay=0.01)
            self._animated_type(stats_y + 2, 6, f"Total scrobbles: {profiling_results.get('row_count', 0)}", color_pair=1, delay=0.01)
            self._animated_type(stats_y + 3, 6, f"Unique artists: {profiling_results.get('artist_count', 0)}", color_pair=1, delay=0.01)

            # Show top artists if available
            if "top_artists" in profiling_results:
                top_artists = profiling_results["top_artists"]
                if top_artists:
                    top_y = stats_y + 5
                    self._animated_type(top_y, 4, "Top Artists:", color_pair=1, highlight=True, delay=0.01)

                    # Display top 5 artists
                    items = list(top_artists.items())[:5]
                    for i, (artist, count) in enumerate(items):
                        artist_text = f"{artist}: {count} scrobbles"
                        # Truncate if too long
                        if len(artist_text) > w - 12:
                            artist_text = artist_text[:w - 15] + "..."
                        self._animated_type(top_y + i + 1, 6, artist_text, color_pair=1, delay=0.01)

            # Show visualization path if available
            if "viz_path" in profiling_results:
                viz_y = stats_y + 12
                self._animated_type(viz_y, 4, "Visualization:", color_pair=1, highlight=True, delay=0.01)
                viz_path = profiling_results["viz_path"]
                self._animated_type(viz_y + 1, 6, f"Top Artists Chart: {viz_path}", color_pair=1, delay=0.01)

                note_msg = "Note: The visualization has been saved and can be viewed with any image viewer."
                self._animated_type(viz_y + 3, 4, note_msg, color_pair=6, delay=0.01)
        else:
            # Show error message
            error_msg = "Error during data analysis:"
            self._animated_type(4, 4, error_msg, color_pair=4, highlight=True, delay=0.01)

            # Show error details
            if "error" in profiling_results:
                error_details = profiling_results["error"]
                # Truncate if too long
                if len(error_details) > w - 8:
                    error_details = error_details[:w - 11] + "..."
                self._animated_type(6, 4, error_details, color_pair=4, delay=0.01)

        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Bottom border
        self.stdscr.addstr(h - 2, 1, separator, curses.color_pair(2))
        self.stdscr.refresh()

        # Wait for keypress
        self.stdscr.getch()

        # Return to main menu
        self._show_main_menu()

    def _run_visualizations(self):
        """Show visualization options."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "VISUALIZATIONS"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Menu options
        menu_title = "VISUALIZATION OPTIONS"
        self._animated_type(4, (w - len(menu_title)) // 2, menu_title, color_pair=1, highlight=True, delay=0.01)

        options = [
            "1. Scrobbles Over Time - View listening patterns by year/month",
            "2. Artist Analysis - Explore top artists and their distribution",
            "3. Trusted Companions - Find artists present across all years",
            "4. Return to Main Menu"
        ]

        option_start_y = 6
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
                self._run_time_visualizations()
                return
            elif c == ord('2'):
                self._run_artist_visualizations()
                return
            elif c == ord('3'):
                self._run_trusted_companions()
                return
            elif c == ord('4') or c in (ord('q'), 27):
                self._show_main_menu()
                return

    def _run_time_visualizations(self):
        """Show time-based visualizations."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "SCROBBLES OVER TIME"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Loading message
        loading_msg = "Loading data and generating visualizations..."
        self._animated_type(4, (w - len(loading_msg)) // 2, loading_msg, color_pair=3, delay=0.01)
        self.stdscr.refresh()

        # Load data and generate visualizations in a separate thread
        visualization_results = {}

        def generate_visualizations():
            try:
                # Load data
                data = load_data()

                # Generate visualizations
                yearly_fig = scrobbles_per_year(data)
                monthly_fig = scrobbles_per_month(data)
                ridgeline_fig = monthly_ridgeline(data)
                seasonal_fig = seasonal_decomposition(data)

                # Save visualizations
                yearly_path = save_figure(yearly_fig, "yearly_scrobbles.png")
                monthly_path = save_figure(monthly_fig, "monthly_scrobbles.png")
                ridgeline_path = save_figure(ridgeline_fig, "monthly_ridgeline.png")
                seasonal_path = save_figure(seasonal_fig, "seasonal_decomposition.png")

                # Store results
                visualization_results["yearly"] = yearly_path
                visualization_results["monthly"] = monthly_path
                visualization_results["ridgeline"] = ridgeline_path
                visualization_results["seasonal"] = seasonal_path
                visualization_results["success"] = True

            except Exception as e:
                visualization_results["error"] = str(e)
                visualization_results["success"] = False

        # Run in a thread to keep UI responsive
        thread = threading.Thread(target=generate_visualizations)
        thread.daemon = True
        thread.start()

        # Show a spinner while waiting
        spinner_chars = ['|', '/', '-', '\\']
        spinner_idx = 0

        while thread.is_alive():
            # Update spinner
            spinner_text = f"Generating visualizations {spinner_chars[spinner_idx]}"
            self.stdscr.addstr(6, (w - len(spinner_text)) // 2, spinner_text, curses.color_pair(3))
            self.stdscr.refresh()

            # Increment spinner index
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)

            # Sleep briefly
            time.sleep(0.1)

            # Check for ESC key to cancel
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                self.stdscr.addstr(8, 4, "Cancellation requested. Please wait for current operations to complete...",
                                  curses.color_pair(4))
                self.stdscr.refresh()

        # Clear the screen
        self.stdscr.clear()

        # Redraw title and separator
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Check if visualizations were generated successfully
        if visualization_results.get("success", False):
            # Show success message
            success_msg = "Visualizations generated successfully!"
            self._animated_type(4, (w - len(success_msg)) // 2, success_msg, color_pair=3, highlight=True, delay=0.01)

            # Show visualization paths
            self._animated_type(6, 4, "Visualization files:", color_pair=1, delay=0.01)

            y_pos = 8
            for viz_type, path in [
                ("Yearly Scrobbles", visualization_results.get("yearly", "")),
                ("Monthly Scrobbles", visualization_results.get("monthly", "")),
                ("Monthly Ridgeline", visualization_results.get("ridgeline", "")),
                ("Seasonal Decomposition", visualization_results.get("seasonal", ""))
            ]:
                if path:
                    self._animated_type(y_pos, 6, f"{viz_type}: {path}", color_pair=1, delay=0.01)
                    y_pos += 1

            # Show note about viewing the visualizations
            note_msg = "Note: The visualizations have been saved to the 'visualizations' directory."
            self._animated_type(y_pos + 1, 4, note_msg, color_pair=6, delay=0.01)

            view_msg = "You can view them using any image viewer application."
            self._animated_type(y_pos + 2, 4, view_msg, color_pair=6, delay=0.01)
        else:
            # Show error message
            error_msg = "Error generating visualizations:"
            self._animated_type(4, 4, error_msg, color_pair=4, highlight=True, delay=0.01)

            # Show error details
            if "error" in visualization_results:
                error_details = visualization_results["error"]
                # Truncate if too long
                if len(error_details) > w - 8:
                    error_details = error_details[:w - 11] + "..."
                self._animated_type(6, 4, error_details, color_pair=4, delay=0.01)

        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Bottom border
        self.stdscr.addstr(h - 2, 1, separator, curses.color_pair(2))
        self.stdscr.refresh()

        # Wait for keypress
        self.stdscr.getch()

        # Return to statistics menu
        self._run_statistics()

    def _run_artist_visualizations(self):
        """Show artist-based visualizations."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "ARTIST ANALYSIS"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Loading message
        loading_msg = "Loading data and generating visualizations..."
        self._animated_type(4, (w - len(loading_msg)) // 2, loading_msg, color_pair=3, delay=0.01)
        self.stdscr.refresh()

        # Load data and generate visualizations in a separate thread
        visualization_results = {}

        def generate_visualizations():
            try:
                # Load data
                data = load_data()

                # Generate visualizations
                top_artists_fig = top_artists_bar(data, n=15)
                distribution_fig = artist_distribution_violin(data)

                # Save visualizations
                top_artists_path = save_figure(top_artists_fig, "top_artists.png")
                distribution_path = save_figure(distribution_fig, "artist_distribution.png")

                # Store results
                visualization_results["top_artists"] = top_artists_path
                visualization_results["distribution"] = distribution_path
                visualization_results["success"] = True

                # Get some statistics for display
                artist_counts = data["Artist"].value_counts()
                visualization_results["stats"] = {
                    "total_artists": len(artist_counts),
                    "total_scrobbles": len(data),
                    "top_artist": artist_counts.index[0],
                    "top_artist_count": artist_counts.iloc[0],
                    "mean_scrobbles": artist_counts.mean(),
                    "median_scrobbles": artist_counts.median()
                }

            except Exception as e:
                visualization_results["error"] = str(e)
                visualization_results["success"] = False

        # Run in a thread to keep UI responsive
        thread = threading.Thread(target=generate_visualizations)
        thread.daemon = True
        thread.start()

        # Show a spinner while waiting
        spinner_chars = ['|', '/', '-', '\\']
        spinner_idx = 0

        while thread.is_alive():
            # Update spinner
            spinner_text = f"Generating visualizations {spinner_chars[spinner_idx]}"
            self.stdscr.addstr(6, (w - len(spinner_text)) // 2, spinner_text, curses.color_pair(3))
            self.stdscr.refresh()

            # Increment spinner index
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)

            # Sleep briefly
            time.sleep(0.1)

            # Check for ESC key to cancel
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                self.stdscr.addstr(8, 4, "Cancellation requested. Please wait for current operations to complete...",
                                  curses.color_pair(4))
                self.stdscr.refresh()

        # Clear the screen
        self.stdscr.clear()

        # Redraw title and separator
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Check if visualizations were generated successfully
        if visualization_results.get("success", False):
            # Show success message
            success_msg = "Visualizations generated successfully!"
            self._animated_type(4, (w - len(success_msg)) // 2, success_msg, color_pair=3, highlight=True, delay=0.01)

            # Show statistics
            if "stats" in visualization_results:
                stats = visualization_results["stats"]
                self._animated_type(6, 4, "Artist Statistics:", color_pair=1, highlight=True, delay=0.01)

                stats_text = [
                    f"Total Artists: {stats['total_artists']}",
                    f"Total Scrobbles: {stats['total_scrobbles']}",
                    f"Top Artist: {stats['top_artist']} ({stats['top_artist_count']} scrobbles)",
                    f"Average Scrobbles per Artist: {stats['mean_scrobbles']:.2f}",
                    f"Median Scrobbles per Artist: {stats['median_scrobbles']:.2f}"
                ]

                for i, text in enumerate(stats_text):
                    self._animated_type(8 + i, 6, text, color_pair=1, delay=0.01)

            # Show visualization paths
            self._animated_type(15, 4, "Visualization files:", color_pair=1, delay=0.01)

            y_pos = 17
            for viz_type, path in [
                ("Top Artists", visualization_results.get("top_artists", "")),
                ("Artist Distribution", visualization_results.get("distribution", ""))
            ]:
                if path:
                    self._animated_type(y_pos, 6, f"{viz_type}: {path}", color_pair=1, delay=0.01)
                    y_pos += 1

            # Show note about viewing the visualizations
            note_msg = "Note: The visualizations have been saved to the 'visualizations' directory."
            self._animated_type(y_pos + 1, 4, note_msg, color_pair=6, delay=0.01)

            view_msg = "You can view them using any image viewer application."
            self._animated_type(y_pos + 2, 4, view_msg, color_pair=6, delay=0.01)
        else:
            # Show error message
            error_msg = "Error generating visualizations:"
            self._animated_type(4, 4, error_msg, color_pair=4, highlight=True, delay=0.01)

            # Show error details
            if "error" in visualization_results:
                error_details = visualization_results["error"]
                # Truncate if too long
                if len(error_details) > w - 8:
                    error_details = error_details[:w - 11] + "..."
                self._animated_type(6, 4, error_details, color_pair=4, delay=0.01)

        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Bottom border
        self.stdscr.addstr(h - 2, 1, separator, curses.color_pair(2))
        self.stdscr.refresh()

        # Wait for keypress
        self.stdscr.getch()

        # Return to statistics menu
        self._run_statistics()

    def _run_trusted_companions(self):
        """Show artists that appear in every year with low variance in scrobble count."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "TRUSTED COMPANIONS"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Loading message
        loading_msg = "Loading data and identifying trusted companions..."
        self._animated_type(4, (w - len(loading_msg)) // 2, loading_msg, color_pair=3, delay=0.01)
        self.stdscr.refresh()

        # Load data and generate visualizations in a separate thread
        visualization_results = {}

        def generate_visualizations():
            try:
                # Load data
                data = load_data()

                # Find trusted companions
                result_df, fig = find_trusted_companions(data, percentile=25)

                # Save visualization
                if not result_df.empty:
                    viz_path = save_figure(fig, "trusted_companions.png")
                    visualization_results["viz_path"] = viz_path
                    visualization_results["result_df"] = result_df
                    visualization_results["success"] = True
                else:
                    visualization_results["success"] = False
                    visualization_results["error"] = "No artists found that appear in all years"

            except Exception as e:
                visualization_results["error"] = str(e)
                visualization_results["success"] = False

        # Run in a thread to keep UI responsive
        thread = threading.Thread(target=generate_visualizations)
        thread.daemon = True
        thread.start()

        # Show a spinner while waiting
        spinner_chars = ['|', '/', '-', '\\']
        spinner_idx = 0

        while thread.is_alive():
            # Update spinner
            spinner_text = f"Analyzing data {spinner_chars[spinner_idx]}"
            self.stdscr.addstr(6, (w - len(spinner_text)) // 2, spinner_text, curses.color_pair(3))
            self.stdscr.refresh()

            # Increment spinner index
            spinner_idx = (spinner_idx + 1) % len(spinner_chars)

            # Sleep briefly
            time.sleep(0.1)

            # Check for ESC key to cancel
            c = self.stdscr.getch()
            if c == 27:  # ESC key
                self.stdscr.addstr(8, 4, "Cancellation requested. Please wait for current operations to complete...",
                                  curses.color_pair(4))
                self.stdscr.refresh()

        # Clear the screen
        self.stdscr.clear()

        # Redraw title and separator
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Check if visualizations were generated successfully
        if visualization_results.get("success", False):
            # Show success message
            success_msg = "Trusted companions identified successfully!"
            self._animated_type(4, (w - len(success_msg)) // 2, success_msg, color_pair=3, highlight=True, delay=0.01)

            # Show explanation
            explanation = "Trusted companions are artists that appear in every year of your listening history"
            self._animated_type(6, 4, explanation, color_pair=1, delay=0.01)
            explanation2 = "with relatively consistent scrobble counts (low coefficient of variation)."
            self._animated_type(7, 4, explanation2, color_pair=1, delay=0.01)

            # Show top trusted companions
            result_df = visualization_results.get("result_df")
            if result_df is not None and not result_df.empty:
                self._animated_type(9, 4, "Top Trusted Companions:", color_pair=1, highlight=True, delay=0.01)

                # Display top 10 companions
                top_n = min(10, len(result_df))
                for i in range(top_n):
                    row = result_df.iloc[i]
                    text = f"{i+1}. {row['Artist']} - Avg: {row['mean']:.1f} scrobbles/year, CV: {row['cv']:.3f}"
                    self._animated_type(11 + i, 6, text, color_pair=1, delay=0.01)

            # Show visualization path
            viz_path = visualization_results.get("viz_path", "")
            if viz_path:
                self._animated_type(22, 4, "Visualization file:", color_pair=1, delay=0.01)
                self._animated_type(23, 6, f"Trusted Companions: {viz_path}", color_pair=1, delay=0.01)

                # Show note about viewing the visualization
                note_msg = "Note: The visualization has been saved to the 'visualizations' directory."
                self._animated_type(25, 4, note_msg, color_pair=6, delay=0.01)

                view_msg = "You can view it using any image viewer application."
                self._animated_type(26, 4, view_msg, color_pair=6, delay=0.01)
        else:
            # Show error message
            error_msg = "Error identifying trusted companions:"
            self._animated_type(4, 4, error_msg, color_pair=4, highlight=True, delay=0.01)

            # Show error details
            if "error" in visualization_results:
                error_details = visualization_results["error"]
                # Truncate if too long
                if len(error_details) > w - 8:
                    error_details = error_details[:w - 11] + "..."
                self._animated_type(6, 4, error_details, color_pair=4, delay=0.01)

        # Instructions
        instructions = "Press any key to return to the menu..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Bottom border
        self.stdscr.addstr(h - 2, 1, separator, curses.color_pair(2))
        self.stdscr.refresh()

        # Wait for keypress
        self.stdscr.getch()

        # Return to statistics menu
        self._run_statistics()

    def _run_model_server(self):
        """Start or stop the XGBoost model prediction server."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        # Draw title
        title = "XGBOOST MODEL SERVER"
        self._animated_type(1, (w - len(title)) // 2, title, color_pair=2, highlight=True, delay=0.01)

        # Draw separator
        separator = "═" * (w - 2)
        self.stdscr.addstr(2, 1, separator, curses.color_pair(2))

        # Get server status
        status = get_server_status()
        is_running = status["is_running"]
        port = status["port"]

        # Display current status
        status_text = f"Server status: {'RUNNING' if is_running else 'STOPPED'}"
        if is_running:
            status_text += f" (Port: {port})"

        self._animated_type(4, 4, status_text, 
                           color_pair=3 if is_running else 4, 
                           highlight=True, delay=0.01)

        # Display options
        options_text = "Choose an option:"
        self._animated_type(6, 4, options_text, color_pair=1, delay=0.01)

        options = [
            f"1. {'Stop' if is_running else 'Start'} server",
            "2. Return to Data Gathering menu"
        ]

        for i, option in enumerate(options):
            self._animated_type(8 + i, 6, option, color_pair=1, delay=0.01)

        # If server is running, show endpoints
        if is_running:
            endpoints_title = "Available endpoints:"
            self._animated_type(11, 4, endpoints_title, color_pair=6, delay=0.01)

            endpoints = [
                f"• Health check: http://localhost:{port}/health",
                f"• Predict: http://localhost:{port}/predict",
                f"• Batch predict: http://localhost:{port}/predict_batch"
            ]

            for i, endpoint in enumerate(endpoints):
                self._animated_type(13 + i, 6, endpoint, color_pair=1, delay=0.01)

            # Show example usage
            usage_title = "Example usage (with curl):"
            self._animated_type(17, 4, usage_title, color_pair=6, delay=0.01)

            usage = [
                f'curl -X POST http://localhost:{port}/predict -H "Content-Type: application/json" -d "{{\'data\': {{\'feature1\': 0.5, \'feature2\': 0.7}}}}"'
            ]

            for i, example in enumerate(usage):
                self._animated_type(19 + i, 6, example, color_pair=1, delay=0.01)

        # Instructions
        instructions = "Press a number key to select an option..."
        self._animated_type(h - 3, (w - len(instructions)) // 2, instructions, color_pair=6, delay=0.01)

        # Bottom border
        self.stdscr.addstr(h - 2, 1, separator, curses.color_pair(2))
        self.stdscr.refresh()

        # Handle menu selection
        while True:
            c = self.stdscr.getch()
            if c == ord('1'):
                # Start or stop server
                if is_running:
                    # Stop server
                    self.stdscr.addstr(h - 5, 4, "Stopping server...", curses.color_pair(4))
                    self.stdscr.refresh()

                    success = stop_server()

                    if success:
                        self.stdscr.addstr(h - 5, 4, "Server stopped successfully!", curses.color_pair(3))
                    else:
                        self.stdscr.addstr(h - 5, 4, "Failed to stop server.", curses.color_pair(4))
                else:
                    # Start server
                    self.stdscr.addstr(h - 5, 4, "Starting server...", curses.color_pair(3))
                    self.stdscr.refresh()

                    success = start_server()

                    if success:
                        self.stdscr.addstr(h - 5, 4, "Server started successfully!", curses.color_pair(3))
                    else:
                        self.stdscr.addstr(h - 5, 4, "Failed to start server.", curses.color_pair(4))

                self.stdscr.refresh()
                time.sleep(1.5)
                self._run_model_server()  # Refresh the screen
                return
            elif c == ord('2') or c in (ord('q'), 27):
                self._show_data_gathering_menu()
                return

    def _run_full_refresh(self):
        """Flush *scrobble* and pull a complete history from Last.fm."""
        # Ask for confirmation – destructive!
        confirm_text = "This will DELETE every scrobble in the database. Proceed? (y/N) "
        self.stdscr.clear()
        self._animated_type(2, 2, confirm_text, color_pair=4, highlight=True)
        self.stdscr.refresh()
        curses.echo();
        curses.curs_set(1)
        ch = self.stdscr.getstr(2, 2 + len(confirm_text)).decode().lower()
        curses.noecho();
        curses.curs_set(0)
        if ch != 'y':
            self._show_main_menu();
            return
        # Re-use the nice progress screen
        progress = self._show_progress_screen("FULL REFRESH")

        def task():
            # Call the full refresh function from data_gathering module
            run_full_refresh(self.username, progress)

        t = threading.Thread(target=task, daemon=True)
        t.start()
        while t.is_alive():
            if self.stdscr.getch() == 27:  # to allow ESC cancel display
                pass
        self.stdscr.getch()
        self._show_data_gathering_menu()

    def _run_full_pipeline(self):
        """Run the full pipeline: fetch, enrich, clean, and profile data."""
        # Show progress screen
        progress_callback = self._show_progress_screen("FULL PIPELINE")

        # Run in a thread to keep UI responsive
        def run_task():
            try:
                # Use the new pipeline module
                from corefunc.pipeline import run_full_pipeline

                result = run_full_pipeline(self.username, progress_callback)

                # Show completion message
                haa, duplavee = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, " " * (duplavee - 4))  # Clear any previous message

                if result['status'] == 'success':
                    fetch_result = result.get('fetch_result', {})
                    enrich_result = result.get('enrich_result', {})
                    clean_result = result.get('clean_result', {})

                    new_scrobbles = fetch_result.get('new_scrobbles', 0)
                    processed = enrich_result.get('processed', 0)
                    created = enrich_result.get('created', 0)
                    updated = enrich_result.get('updated', 0)
                    cleaned = clean_result.get('cleaned', 0)

                    success_msg = f"Pipeline completed successfully!"
                    self.stdscr.addstr(12, 2, success_msg, curses.color_pair(3) | curses.A_BOLD)

                    if new_scrobbles > 0:
                        details_msg = f"Fetched {new_scrobbles} new scrobbles."
                        self.stdscr.addstr(13, 2, details_msg, curses.color_pair(1))

                    artist_msg = f"Artists: processed {processed}, created {created}, updated {updated}, cleaned {cleaned}"
                    self.stdscr.addstr(14, 2, artist_msg, curses.color_pair(1))
                else:
                    error_msg = f"Error: {result['message']}"
                    # Only truncate if necessary (not enough space on screen)
                    if len(error_msg) > duplavee - 4:
                        # Check if it's an SQL log or similar that we want to show in full if possible
                        if "[SQL:" in result['message'] or result['message'].startswith("SQL:"):
                            # Display as much as possible without scrolling
                            error_msg = error_msg[:duplavee - 4]
                        else:
                            # For other messages, truncate with ellipsis
                            error_msg = error_msg[:duplavee - 7] + "..."
                    self.stdscr.addstr(12, 2, error_msg, curses.color_pair(4) | curses.A_BOLD)

                self.stdscr.addstr(16, 2, "Press any key to return to the menu...", curses.color_pair(1))
                self.stdscr.refresh()
            except Exception as e:
                # Show error message
                haa, duplavee = self.stdscr.getmaxyx()
                self.stdscr.addstr(12, 2, " " * (duplavee - 4))  # Clear any previous message
                error_msg = f"Error: {str(e)}"
                # Only truncate if necessary (not enough space on screen)
                if len(error_msg) > duplavee - 4:
                    # Check if it's an SQL log or similar that we want to show in full if possible
                    if "[SQL:" in str(e) or str(e).startswith("SQL:"):
                        # Display as much as possible without scrolling
                        error_msg = error_msg[:duplavee - 4]
                    else:
                        # For other messages, truncate with ellipsis
                        error_msg = error_msg[:duplavee - 7] + "..."
                self.stdscr.addstr(12, 2, error_msg, curses.color_pair(4) | curses.A_BOLD)
                self.stdscr.addstr(14, 2, "Press any key to return to the menu...", curses.color_pair(1))
                self.stdscr.refresh()
                logging.exception("Error during pipeline execution")

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
        # Return to data gathering menu
        self._show_data_gathering_menu()
