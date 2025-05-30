"""
CLI-based entry-point for CanonFodder
A hybrid CLI/graphical approach with animated typing and steampunk-inspired aesthetics.
The application maintains all functionality of the previous GUI while presenting
a more console-like experience.
"""
from __future__ import annotations
import os
import sys
import logging
import argparse
import platform
import signal
import re
import threading
from pathlib import Path
from typing import Optional
from sqlalchemy import delete, select, text, func

# Import from helpers
from helpers.progress import ProgressCallback, null_progress_callback
from helpers.cli_interface import CliInterface, Colors, init_colors, start_gui, get_db_statistics
from helpers.formatting import format_sql_for_display
from helpers.cli import choose_lastfm_user

# Import from corefunc
from corefunc.workflow import run_data_gathering_workflow

# Import from HTTP
from HTTP.lfAPI import fetch_lastfm_with_progress

# Import from DB
from DB import SessionLocal
from DB.models import ArtistInfo, Scrobble, Base
from DB.ops import populate_artist_info_from_scrobbles

# Import API modules
from HTTP import lfAPI
from HTTP import mbAPI


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
        "--refresh-aliases",
        action="store_true",
        help="Refresh artist aliases from MusicBrainz API.",
    )
    parser.add_argument(
        "--debug-artist-aliases",
        type=str,
        help="Debug alias retrieval for a specific artist (provide artist name or MBID)",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Last.fm username to use with --enrich-artist-mbid or --legacy-cli",
    )
    args = parser.parse_args()

    # Check for special operations
    if args.debug_artist_aliases:
        print(f"Debugging aliases for: {args.debug_artist_aliases}")
        from pprint import pprint

        is_mbid = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", args.debug_artist_aliases))

        if is_mbid:
            print(f"Looking up artist by MBID: {args.debug_artist_aliases}")
            try:
                aliases = mbAPI.get_aliases(args.debug_artist_aliases)
                artist_data = mbAPI.lookup_artist(args.debug_artist_aliases, with_aliases=True)
                print(f"Artist: {artist_data.get('name')}")
                print(f"Found {len(aliases)} aliases:")
                for alias in aliases:
                    print(f"  - {alias}")

                # Check DB record
                with SessionLocal() as session:
                    db_artist = session.execute(
                        select(ArtistInfo).where(ArtistInfo.mbid == args.debug_artist_aliases)
                    ).scalar_one_or_none()

                    if db_artist:
                        print("\nDatabase record:")
                        print(f"  Name: {db_artist.artist_name}")
                        print(f"  MBID: {db_artist.mbid}")
                        print(f"  Aliases: {db_artist.aliases or 'None'}")

                        # Count actual aliases
                        if db_artist.aliases:
                            db_aliases = db_artist.aliases.split(',')
                            print(f"  Alias count: {len(db_aliases)}")
                    else:
                        print("\nNo database record found!")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Looking up artist by name: {args.debug_artist_aliases}")
            try:
                # First search for the artist
                search_results = mbAPI.search_artist(args.debug_artist_aliases, limit=5)

                if not search_results:
                    print("No artists found with that name")
                    return 1

                print(f"Found {len(search_results)} matching artists:")
                for i, artist in enumerate(search_results):
                    print(f"{i+1}. {artist['name']} ({artist['id']})")
                    if 'disambiguation' in artist and artist['disambiguation']:
                        print(f"   {artist['disambiguation']}")

                # Ask which one to check
                if len(search_results) > 1:
                    choice = input("\nEnter number to check aliases (or press Enter for #1): ")
                    idx = 0
                    if choice.strip():
                        try:
                            idx = int(choice) - 1
                            if idx < 0 or idx >= len(search_results):
                                print("Invalid choice, using #1")
                                idx = 0
                        except ValueError:
                            print("Invalid choice, using #1")
                            idx = 0
                else:
                    idx = 0

                # Get aliases for selected artist
                selected_mbid = search_results[idx]['id']
                aliases = mbAPI.get_aliases(selected_mbid)
                print(f"\nFound {len(aliases)} aliases for {search_results[idx]['name']}:")
                for alias in aliases:
                    print(f"  - {alias}")

                # Check DB record
                with SessionLocal() as session:
                    db_artist = session.execute(
                        select(ArtistInfo).where(ArtistInfo.mbid == selected_mbid)
                    ).scalar_one_or_none()

                    if db_artist:
                        print("\nDatabase record:")
                        print(f"  Name: {db_artist.artist_name}")
                        print(f"  MBID: {db_artist.mbid}")
                        print(f"  Aliases: {db_artist.aliases or 'None'}")

                        # Count actual aliases
                        if db_artist.aliases:
                            db_aliases = db_artist.aliases.split(',')
                            print(f"  Alias count: {len(db_aliases)}")
                    else:
                        print("\nNo database record found!")
            except Exception as e:
                print(f"Error: {e}")
        return 0

    if args.refresh_aliases:
        print("Refreshing artist aliases from MusicBrainz...")
        from tqdm import tqdm

        with SessionLocal() as session:
            # First, count total artists with MBIDs
            total_artists = session.execute(
                select(func.count()).where(ArtistInfo.mbid.is_not(None))
            ).scalar_one()

            # Counting artists with empty aliases
            empty_aliases = session.execute(
                select(func.count()).where(
                    ArtistInfo.mbid.is_not(None),
                    (ArtistInfo.aliases.is_(None) | (ArtistInfo.aliases == ""))
                )
            ).scalar_one()

            print(f"Database has {total_artists} artists with MBIDs")
            print(f"Found {empty_aliases} artists with empty aliases")

            # Asking for confirmation if all artists should be refreshed
            refresh_all = False
            if empty_aliases < total_artists:
                choice = input("Do you want to refresh ALL artists (y) or only those with empty aliases (n)? [y/N] ")
                refresh_all = choice.lower() == 'y'

            # Getting artists to process based on user choice
            if refresh_all:
                print(f"Refreshing ALL {total_artists} artists...")
                query = select(ArtistInfo).where(ArtistInfo.mbid.is_not(None))
                artists = session.execute(query).scalars().all()
            else:
                print(f"Refreshing {empty_aliases} artists with empty aliases...")
                query = select(ArtistInfo).where(
                    ArtistInfo.mbid.is_not(None),
                    (ArtistInfo.aliases.is_(None) | (ArtistInfo.aliases == ""))
                )
                artists = session.execute(query).scalars().all()

            # Processing each artist with a progress bar
            success_count = 0
            error_count = 0

            for artist in tqdm(artists, desc="Fetching aliases"):
                try:
                    # Getting complete artist info including aliases
                    artist_data = mbAPI.lookup_artist(artist.mbid, with_aliases=True)

                    # Updating the database record with aliases
                    if artist_data and "aliases" in artist_data and artist_data["aliases"]:
                        aliases_str = ",".join(artist_data["aliases"])
                        artist.aliases = aliases_str
                        session.commit()
                        success_count += 1
                    else:
                        # If no aliases found but lookup was successful, marking as processed with empty aliases
                        session.commit()
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {artist.artist_name}: {e}")

            # Displaying summary
            print("\nAliases refresh completed:")
            print(f"  ✅ Successfully updated: {success_count} artists")
            if error_count > 0:
                print(f"  ❌ Errors: {error_count} artists")
            else:
                print("  No errors encountered!")

        # Final check - counting non-empty aliases
        with SessionLocal() as session:
            non_empty_count = session.execute(
                select(func.count()).where(
                    ArtistInfo.mbid.is_not(None),
                    ArtistInfo.aliases.is_not(None),
                    ArtistInfo.aliases != ""
                )
            ).scalar_one()

            print(f"Database now has {non_empty_count} artists with aliases")

        return 0

    if args.enrich_artist_mbid:
        # Getting username from args, env, or config
        username = args.username or os.getenv("LASTFM_USER")
        if not username:
            # Trying to read from config file
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

        # Setup console progress display
        from tqdm import tqdm
        pbar = tqdm(total=100, desc="Fetching MBIDs")

        class TqdmProgressCallback:
            def __call__(self, task: str, percentage: float, message: Optional[str] = None) -> None:
                pbar.n = percentage
                if message:
                    pbar.set_description(f"{task}: {message}")
                else:
                    pbar.set_description(task)
                pbar.refresh()

        result = lfAPI.enrich_artist_mbids(username, TqdmProgressCallback())
        pbar.close()

        if result["status"] == "success":
            print(f"✅ Success: {result['message']}")
        else:
            print(f"❌ Error: {result['message']}")

        return 0

    # For backward compatibility, treating --cli the same as --legacy-cli
    if args.legacy_cli or args.cli:
        # In legacy CLI mode, let the function handle username input directly
        # If username was provided, pass it to the function
        return run_data_gathering_workflow(args.username) or 0
    else:
        # Initializing terminal colors
        init_colors()

        # Setting global animation flag
        if args.no_animation:
            import helpers.cli_interface as cli
            cli.DISABLE_ANIMATIONS = True

        # Starting the CLI interface
        start_gui()
        return 0


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    sys.exit(_cli_entry())
