from __future__ import annotations
"""
Main entry-point for CanonFodder.
The script walks the operator through a *linear* workflow:
    1) decide which Last.fm user we are working with
    2) obtain a CSV (fresh download or reuse newest local one)
    3) verify CSV quirks (commas stripped)
    4) fetch recent scrobbles via the Last.fm API and load into the DB
    5) guide operator to further notebook-style workflows in dev_profile.py and dev_canon.py
"""
import sys
from pathlib import Path
import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)
logging.getLogger("musicbrainzngs").setLevel(logging.WARNING)
# All DB plumbing resolved by DB.__init__
from DB import engine, SessionLocal  # noqa: I202
from DB.ops import ascii_freq, bulk_insert_scrobbles, latest_scrobble_table_to_df, seed_ascii_chars
from DB.models import Base
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
from corefunc import dataprofiler as dp, canonizer as cz
from CSV.autofetch import fetch_scrobbles_csv
from enrich import enrich_artist_country
from helpers.io import (CSV_DIR,
                        latest_csv,
                        latest_parquet,
                        dump_parquet,
                        register_custom_palette)
from helpers.cli import (choose_lastfm_user,
                         verify_commas,
                         yes_no)
import lfAPI
import mbAPI
mbAPI.init()
# Display options and directory setting
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{: .2f}".format
HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "JSON"
PALETTES_FILE = JSON_DIR / "palettes.json"
os.environ.pop("FLASK_APP", None)
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors))
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# ---------------------------------------------------------------------------
# main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    user = choose_lastfm_user()
    # Building up all data tables for env-inferred DB connection
    Base.metadata.create_all(engine)
    # CSV stage
    local_csv = latest_csv(user)
    print("\n===========================================")
    print("Welcome to CanonFodder!")
    print("===========================================\n")
    print("CSV stage")
    if local_csv:
        print(f"Newest local file: {local_csv.name}")
        dl_needed = yes_no("\nDownload a fresh CSV (can take 30 min)?"
                           "\n[y]es or [n]o – default is: ",
                           default="n")
    else:
        dl_needed = True
    csv_path: Path
    if dl_needed:
        print("\nFetching a fresh CSV – this may take ±30 minutes…")
        csv_path = fetch_scrobbles_csv(user, out_dir=CSV_DIR, once_per="week")
        if csv_path is None:
            # ► the fetcher said “already ran this week” or the
            #   remote site was down – fall back to the latest local file
            print("\nDownload skipped – using newest local CSV.")
            csv_path = local_csv
            if csv_path is None:  # nothing cached at all → real error
                sys.exit("\nNo CSV available, aborting.")
    else:
        csv_path = local_csv
    print("===========================================")
    print("Verifying CSV")
    print("===========================================")
    verify_commas(csv_path)
    print("===========================================")
    print(f"↑ Note how commas inside values are gone in ({csv_path.name})?")
    print("\nThe third-party solution failed to enclose field values within quotation marks.")
    print("\nLet us use the last.fm API instead.")
    # small profile snapshot directly via API (cheap call)
    print("\nLast.fm profile snapshot")
    lfAPI.fetch_misc_data_from_lastfmapi(user)
    # Decide: full refresh vs. reuse parquet
    print("===========================================")
    print("\nLast.fm API fetch")
    use_api = yes_no("\nRefresh scrobbles from last.fm API knowing that it takes some time? "
                     "\n[y]es or [n]o – default is: ", default="n")
    print("===========================================")
    if use_api:
        print("\nFetching scrobbles from last.fm API. Please wait…\n")
        df_recent = lfAPI.fetch_recent_tracks_all_pages(user)
        print(f"\nFetched {len(df_recent)} scrobbles\n.")
        if df_recent.empty:
            print("\nNothing to store, aborting workflow.\n")
            sys.exit(1)
        bulk_insert_scrobbles(df_recent, engine)
        enrich_artist_country()
        print("\nWe can check how many times each special ASCII character is found in artist names.")
        seed_ascii_chars(engine)
        print(ascii_freq(engine))
        # Dumping table to parquet for the next fast run
        dump_parquet(df_recent)
    else:  # fast lane
        df_recent, _ = latest_parquet(return_df=True)
    # From here on, profiling and canonisation steps
    print("===========================================")
    print("Your data is accessible for the project now. Please head on to dev_profile.py and dev_canon.py for further.")
    print("===========================================")
    '''
    TODO FOR FUTURE DEVELOPMENT:
      •  creating menu structure for CanonFodder
      •  restructure above workflow as menu option 1: data fetch
      •  restructure dev_profile.py to corefunc/dataprofiler.py, dev_canon.py to corefunc/canonizer.py
      •  write menu option 2: dev_profile.py with MBID connector and user_country logic, menu option 3: dev_canon.py
          # profile = dp.run_profiling(df_canon),
          including # mbAPI.fetch_country(mbid) if available else mbAPI.search(...)
          # df_canon = cz.apply_previous(df_recent)
      •  add menu option 4: country stats display
    '''
    print("\nWorkflow finished OK.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    finally:
        # Disposing SQLAlchemy connection pool cleanly
        engine.dispose()
