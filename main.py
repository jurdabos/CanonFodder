from __future__ import annotations

"""
Main entry-point for CanonFodder.
The script walks the operator through a *linear* workflow:
    1) decide which Last.fm user we are working with
    2) obtain a CSV (fresh download or reuse newest local one)
    3) verify CSV quirks (commas stripped etc.) – show a preview
    4) fetch recent scrobbles via the Last.fm API and load into the DB
    5) (placeholder) canonisation / MusicBrainz enrichment / EDA steps (to be filled out later when developed)
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# All DB plumbing is resolved by DB.__init__
from DB import engine, SessionLocal                     # noqa: I202
from DB.ops import bulk_insert_scrobbles, latest_scrobble_df
from CSV.autofetch import fetch_scrobbles_csv
import helper
import lfAPI
import mbAPI

# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

HERE = Path(__file__).resolve().parent
CSV_DIR = HERE / "CSV"
PQ_DIR = HERE / "PQ"
PQ_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def newest_csv_for(user: str) -> Optional[Path]:
    """Return the most recent CSV for *user* inside CSV_DIR, else None."""
    pattern = f"{user}_*.csv"
    found   = sorted(CSV_DIR.glob(pattern), reverse=True)
    return found[0] if found else None


def newest_parquet() -> Optional[Path]:
    """Return the newest parquet snapshot inside PQ_DIR (or *None*)."""
    files = sorted(PQ_DIR.glob("scrobbles_*.parquet"), reverse=True)
    return files[0] if files else None


def dump_latest_table_to_parquet() -> None:
    """Find the most recent scrobbles table,
    materialise it into PQ_DIR / scrobbles_YYYYMMDD_HHMMSS.parquet
    and announce the path."""
    df_db, latest_tbl = latest_scrobble_df(engine)
    if df_db is None:
        print("No scrobble table in DB – nothing to dump.")
        return
    pq_file = PQ_DIR / f"{latest_tbl}.parquet"
    df_db.to_parquet(pq_file, index=False)
    print(f"Latest scrobble table persisted → {pq_file}")


# ---------------------------------------------------------------------------
# main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    user = helper.choose_lastfm_user()
    # CSV stage
    local_csv = newest_csv_for(user)
    print("\n─ CSV stage ─")
    if local_csv:
        print(f"Newest local file: {local_csv.name}")
        dl_needed = helper.yes_no("Download a fresh CSV (can take 30 min)?"
                                  "[y]es or [n]o – default is: ",
                                  default="n")
    else:
        dl_needed = True
    csv_path: Path
    if dl_needed:
        print("\nFetching a fresh CSV – this may take ±30 minutes …")
        csv_path = fetch_scrobbles_csv(user, out_dir=CSV_DIR, once_per="week")
        if csv_path is None:
            print("Download skipped / failed, aborting.")
            sys.exit(1)
    else:
        csv_path = local_csv
    print("\nVerifying CSV (commas stripped etc.) …")
    helper.verify_volcano_name(csv_path)
    df_csv = pd.read_csv(csv_path)
    print(f"↑ Note how commas inside values are gone? ({csv_path.name})")
    # small profile snapshot directly via API (cheap call)
    print("\n─ Profile snapshot ─")
    lfAPI.fetch_misc_data_from_lastfmapi(user)
    # Decide: full refresh vs. reuse parquet
    print("\n─ Last.fm API / parquet stage ─")
    use_api = helper.yes_no("Refresh scrobbles from last.fm API now knowing that it takes some time? "
                            "[y]es or [n]o – default is: ", default="n")
    if use_api:
        df_recent = lfAPI.fetch_recent_tracks_all_pages(user)
        print(f"Fetched {len(df_recent)} scrobbles.")
        if df_recent.empty:
            print("→ nothing to store, aborting workflow.")
            sys.exit(1)
        # store → DB
        new_tbl = bulk_insert_scrobbles(df_recent, engine)
        print(f"Data written to {new_tbl}")
        # Dumping table to parquet for the next fast run
        dump_latest_table_to_parquet()
    else:  # fast path
        pq_file = newest_parquet()
        if pq_file is None:
            print("No parquet snapshot found – have to hit the API once.")
            # simple recursion: rerun main() but force API path
            print("Restarting with fresh-API-fetch …")
            os.execv(sys.executable, [sys.executable, *sys.argv])

        print(f"Loading cached snapshot → {pq_file.name}")
        df_recent = pd.read_parquet(pq_file)
    # From here on, *df_recent* holds the working data – we should hand it to canonisation steps
    print("\nDataFrame ready for canonisation / enrichment:")
    print(df_recent.head())
    # 5) Canonization experiment

    # 6) MusicBrainz enrichment placeholder ----------------------------------
    '''
    TODO:
      •  run canonisation clustering on Artist names
      •  mbAPI.fetch_country / lookup by MBID
      •  decide epsilon by “first Bohren breakpoint”
      •  interactive confirmation etc.
    '''
    # Example stub:
    # mbAPI.fetch_country(mbid) if available else mbAPI.search(...)
    # -------------------------------------------------------------

    # 7) Dumping latest table to parquet for EDA workflow
    df_db, latest_tbl = latest_scrobble_df(engine)
    if df_db is not None:
        pq_file = HERE / "PQ" / f"{latest_tbl}.parquet"
        pq_file.parent.mkdir(exist_ok=True)
        df_db.to_parquet(pq_file, index=False)
        print(f"Latest scrobble table persisted → {pq_file}")

    print("\nWorkflow finished OK.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    finally:
        # Disposing SQLAlchemy connection pool cleanly
        engine.dispose()
