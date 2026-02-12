"""
Incrementally refreshes forevian_scrobble.parquet with new listens from ListenBrainz.
Usage:
    uv run python dev/lb_refresh.py --user forevian [--dry-run]
Avoids duplicates by:
    1. Finding the latest play_time in existing parquet
    2. Fetching only newer listens via min_ts
    3. Deduplicating on (artist_name, track_title, play_time)
"""
from __future__ import annotations
import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from lblink import LBClient

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
PQ_DIR = Path(__file__).resolve().parent.parent / "PQ"
# ListenBrainz API returns max 100 listens per request
MAX_PER_REQUEST = 100


def get_parquet_path(username: str) -> Path:
    """Returns the parquet path for a given username."""
    return PQ_DIR / f"{username}_scrobble.parquet"


def load_existing(parquet_path: Path) -> pd.DataFrame:
    """Loads existing parquet or returns empty DataFrame with correct schema."""
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info("Loaded %d existing scrobbles from %s", len(df), parquet_path.name)
        return df
    logger.info("No existing parquet found at %s – starting fresh", parquet_path.name)
    return pd.DataFrame(columns=["artist_name", "album_title", "track_title", "play_time"])


def get_latest_timestamp(df: pd.DataFrame) -> int:
    """Returns latest play_time as Unix timestamp, or 0 if empty."""
    if df.empty or "play_time" not in df.columns:
        return 0
    max_dt = df["play_time"].max()
    if pd.isna(max_dt):
        return 0
    return int(max_dt.timestamp())


def fetch_all_new_listens(client: LBClient, username: str, min_ts: int) -> List[Dict[str, Any]]:
    """
    Fetches all listens newer than min_ts using pagination.
    Starts from most recent and paginates backward until reaching min_ts.
    Includes 1-second delays between requests to respect API rate limits.
    """
    all_listens = []
    max_ts = None  # Start from most recent (no upper bound)
    request_count = 0
    while True:
        # Rate limiting: 1-second pause between requests (skip first request)
        if request_count > 0:
            logger.debug("Rate limit pause (1s)...")
            time.sleep(1)
        # Fetching without min_ts to get proper descending order from newest
        listens = client.get_listens(username, max_ts=max_ts, count=MAX_PER_REQUEST)
        request_count += 1
        if not listens:
            logger.info("No more listens returned – pagination complete")
            break
        # Filtering out any listens at or before min_ts
        new_listens = [l for l in listens if l["listened_at"] > min_ts]
        if not new_listens:
            logger.info("All listens in batch are older than min_ts – pagination complete")
            break
        all_listens.extend(new_listens)
        newest_in_batch = datetime.fromtimestamp(max(l["listened_at"] for l in new_listens), timezone.utc)
        oldest_in_batch = datetime.fromtimestamp(min(l["listened_at"] for l in new_listens), timezone.utc)
        logger.info(
            "Batch %d: %d listens [%s → %s] (total: %d)",
            request_count,
            len(new_listens),
            oldest_in_batch.strftime("%Y-%m-%d %H:%M"),
            newest_in_batch.strftime("%Y-%m-%d %H:%M"),
            len(all_listens),
        )
        # Checking if we've reached or passed our target timestamp
        oldest_ts = min(listen["listened_at"] for listen in listens)
        if oldest_ts <= min_ts:
            logger.info("Reached min_ts boundary – pagination complete")
            break
        # Pagination: move max_ts backward
        max_ts = oldest_ts
        if len(listens) < MAX_PER_REQUEST:
            logger.info("Partial batch received – pagination complete")
            break
    logger.info("Finished fetching: %d requests, %d total listens", request_count, len(all_listens))
    return all_listens


def listens_to_dataframe(listens: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts ListenBrainz listens to DataFrame matching existing schema."""
    if not listens:
        return pd.DataFrame(columns=["artist_name", "album_title", "track_title", "play_time"])
    data = []
    for entry in listens:
        md = entry["track_metadata"]
        ts = entry.get("listened_at", 0)
        data.append({
            "artist_name": md.get("artist_name", ""),
            "album_title": md.get("release_name") or None,
            "track_title": md.get("track_name", ""),
            "play_time": datetime.fromtimestamp(ts, timezone.utc) if ts else None,
        })
    df = pd.DataFrame(data)
    df["play_time"] = pd.to_datetime(df["play_time"], utc=True)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate scrobbles based on artist, track, and exact play_time."""
    before = len(df)
    df = df.drop_duplicates(subset=["artist_name", "track_title", "play_time"], keep="first")
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicates", removed)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally refresh forevian_scrobble.parquet")
    parser.add_argument("--user", required=True, help="ListenBrainz username")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched without saving")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    # Loading existing data
    parquet_path = get_parquet_path(args.user)
    existing_df = load_existing(parquet_path)
    latest_ts = get_latest_timestamp(existing_df)
    if latest_ts:
        latest_dt = datetime.fromtimestamp(latest_ts, timezone.utc)
        logger.info("Latest existing scrobble: %s", latest_dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
    else:
        logger.info("No existing data – will fetch all available listens")
    # Fetching new listens
    client = LBClient()
    logger.info("Fetching listens for %s newer than ts=%d", args.user, latest_ts)
    new_listens = fetch_all_new_listens(client, args.user, min_ts=latest_ts)
    if not new_listens:
        logger.info("No new listens found – parquet is up to date")
        return
    logger.info("Fetched %d new listens total", len(new_listens))
    # Converting and merging
    new_df = listens_to_dataframe(new_listens)
    if args.dry_run:
        logger.info("[DRY RUN] Would add %d new scrobbles:", len(new_df))
        print(new_df.head(10).to_string())
        return
    # Combining and deduplicating
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = deduplicate(combined)
    combined = combined.sort_values("play_time").reset_index(drop=True)
    # Saving
    combined.to_parquet(parquet_path, compression="zstd", index=False)
    added = len(combined) - len(existing_df)
    logger.info("✅ Saved %d total scrobbles (+%d new) to %s", len(combined), added, parquet_path)


if __name__ == "__main__":
    main()
