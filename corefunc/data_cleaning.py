"""
Provides data-cleaning helpers over Parquet files.
"""
from __future__ import annotations
import logging
from typing import Tuple
import pandas as pd
from helpers.io import (
    ARTIST_INFO_PQ, SCROBBLE_PQ,
    read_parquet, dump_parquet,
)

log = logging.getLogger(__name__)


def clean_artist_info() -> Tuple[int, int]:
    """
    Deduplicates artist_info.parquet, keeping the most complete row per artist.

    Returns (removed_count, remaining_count).
    """
    df = read_parquet(ARTIST_INFO_PQ)
    if df is None or df.empty:
        return 0, 0
    before = len(df)
    # Scoring completeness: mbid(2) + country(1) + disambiguation(1) + aliases(1)
    df["_score"] = (
        df["mbid"].notna().astype(int) * 2
        + df["country"].notna().astype(int)
        + df["disambiguation_comment"].notna().astype(int)
        + df["aliases"].notna().astype(int)
    )
    df = df.sort_values("_score", ascending=False).drop_duplicates(subset=["artist_name"], keep="first")
    df = df.drop(columns=["_score"])
    dump_parquet(df, ARTIST_INFO_PQ)
    removed = before - len(df)
    log.info("Cleaned artist_info: removed %d dupes, %d remain.", removed, len(df))
    return removed, len(df)
