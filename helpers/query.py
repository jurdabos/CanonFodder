"""
Provides DuckDB-based analytical queries over the Parquet file store.

Every function opens a fresh, in-process DuckDB connection, queries the
Parquet files directly, and returns the result as a pandas DataFrame or
scalar.  No persistent DuckDB database file is created.
"""
from __future__ import annotations
import logging
from pathlib import Path
import duckdb
import pandas as pd
from helpers.io import PQ_DIR, SCROBBLE_PQ, ARTIST_INFO_PQ, AVC_PQ, QA_REPORT_PQ

logger = logging.getLogger(__name__)


def query(sql: str, **params) -> pd.DataFrame:
    """
    Executes *sql* against the Parquet file store via DuckDB.

    The SQL may reference file paths directly, e.g.
    ``SELECT * FROM 'PQ/scrobble.parquet' LIMIT 10``.
    """
    con = duckdb.connect()
    try:
        return con.execute(sql, list(params.values()) if params else []).df()
    finally:
        con.close()


def _pq(path: Path) -> str:
    """Returns a quoted Parquet path string safe for SQL embedding."""
    return f"'{path.as_posix()}'"


# ── Common analytics queries ──────────────────────────────────────────────────
def top_artists(n: int = 10) -> pd.DataFrame:
    """Returns the top *n* artists by scrobble count."""
    return query(f"""
        SELECT artist_name, COUNT(*) AS play_count
        FROM {_pq(SCROBBLE_PQ)}
        WHERE artist_name IS NOT NULL AND artist_name != ''
        GROUP BY artist_name
        ORDER BY play_count DESC
        LIMIT {n}
    """)


def scrobble_count() -> int:
    """Returns the total number of scrobbles."""
    df = query(f"""
        SELECT COUNT(*) AS cnt FROM {_pq(SCROBBLE_PQ)}
        WHERE artist_name IS NOT NULL AND artist_name != ''
    """)
    return int(df["cnt"].iloc[0]) if not df.empty else 0


def unique_artists() -> int:
    """Returns the number of distinct artist names."""
    df = query(f"""
        SELECT COUNT(DISTINCT artist_name) AS cnt FROM {_pq(SCROBBLE_PQ)}
        WHERE artist_name IS NOT NULL AND artist_name != ''
    """)
    return int(df["cnt"].iloc[0]) if not df.empty else 0


def artist_info_df() -> pd.DataFrame:
    """Returns the full artist_info table as a DataFrame."""
    if not ARTIST_INFO_PQ.exists():
        return pd.DataFrame(columns=["artist_name", "mbid", "country", "disambiguation_comment", "aliases"])
    return query(f"SELECT * FROM {_pq(ARTIST_INFO_PQ)}")


def avc_df() -> pd.DataFrame:
    """Returns the full artist_variants_canonized table as a DataFrame."""
    if not AVC_PQ.exists():
        return pd.DataFrame(
            columns=["artist_variants_hash", "artist_variants_text", "canonical_name", "to_link", "comment", "stamp"]
        )
    return query(f"SELECT * FROM {_pq(AVC_PQ)}")


def scrobbles_between(start: str, end: str) -> pd.DataFrame:
    """Returns scrobbles within a date range (ISO format strings)."""
    return query(f"""
        SELECT *
        FROM {_pq(SCROBBLE_PQ)}
        WHERE play_time >= '{start}' AND play_time < '{end}'
        ORDER BY play_time
    """)


def qa_reports(
    *,
    last_n: int | None = None,
    fail_only: bool = False,
) -> pd.DataFrame:
    """
    Returns QA report rows from qa_report.parquet.

    Parameters
    ----------
    last_n : int, optional
        When set, returns only the most recent *last_n* rows.
    fail_only : bool
        When True, returns only rows where ``passed`` is False.
    """
    if not QA_REPORT_PQ.exists():
        return pd.DataFrame()
    where = "WHERE passed = false" if fail_only else ""
    order = "ORDER BY timestamp DESC"
    limit = f"LIMIT {last_n}" if last_n else ""
    return query(f"""
        SELECT * FROM {_pq(QA_REPORT_PQ)}
        {where}
        {order}
        {limit}
    """)


def artist_country_stats() -> pd.DataFrame:
    """
    Joins scrobbles with artist_info to produce play counts per country.

    Returns a DataFrame with columns: country, play_count, artist_count.
    """
    if not ARTIST_INFO_PQ.exists():
        return pd.DataFrame(columns=["country", "play_count", "artist_count"])
    return query(f"""
        SELECT
            ai.country,
            COUNT(*) AS play_count,
            COUNT(DISTINCT s.artist_name) AS artist_count
        FROM {_pq(SCROBBLE_PQ)} s
        JOIN {_pq(ARTIST_INFO_PQ)} ai
            ON s.artist_name = ai.artist_name
        WHERE ai.country IS NOT NULL AND ai.country != ''
        GROUP BY ai.country
        ORDER BY play_count DESC
    """)
