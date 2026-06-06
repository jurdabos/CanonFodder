"""
Provides DuckDB-based analytical queries over the Parquet file store.

Every function opens a fresh, in-process DuckDB connection, queries the
Parquet files directly, and returns the result as a pandas DataFrame or
scalar.  No persistent DuckDB database file is created.
"""

from __future__ import annotations
import logging
from collections.abc import Sequence
from pathlib import Path
import duckdb
import pandas as pd
from helpers.io import (
    ARTIST_INFO_PQ,
    AVC_PQ,
    QA_REPORT_PQ,
    UC_PQ,
    ALIAS_SEP,
    scrobble_data_exists,
    scrobble_duckdb_from,
)

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


# ── Canonical name resolution ─────────────────────────────────────────────────
def _canonical_cte() -> str:
    """
    Returns a DuckDB CTE that maps variant artist names to their canonical form.

    Uses artist_info.aliases to build the mapping.  When artist_info is absent,
    returns an empty-result CTE so COALESCE falls through to the raw name.
    Intended for cross-module use (imported by corefunc.profile).
    """
    if not ARTIST_INFO_PQ.exists():
        return "canonical_map AS (SELECT NULL::VARCHAR AS canonical_name, NULL::VARCHAR AS variant_name WHERE false)"
    return f"""canonical_map AS (
        SELECT canonical_name, variant_name
        FROM (
            SELECT artist_name AS canonical_name,
                   artist_name AS variant_name,
                   1 AS priority
            FROM {_pq(ARTIST_INFO_PQ)}
            WHERE artist_name IS NOT NULL AND artist_name != ''
            UNION ALL
            SELECT ai.artist_name AS canonical_name,
                   TRIM(v.alias) AS variant_name,
                   2 AS priority
            FROM {_pq(ARTIST_INFO_PQ)} ai,
            LATERAL UNNEST(string_split(ai.aliases, '{ALIAS_SEP}')) AS v(alias)
            WHERE ai.aliases IS NOT NULL
              AND ai.aliases != ''
              AND LOWER(ai.aliases) != 'none'
              AND TRIM(v.alias) != ''
              AND TRIM(v.alias) != ai.artist_name
        )
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY variant_name ORDER BY priority
        ) = 1
    )"""


# ── Common analytics queries ──────────────────────────────────────────────────
def top_artists(n: int = 10) -> pd.DataFrame:
    """Returns the top *n* artists by scrobble count, resolving canonical names."""
    return query(f"""
        WITH {_canonical_cte()}
        SELECT COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
               COUNT(*) AS play_count
        FROM {scrobble_duckdb_from()} s
        LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
        WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
        GROUP BY COALESCE(cm.canonical_name, s.artist_name)
        ORDER BY play_count DESC
        LIMIT {n}
    """)


def scrobble_count() -> int:
    """Returns the total number of scrobbles."""
    df = query(f"""
        SELECT COUNT(*) AS cnt FROM {scrobble_duckdb_from()}
        WHERE artist_name IS NOT NULL AND artist_name != ''
    """)
    return int(df["cnt"].iloc[0]) if not df.empty else 0


def unique_artists() -> int:
    """Returns the number of distinct canonical artist names."""
    df = query(f"""
        WITH {_canonical_cte()}
        SELECT COUNT(DISTINCT COALESCE(cm.canonical_name, s.artist_name)) AS cnt
        FROM {scrobble_duckdb_from()} s
        LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
        WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
    """)
    return int(df["cnt"].iloc[0]) if not df.empty else 0


def top_albums(n: int = 10) -> pd.DataFrame:
    """Returns the top *n* albums by scrobble count, resolving canonical names."""
    return query(f"""
        WITH {_canonical_cte()}
        SELECT COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
               s.album_title, COUNT(*) AS play_count
        FROM {scrobble_duckdb_from()} s
        LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
        WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
          AND s.album_title IS NOT NULL AND s.album_title != ''
        GROUP BY COALESCE(cm.canonical_name, s.artist_name), s.album_title
        ORDER BY play_count DESC
        LIMIT {n}
    """)


def top_tracks(n: int = 10) -> pd.DataFrame:
    """Returns the top *n* tracks by scrobble count, resolving canonical names."""
    return query(f"""
        WITH {_canonical_cte()}
        SELECT COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
               s.track_title, s.album_title, COUNT(*) AS play_count
        FROM {scrobble_duckdb_from()} s
        LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
        WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
          AND s.track_title IS NOT NULL AND s.track_title != ''
        GROUP BY COALESCE(cm.canonical_name, s.artist_name), s.track_title, s.album_title
        ORDER BY play_count DESC
        LIMIT {n}
    """)


def recent_scrobbles(n: int = 10) -> pd.DataFrame:
    """Returns the *n* most recent scrobbles, resolving canonical names."""
    return query(f"""
        WITH {_canonical_cte()}
        SELECT COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
               s.track_title, s.album_title, s.play_time
        FROM {scrobble_duckdb_from()} s
        LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
        WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
        ORDER BY s.play_time DESC
        LIMIT {n}
    """)


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
        FROM {scrobble_duckdb_from()}
        WHERE play_time >= '{start}' AND play_time < '{end}'
        ORDER BY play_time
    """)


def qa_reports(
    *,
    last_n: int | None = None,
    fail_only: bool = False,
    target: str | None = None,
) -> pd.DataFrame:
    """
    Returns QA report rows from qa_report.parquet.

    Parameters
    ----------
    last_n : int, optional
        When set, returns only the most recent *last_n* rows.
    fail_only : bool
        When True, returns only rows where ``passed`` is False.
    target : str, optional
        When set, filters to rows matching this ``target`` value.
    """
    if not QA_REPORT_PQ.exists():
        return pd.DataFrame()
    clauses: list[str] = []
    if fail_only:
        clauses.append("passed = false")
    if target:
        clauses.append(f"target = '{target}'")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "ORDER BY timestamp DESC"
    limit = f"LIMIT {last_n}" if last_n else ""
    return query(f"""
        SELECT * FROM {_pq(QA_REPORT_PQ)}
        {where}
        {order}
        {limit}
    """)


# ── Temporal / time-series queries ─────────────────────────────────────────────
def monthly_scrobble_counts() -> pd.DataFrame:
    """
    Returns monthly scrobble counts across the entire history.

    Result columns: year (int), month (int), scrobble_count (int).
    Ordered by year, month ascending.
    """
    if not scrobble_data_exists():
        return pd.DataFrame(columns=["year", "month", "scrobble_count"])
    # Note: when scrobble data is hive-partitioned on year=YYYY, the source
    # already exposes a `year` column.  Repeating the EXTRACT expression in
    # GROUP BY avoids the alias-vs-partition-column ambiguity that triggers
    # "play_time must appear in the GROUP BY clause" in DuckDB.
    return query(f"""
        SELECT
            EXTRACT(YEAR  FROM play_time)::INT AS play_year,
            EXTRACT(MONTH FROM play_time)::INT AS play_month,
            COUNT(*) AS scrobble_count
        FROM {scrobble_duckdb_from()}
        WHERE artist_name IS NOT NULL AND artist_name != ''
          AND play_time IS NOT NULL
        GROUP BY EXTRACT(YEAR  FROM play_time)::INT,
                 EXTRACT(MONTH FROM play_time)::INT
        ORDER BY play_year, play_month
    """).rename(columns={"play_year": "year", "play_month": "month"})


def yearly_top_n_artists(top_n: int = 3) -> pd.DataFrame:
    """
    Returns the top *top_n* artists per year, resolving canonical names.

    Result columns: year (int), rank (int), artist_name (str),
    play_count (int).
    """
    if not scrobble_data_exists():
        return pd.DataFrame(columns=["year", "rank", "artist_name", "play_count"])
    cte = _canonical_cte()
    # Note: when scrobble data is hive-partitioned on year=YYYY, the source
    # already exposes a `year` column.  Repeating the EXTRACT expression in
    # GROUP BY avoids the alias-vs-partition-column ambiguity that triggers
    # "play_time must appear in the GROUP BY clause" in DuckDB.
    return query(f"""
        WITH {cte},
        yearly_artists AS (
            SELECT
                EXTRACT(YEAR FROM s.play_time)::INT AS play_year,
                COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                COUNT(*) AS play_count
            FROM {scrobble_duckdb_from()} s
            LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
            WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
              AND s.play_time IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM s.play_time)::INT,
                     COALESCE(cm.canonical_name, s.artist_name)
        ),
        ranked AS (
            SELECT play_year,
                   artist_name,
                   play_count,
                   ROW_NUMBER() OVER (
                       PARTITION BY play_year ORDER BY play_count DESC
                   ) AS rank
            FROM yearly_artists
        )
        SELECT play_year AS year, rank::INT AS rank, artist_name, play_count
        FROM ranked
        WHERE rank <= {top_n}
        ORDER BY year, rank
    """)


def listening_clock(granularity: str = "hour") -> pd.DataFrame:
    """
    Returns scrobble counts bucketed by hour-of-day or day-of-week.

    Parameters
    ----------
    granularity : str
        ``"hour"`` → columns ``hour`` (0–23), ``scrobble_count``.
        ``"weekday"`` → columns ``weekday`` (0=Mon … 6=Sun), ``scrobble_count``.
    """
    if not scrobble_data_exists():
        col = "hour" if granularity == "hour" else "weekday"
        return pd.DataFrame(columns=[col, "scrobble_count"])
    if granularity == "weekday":
        # DuckDB DAYOFWEEK: 0=Sun … 6=Sat → remap to ISO 0=Mon … 6=Sun
        return query(f"""
            SELECT
                (DAYOFWEEK(play_time) + 6) % 7 AS weekday,
                COUNT(*) AS scrobble_count
            FROM {scrobble_duckdb_from()}
            WHERE artist_name IS NOT NULL AND artist_name != ''
              AND play_time IS NOT NULL
            GROUP BY weekday
            ORDER BY weekday
        """)
    return query(f"""
        SELECT
            EXTRACT(HOUR FROM play_time)::INT AS hour,
            COUNT(*) AS scrobble_count
        FROM {scrobble_duckdb_from()}
        WHERE artist_name IS NOT NULL AND artist_name != ''
          AND play_time IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    """)


def daily_scrobble_dates() -> pd.DataFrame:
    """
    Returns distinct dates with at least one scrobble.

    Result columns: play_date (date).  Used by streak analysis.
    """
    if not scrobble_data_exists():
        return pd.DataFrame(columns=["play_date"])
    return query(f"""
        SELECT DISTINCT play_time::DATE AS play_date
        FROM {scrobble_duckdb_from()}
        WHERE artist_name IS NOT NULL AND artist_name != ''
          AND play_time IS NOT NULL
        ORDER BY play_date
    """)


def user_country_scrobble_counts() -> pd.DataFrame:
    """
    Joins scrobbles with user-country timeline via interval matching.

    For each scrobble, assigns the user's country at play_time by finding
    the uc.parquet row whose [start_date, end_date] contains play_time.
    Returns columns: country_code (str), scrobble_count (int).
    """
    if not scrobble_data_exists() or not UC_PQ.exists():
        return pd.DataFrame(columns=["country_code", "scrobble_count"])
    return query(f"""
        SELECT uc.country_code,
               COUNT(*) AS scrobble_count
        FROM {scrobble_duckdb_from()} s
        JOIN {_pq(UC_PQ)} uc
          ON s.play_time::DATE >= uc.start_date::DATE
         AND (uc.end_date IS NULL OR s.play_time::DATE <= uc.end_date::DATE)
        WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
          AND s.play_time IS NOT NULL
        GROUP BY uc.country_code
        ORDER BY scrobble_count DESC
    """)


def user_country_top_entities(top_n: int = 3) -> dict[str, pd.DataFrame]:
    """
    Returns top-N artists, albums, and tracks per user-country.

    Joins scrobbles with uc.parquet via interval matching, resolves
    canonical artist names, and ranks within each country.
    Returns a dict with keys "artists", "albums", "tracks", each a
    DataFrame with columns: country_code, rank, entity columns, play_count.
    """
    empty = {
        "artists": pd.DataFrame(columns=["country_code", "rank", "artist_name", "play_count"]),
        "albums": pd.DataFrame(columns=["country_code", "rank", "artist_name", "album_title", "play_count"]),
        "tracks": pd.DataFrame(
            columns=["country_code", "rank", "artist_name", "track_title", "album_title", "play_count"]
        ),
    }
    if not scrobble_data_exists() or not UC_PQ.exists():
        return empty
    cte = _canonical_cte()
    uc_join = (
        f"{scrobble_duckdb_from()} s"
        f" JOIN {_pq(UC_PQ)} uc"
        f"   ON s.play_time::DATE >= uc.start_date::DATE"
        f"  AND (uc.end_date IS NULL OR s.play_time::DATE <= uc.end_date::DATE)"
    )
    base_where = "WHERE s.artist_name IS NOT NULL AND s.artist_name != '' AND s.play_time IS NOT NULL"
    # Ranking artists per country
    artists = query(f"""
        WITH {cte},
        agg AS (
            SELECT uc.country_code,
                   COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                   COUNT(*) AS play_count
            FROM {uc_join}
            LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
            {base_where}
            GROUP BY uc.country_code, COALESCE(cm.canonical_name, s.artist_name)
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY country_code ORDER BY play_count DESC
            ) AS rank FROM agg
        )
        SELECT country_code, rank::INT AS rank, artist_name, play_count
        FROM ranked WHERE rank <= {top_n}
        ORDER BY country_code, rank
    """)
    # Ranking albums per country
    albums = query(f"""
        WITH {cte},
        agg AS (
            SELECT uc.country_code,
                   COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                   s.album_title,
                   COUNT(*) AS play_count
            FROM {uc_join}
            LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
            {base_where}
              AND s.album_title IS NOT NULL AND s.album_title != ''
            GROUP BY uc.country_code, COALESCE(cm.canonical_name, s.artist_name), s.album_title
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY country_code ORDER BY play_count DESC
            ) AS rank FROM agg
        )
        SELECT country_code, rank::INT AS rank, artist_name, album_title, play_count
        FROM ranked WHERE rank <= {top_n}
        ORDER BY country_code, rank
    """)
    # Ranking tracks per country
    tracks = query(f"""
        WITH {cte},
        agg AS (
            SELECT uc.country_code,
                   COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                   s.track_title,
                   s.album_title,
                   COUNT(*) AS play_count
            FROM {uc_join}
            LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
            {base_where}
              AND s.track_title IS NOT NULL AND s.track_title != ''
            GROUP BY uc.country_code, COALESCE(cm.canonical_name, s.artist_name),
                     s.track_title, s.album_title
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY country_code ORDER BY play_count DESC
            ) AS rank FROM agg
        )
        SELECT country_code, rank::INT AS rank, artist_name, track_title, album_title, play_count
        FROM ranked WHERE rank <= {top_n}
        ORDER BY country_code, rank
    """)
    return {"artists": artists, "albums": albums, "tracks": tracks}


def scrobble_counts_for_artist_patterns(labels: Sequence[str]) -> pd.DataFrame:
    """
    Returns scrobble counts for the given artist labels via substring matching.

    Each non-empty label is used as a case-insensitive ``ILIKE`` pattern
    (wrapped in ``%…%``) to bucket matching scrobbles.  Useful for quickly
    tallying recent listening before, e. g., a festival lineup.

    Result columns: canonical_artist_name (str), scrobble_count (int).
    The ``canonical_artist_name`` is the most-played ``artist_name``
    variant from the scrobble store that matched the label (preserving the
    original casing as stored).  Rows are ordered by scrobble_count
    descending.  Labels that did not match any scrobble are returned with
    scrobble_count = 0 and the original user-provided label as the
    canonical name, so callers can see the full input set.  Duplicate
    labels (case-insensitive) are deduplicated; the first occurrence wins.
    """
    empty = pd.DataFrame(columns=["canonical_artist_name", "scrobble_count"])
    if not scrobble_data_exists():
        return empty
    # Cleaning and deduplicating labels while preserving user-provided order
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        if raw is None:
            continue
        lbl = raw.strip()
        key = lbl.casefold()
        if not lbl or key in seen:
            continue
        seen.add(key)
        cleaned.append(lbl)
    if not cleaned:
        return empty

    def _esc(s: str) -> str:
        """Escapes single quotes for safe SQL literal embedding."""
        return s.replace("'", "''")

    # Bucketing each scrobble by the first label whose ILIKE pattern matches.
    case_branches = "\n".join(
        f"            WHEN artist_name ILIKE '%{_esc(lbl)}%' THEN '{_esc(lbl)}'" for lbl in cleaned
    )
    where_predicates = " OR ".join(f"artist_name ILIKE '%{_esc(lbl)}%'" for lbl in cleaned)
    df = query(f"""
        WITH labeled AS (
            SELECT
                artist_name,
                CASE
{case_branches}
                END AS bucket_label
            FROM {scrobble_duckdb_from()}
            WHERE artist_name IS NOT NULL AND artist_name != ''
              AND ({where_predicates})
        ),
        per_variant AS (
            SELECT bucket_label, artist_name, COUNT(*) AS variant_count
            FROM labeled
            WHERE bucket_label IS NOT NULL
            GROUP BY bucket_label, artist_name
        ),
        ranked AS (
            SELECT bucket_label, artist_name, variant_count,
                   ROW_NUMBER() OVER (
                       PARTITION BY bucket_label
                       ORDER BY variant_count DESC, artist_name
                   ) AS rn
            FROM per_variant
        ),
        display_name AS (
            SELECT bucket_label, artist_name AS canonical_artist_name
            FROM ranked
            WHERE rn = 1
        ),
        totals AS (
            SELECT bucket_label, CAST(SUM(variant_count) AS BIGINT) AS scrobble_count
            FROM per_variant
            GROUP BY bucket_label
        )
        SELECT d.canonical_artist_name,
               t.scrobble_count,
               t.bucket_label
        FROM totals t
        JOIN display_name d ON d.bucket_label = t.bucket_label
        ORDER BY t.scrobble_count DESC, d.canonical_artist_name
    """)
    # Filling in zero-count rows for labels that did not match anything
    if df.empty:
        matched_buckets: set[str] = set()
    else:
        matched_buckets = set(df["bucket_label"].tolist())
        df = df.drop(columns=["bucket_label"])
    missing = [lbl for lbl in cleaned if lbl not in matched_buckets]
    if missing:
        zero_df = pd.DataFrame({"canonical_artist_name": missing, "scrobble_count": [0] * len(missing)})
        df = pd.concat([df, zero_df], ignore_index=True)
    return df


def artist_country_stats() -> pd.DataFrame:
    """
    Joins scrobbles with artist_info to produce play counts per country.

    Resolves variant names through artist_info aliases before joining.
    Returns a DataFrame with columns: country, play_count, artist_count.
    """
    if not ARTIST_INFO_PQ.exists():
        return pd.DataFrame(columns=["country", "play_count", "artist_count"])
    return query(f"""
        WITH {_canonical_cte()}
        SELECT
            ai.country,
            COUNT(*) AS play_count,
            COUNT(DISTINCT COALESCE(cm.canonical_name, s.artist_name)) AS artist_count
        FROM {scrobble_duckdb_from()} s
        LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
        JOIN {_pq(ARTIST_INFO_PQ)} ai
            ON COALESCE(cm.canonical_name, s.artist_name) = ai.artist_name
        WHERE ai.country IS NOT NULL AND ai.country != ''
        GROUP BY ai.country
        ORDER BY play_count DESC
    """)
