"""
Provides text-oriented data profiling functions for CLI output.

Extracts analytical insights from scrobble and artist_info Parquet stores
using DuckDB queries and RapidFuzz for fuzzy name matching.
"""
from __future__ import annotations
import logging
from pathlib import Path
import duckdb
import pandas as pd
from helpers.io import SCROBBLE_PQ, ARTIST_INFO_PQ, AVC_PQ, C_PQ

logger = logging.getLogger(__name__)


def _pq(path: Path) -> str:
    """Returns a quoted Parquet path string safe for SQL embedding."""
    return f"'{path.as_posix()}'"


def _con() -> duckdb.DuckDBPyConnection:
    """Opens a disposable in-process DuckDB connection."""
    return duckdb.connect()


# ── Overview ──────────────────────────────────────────────────────────────────
def overview_stats() -> dict:
    """
    Computes high-level scrobble and artist statistics.

    Returns a dict with total scrobbles, unique artists, date range,
    yearly totals, and play-count distribution quartiles.
    """
    if not SCROBBLE_PQ.exists():
        return {"error": "scrobble.parquet not found"}
    con = _con()
    try:
        # Aggregating basic stats
        basics = con.execute(f"""
            SELECT
                COUNT(*)                          AS total,
                COUNT(DISTINCT artist_name)       AS unique_artists,
                COUNT(DISTINCT track_title)       AS unique_tracks,
                COUNT(DISTINCT album_title)       AS unique_albums,
                MIN(play_time)                    AS earliest,
                MAX(play_time)                    AS latest
            FROM {_pq(SCROBBLE_PQ)}
            WHERE artist_name IS NOT NULL AND artist_name != ''
        """).df()
        row = basics.iloc[0]
        # Yearly totals
        yearly = con.execute(f"""
            SELECT EXTRACT(YEAR FROM play_time) AS year, COUNT(*) AS plays
            FROM {_pq(SCROBBLE_PQ)}
            WHERE artist_name IS NOT NULL AND artist_name != ''
            GROUP BY year
            ORDER BY year
        """).df()
        # Play-count distribution per artist
        dist = con.execute(f"""
            WITH counts AS (
                SELECT artist_name, COUNT(*) AS plays
                FROM {_pq(SCROBBLE_PQ)}
                WHERE artist_name IS NOT NULL AND artist_name != ''
                GROUP BY artist_name
            )
            SELECT
                MIN(plays)                                   AS min,
                APPROX_QUANTILE(plays, 0.25)                 AS q25,
                APPROX_QUANTILE(plays, 0.50)                 AS median,
                APPROX_QUANTILE(plays, 0.75)                 AS q75,
                MAX(plays)                                   AS max,
                AVG(plays)                                   AS mean,
                COUNT(*) FILTER (WHERE plays = 1)            AS singletons,
                COUNT(*) FILTER (WHERE plays <= 5)           AS lte5,
                COUNT(*)                                     AS total_artists
            FROM counts
        """).df()
        d = dist.iloc[0]
        return {
            "total_scrobbles": int(row["total"]),
            "unique_artists": int(row["unique_artists"]),
            "unique_tracks": int(row["unique_tracks"]),
            "unique_albums": int(row["unique_albums"]),
            "earliest": str(row["earliest"])[:19],
            "latest": str(row["latest"])[:19],
            "yearly": list(yearly.itertuples(index=False, name=None)),
            "distribution": {
                "min": int(d["min"]),
                "q25": int(d["q25"]),
                "median": int(d["median"]),
                "q75": int(d["q75"]),
                "max": int(d["max"]),
                "mean": round(float(d["mean"]), 2),
                "singletons": int(d["singletons"]),
                "lte5": int(d["lte5"]),
                "total_artists": int(d["total_artists"]),
            },
        }
    finally:
        con.close()


# ── Variant candidates (the Bohren problem) ──────────────────────────────────
def variant_candidates(
    *,
    threshold: int = 85,
    min_plays: int = 3,
    limit: int = 500,
) -> list[dict]:
    """
    Finds fuzzy-similar artist name pairs that likely need canonisation.

    Demonstrates the "Bohren problem": variants like "Bohren & der Club of
    Gore" vs "Bohren und der Club of Gore" split scrobble counts and distort
    rankings.

    Parameters
    ----------
    threshold : int
        Minimum RapidFuzz WRatio score (0–100) to consider a pair.
    min_plays : int
        Minimum play count for an artist to be considered.
    limit : int
        Maximum number of artists to compare (by descending play count).

    Returns a list of dicts, each with keys: variants (list of name/count
    dicts), combined_count, similarity_score.
    """
    from rapidfuzz import fuzz, process
    if not SCROBBLE_PQ.exists():
        return []
    con = _con()
    try:
        # Getting artist play counts above threshold
        artists_df = con.execute(f"""
            SELECT artist_name, COUNT(*) AS plays
            FROM {_pq(SCROBBLE_PQ)}
            WHERE artist_name IS NOT NULL AND artist_name != ''
            GROUP BY artist_name
            HAVING plays >= {min_plays}
            ORDER BY plays DESC
            LIMIT {limit}
        """).df()
    finally:
        con.close()
    if artists_df.empty:
        return []
    names = artists_df["artist_name"].tolist()
    counts = dict(zip(artists_df["artist_name"], artists_df["plays"]))
    # Finding fuzzy-similar pairs via RapidFuzz
    seen: set[frozenset] = set()
    clusters: list[dict] = []
    for name in names:
        matches = process.extract(
            name,
            names,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
            limit=10,
        )
        for match_name, score, _ in matches:
            if match_name == name:
                continue
            pair = frozenset((name, match_name))
            if pair in seen:
                continue
            seen.add(pair)
            variants = [
                {"name": name, "plays": int(counts[name])},
                {"name": match_name, "plays": int(counts[match_name])},
            ]
            clusters.append({
                "variants": sorted(variants, key=lambda v: v["plays"], reverse=True),
                "combined_count": int(counts[name]) + int(counts[match_name]),
                "similarity": round(score, 1),
            })
    # Sorting by combined count descending to highlight highest-impact merges
    clusters.sort(key=lambda c: c["combined_count"], reverse=True)
    return clusters


# ── Top artists with optional AVC canonisation ────────────────────────────────
def top_artists_profile(n: int = 20, *, canonize: bool = False) -> dict:
    """
    Returns top N artists by play count, optionally after AVC canonisation.

    When canonize=True, applies the artist_variants_canonized mapping
    before aggregation.
    """
    if not SCROBBLE_PQ.exists():
        return {"error": "scrobble.parquet not found"}
    con = _con()
    try:
        raw_df = con.execute(f"""
            SELECT artist_name, COUNT(*) AS plays
            FROM {_pq(SCROBBLE_PQ)}
            WHERE artist_name IS NOT NULL AND artist_name != ''
            GROUP BY artist_name
            ORDER BY plays DESC
        """).df()
    finally:
        con.close()
    result: dict = {"raw_top": _df_to_top(raw_df, n)}
    if canonize and AVC_PQ.exists():
        mapping = _load_avc_mapping()
        if mapping:
            canon_df = raw_df.copy()
            canon_df["artist_name"] = canon_df["artist_name"].replace(mapping)
            canon_df = (
                canon_df
                .groupby("artist_name", as_index=False)["plays"]
                .sum()
                .sort_values("plays", ascending=False)
            )
            result["canon_top"] = _df_to_top(canon_df, n)
            result["mapping_size"] = len(mapping)
        else:
            result["canon_top"] = result["raw_top"]
            result["mapping_size"] = 0
    return result


def _df_to_top(df: pd.DataFrame, n: int) -> list[dict]:
    """Converts the first n rows of a name/plays DataFrame to list of dicts."""
    return [
        {"rank": i + 1, "name": row["artist_name"], "plays": int(row["plays"])}
        for i, row in df.head(n).iterrows()
    ]


def _load_avc_mapping() -> dict[str, str]:
    """Loads the variant→canonical mapping from avc.parquet."""
    if not AVC_PQ.exists():
        return {}
    con = _con()
    try:
        df = con.execute(f"""
            SELECT artist_variants_text, canonical_name
            FROM {_pq(AVC_PQ)}
            WHERE to_link = true
        """).df()
    finally:
        con.close()
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        canonical = row["canonical_name"]
        # Splitting variant text by "{" separator
        for variant in str(row["artist_variants_text"]).split("{"):
            v = variant.strip()
            if v and v != canonical:
                mapping[v] = canonical
    return mapping


# ── Trusted companions ────────────────────────────────────────────────────────
def trusted_companions(*, start_year: int = 2006, end_year: int = 2025) -> dict:
    """
    Finds artists that appear in every year of the given range.

    Returns dict with companion artists, their per-year play counts, and
    consistency metrics (standard deviation of yearly plays).
    """
    if not SCROBBLE_PQ.exists():
        return {"error": "scrobble.parquet not found"}
    con = _con()
    try:
        # Getting unique years in the range
        years = con.execute(f"""
            SELECT DISTINCT EXTRACT(YEAR FROM play_time)::INT AS year
            FROM {_pq(SCROBBLE_PQ)}
            WHERE play_time IS NOT NULL
              AND EXTRACT(YEAR FROM play_time) BETWEEN {start_year} AND {end_year}
            ORDER BY year
        """).df()["year"].tolist()
        if not years:
            return {"years": [], "companions": []}
        num_years = len(years)
        # Finding artists present in every year
        companions_df = con.execute(f"""
            WITH yearly AS (
                SELECT
                    artist_name,
                    EXTRACT(YEAR FROM play_time)::INT AS year,
                    COUNT(*) AS plays
                FROM {_pq(SCROBBLE_PQ)}
                WHERE artist_name IS NOT NULL AND artist_name != ''
                  AND EXTRACT(YEAR FROM play_time) BETWEEN {start_year} AND {end_year}
                GROUP BY artist_name, year
            ),
            coverage AS (
                SELECT artist_name, COUNT(DISTINCT year) AS year_count
                FROM yearly
                GROUP BY artist_name
                HAVING year_count = {num_years}
            )
            SELECT y.artist_name, y.year, y.plays
            FROM yearly y
            JOIN coverage c ON y.artist_name = c.artist_name
            ORDER BY y.artist_name, y.year
        """).df()
    finally:
        con.close()
    if companions_df.empty:
        return {"years": years, "companions": []}
    # Computing consistency metrics per companion
    companions: list[dict] = []
    for artist, group in companions_df.groupby("artist_name"):
        plays = group["plays"].tolist()
        std = float(group["plays"].std())
        total = int(group["plays"].sum())
        companions.append({
            "name": artist,
            "total_plays": total,
            "std_dev": round(std, 1),
            "mean_per_year": round(total / num_years, 1),
            "yearly_plays": dict(zip(group["year"].astype(int).tolist(), [int(p) for p in plays])),
        })
    # Sorting by standard deviation (most consistent first)
    companions.sort(key=lambda c: c["std_dev"])
    return {
        "years": years,
        "year_count": num_years,
        "companions": companions,
    }


# ── Country breakdown ─────────────────────────────────────────────────────────
def country_breakdown(top_n: int = 15) -> list[dict]:
    """
    Returns the top countries by scrobble count via artist_info join.

    Each entry has: country, name (English), play_count, artist_count,
    pct (share of all enriched scrobbles).
    """
    if not SCROBBLE_PQ.exists() or not ARTIST_INFO_PQ.exists():
        return []
    con = _con()
    try:
        # Loading country-code → English name mapping from c.parquet
        name_map: dict[str, str] = {}
        if C_PQ.exists():
            names_df = con.execute(f"""
                SELECT "ISO-2" AS cc, en_name FROM {_pq(C_PQ)}
            """).df()
            name_map = dict(zip(names_df["cc"], names_df["en_name"]))
        df = con.execute(f"""
            WITH enriched AS (
                SELECT ai.country, COUNT(*) AS play_count,
                       COUNT(DISTINCT s.artist_name) AS artist_count
                FROM {_pq(SCROBBLE_PQ)} s
                JOIN {_pq(ARTIST_INFO_PQ)} ai
                    ON s.artist_name = ai.artist_name
                WHERE ai.country IS NOT NULL
                  AND ai.country != ''
                  AND ai.country != 'None'
                GROUP BY ai.country
            ),
            total AS (
                SELECT SUM(play_count) AS grand_total FROM enriched
            )
            SELECT e.country, e.play_count, e.artist_count,
                   ROUND(100.0 * e.play_count / t.grand_total, 2) AS pct
            FROM enriched e, total t
            ORDER BY e.play_count DESC
            LIMIT {top_n}
        """).df()
    finally:
        con.close()
    return [
        {
            "country": row["country"],
            "name": name_map.get(row["country"], ""),
            "play_count": int(row["play_count"]),
            "artist_count": int(row["artist_count"]),
            "pct": float(row["pct"]),
        }
        for _, row in df.iterrows()
    ]
