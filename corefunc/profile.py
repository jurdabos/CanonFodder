"""
Provides text-oriented data profiling functions for CLI output.

Extracts analytical insights from scrobble and artist_info Parquet stores
using DuckDB queries and RapidFuzz for fuzzy name matching.
"""
from __future__ import annotations
import calendar
import logging
from datetime import datetime, UTC, timedelta
from pathlib import Path
import duckdb
import pandas as pd
from helpers.io import SCROBBLE_PQ, ARTIST_INFO_PQ, AVC_PQ, C_PQ
from helpers.query import (
    _canonical_cte,
    artist_country_stats,
    daily_scrobble_dates,
    listening_clock as _listening_clock_query,
    monthly_scrobble_counts,
    user_country_scrobble_counts,
    user_country_top_entities,
    yearly_top_n_artists,
)

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
    cte = _canonical_cte()
    con = _con()
    try:
        # Aggregating basic stats
        basics = con.execute(f"""
            WITH {cte}
            SELECT
                COUNT(*)                          AS total,
                COUNT(DISTINCT COALESCE(cm.canonical_name, s.artist_name)) AS unique_artists,
                COUNT(DISTINCT s.track_title)     AS unique_tracks,
                COUNT(DISTINCT s.album_title)     AS unique_albums,
                MIN(s.play_time)                  AS earliest,
                MAX(s.play_time)                  AS latest
            FROM {_pq(SCROBBLE_PQ)} s
            LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
            WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
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
            WITH {cte},
            counts AS (
                SELECT COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                       COUNT(*) AS plays
                FROM {_pq(SCROBBLE_PQ)} s
                LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
                WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
                GROUP BY COALESCE(cm.canonical_name, s.artist_name)
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
    if canonize and ARTIST_INFO_PQ.exists():
        cte = _canonical_cte()
        con = _con()
        try:
            canon_df = con.execute(f"""
                WITH {cte}
                SELECT COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                       COUNT(*) AS plays
                FROM {_pq(SCROBBLE_PQ)} s
                LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
                WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
                GROUP BY COALESCE(cm.canonical_name, s.artist_name)
                ORDER BY plays DESC
            """).df()
        finally:
            con.close()
        result["canon_top"] = _df_to_top(canon_df, n)
    return result


def _df_to_top(df: pd.DataFrame, n: int) -> list[dict]:
    """Converts the first n rows of a name/plays DataFrame to list of dicts."""
    return [
        {"rank": rank, "name": row["artist_name"], "plays": int(row["plays"])}
        for rank, (_, row) in enumerate(df.head(n).iterrows(), start=1)
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
    cte = _canonical_cte()
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
            WITH {cte},
            yearly AS (
                SELECT
                    COALESCE(cm.canonical_name, s.artist_name) AS artist_name,
                    EXTRACT(YEAR FROM s.play_time)::INT AS year,
                    COUNT(*) AS plays
                FROM {_pq(SCROBBLE_PQ)} s
                LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
                WHERE s.artist_name IS NOT NULL AND s.artist_name != ''
                  AND EXTRACT(YEAR FROM s.play_time) BETWEEN {start_year} AND {end_year}
                GROUP BY COALESCE(cm.canonical_name, s.artist_name), year
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
    cte = _canonical_cte()
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
            WITH {cte},
            enriched AS (
                SELECT ai.country, COUNT(*) AS play_count,
                       COUNT(DISTINCT COALESCE(cm.canonical_name, s.artist_name)) AS artist_count
                FROM {_pq(SCROBBLE_PQ)} s
                LEFT JOIN canonical_map cm ON s.artist_name = cm.variant_name
                JOIN {_pq(ARTIST_INFO_PQ)} ai
                    ON COALESCE(cm.canonical_name, s.artist_name) = ai.artist_name
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


# ── Monthly summary (temporal backbone) ───────────────────────────────────────
def monthly_summary() -> dict:
    """
    Aggregates scrobble counts by calendar month across all years.

    Returns a dict with per-month stats (min, max, mean, total, year_count)
    and identification of the strongest and weakest months.
    """
    df = monthly_scrobble_counts()
    if df.empty:
        return {"error": "No scrobble data available."}
    months: list[dict] = []
    for m in range(1, 13):
        chunk = df[df["month"] == m]
        if chunk.empty:
            months.append({
                "month": m,
                "name": calendar.month_name[m],
                "total": 0,
                "mean": 0.0,
                "min": 0,
                "max": 0,
                "year_count": 0,
            })
            continue
        months.append({
            "month": m,
            "name": calendar.month_name[m],
            "total": int(chunk["scrobble_count"].sum()),
            "mean": round(float(chunk["scrobble_count"].mean()), 1),
            "min": int(chunk["scrobble_count"].min()),
            "max": int(chunk["scrobble_count"].max()),
            "year_count": len(chunk),
        })
    # Identifying strongest / weakest by mean
    active = [m for m in months if m["year_count"] > 0]
    strongest = max(active, key=lambda m: m["mean"]) if active else None
    weakest = min(active, key=lambda m: m["mean"]) if active else None
    # Year-over-year raw series for downstream consumers
    yearly_monthly = [
        {"year": int(row["year"]), "month": int(row["month"]), "count": int(row["scrobble_count"])}
        for _, row in df.iterrows()
    ]
    return {
        "months": months,
        "strongest": strongest,
        "weakest": weakest,
        "yearly_monthly": yearly_monthly,
    }


# ── Yearly top artists (gold / silver / bronze) ────────────────────────────
def yearly_top_artists_profile(top_n: int = 3) -> dict:
    """
    Returns a year-by-year breakdown of the top *top_n* artists.

    Each year entry contains ranked artists with play counts.
    """
    df = yearly_top_n_artists(top_n=top_n)
    if df.empty:
        return {"error": "No scrobble data available."}
    years: list[dict] = []
    for year, group in df.groupby("year"):
        artists = [
            {"rank": int(row["rank"]), "name": row["artist_name"], "plays": int(row["play_count"])}
            for _, row in group.iterrows()
        ]
        years.append({"year": int(year), "artists": artists})
    return {"years": years, "top_n": top_n}


# ── Streak analysis ─────────────────────────────────────────────────────────
def streak_analysis() -> dict:
    """
    Computes listening streak and gap statistics.

    Returns longest streak (consecutive days with ≥1 scrobble),
    current streak, longest gap, and total active days.
    """
    df = daily_scrobble_dates()
    if df.empty:
        return {"error": "No scrobble data available."}
    dates = pd.to_datetime(df["play_date"]).dt.date.sort_values().tolist()
    if not dates:
        return {"error": "No scrobble data available."}
    today = datetime.now(UTC).date()
    # Computing streaks by detecting day-over-day gaps
    longest_streak = 1
    current_streak_len = 1
    longest_gap = 0
    longest_gap_start = dates[0]
    longest_gap_end = dates[0]
    current_start = dates[0]
    best_start = dates[0]
    best_end = dates[0]
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i - 1]).days
        if delta == 1:
            current_streak_len += 1
        else:
            # Recording streak if it is the best so far
            if current_streak_len > longest_streak:
                longest_streak = current_streak_len
                best_start = current_start
                best_end = dates[i - 1]
            # Recording gap
            gap = delta - 1
            if gap > longest_gap:
                longest_gap = gap
                longest_gap_start = dates[i - 1]
                longest_gap_end = dates[i]
            current_streak_len = 1
            current_start = dates[i]
    # Finalising the last streak
    if current_streak_len > longest_streak:
        longest_streak = current_streak_len
        best_start = current_start
        best_end = dates[-1]
    # Current streak: counting backwards from today
    cur = 0
    check = today
    date_set = set(dates)
    while check in date_set:
        cur += 1
        check -= timedelta(days=1)
    return {
        "total_active_days": len(dates),
        "first_day": str(dates[0]),
        "last_day": str(dates[-1]),
        "longest_streak": longest_streak,
        "longest_streak_start": str(best_start),
        "longest_streak_end": str(best_end),
        "current_streak": cur,
        "longest_gap_days": longest_gap,
        "longest_gap_start": str(longest_gap_start),
        "longest_gap_end": str(longest_gap_end),
    }


# ── Listening clock ──────────────────────────────────────────────────────────
_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def listening_clock_profile() -> dict:
    """
    Returns hour-of-day and day-of-week scrobble distributions.

    Each bucket includes count and share percentage.
    """
    hourly_df = _listening_clock_query(granularity="hour")
    weekly_df = _listening_clock_query(granularity="weekday")
    if hourly_df.empty and weekly_df.empty:
        return {"error": "No scrobble data available."}
    # Hourly breakdown
    hourly_total = int(hourly_df["scrobble_count"].sum()) if not hourly_df.empty else 0
    hours: list[dict] = []
    if not hourly_df.empty:
        for _, row in hourly_df.iterrows():
            h = int(row["hour"])
            cnt = int(row["scrobble_count"])
            hours.append({
                "hour": h,
                "label": f"{h:02d}:00",
                "count": cnt,
                "pct": round(100.0 * cnt / hourly_total, 1) if hourly_total else 0.0,
            })
    peak_hour = max(hours, key=lambda h: h["count"]) if hours else None
    quiet_hour = min(hours, key=lambda h: h["count"]) if hours else None
    # Weekly breakdown
    weekly_total = int(weekly_df["scrobble_count"].sum()) if not weekly_df.empty else 0
    weekdays: list[dict] = []
    if not weekly_df.empty:
        for _, row in weekly_df.iterrows():
            wd = int(row["weekday"])
            cnt = int(row["scrobble_count"])
            weekdays.append({
                "weekday": wd,
                "name": _WEEKDAY_NAMES[wd] if wd < 7 else "?",
                "count": cnt,
                "pct": round(100.0 * cnt / weekly_total, 1) if weekly_total else 0.0,
            })
    peak_day = max(weekdays, key=lambda d: d["count"]) if weekdays else None
    quiet_day = min(weekdays, key=lambda d: d["count"]) if weekdays else None
    return {
        "hours": hours,
        "peak_hour": peak_hour,
        "quiet_hour": quiet_hour,
        "weekdays": weekdays,
        "peak_day": peak_day,
        "quiet_day": quiet_day,
    }


# ── Country population vs. scrobble count ──────────────────────────────────
def population_vs_scrobbles(top_n: int = 20) -> dict:
    """
    Correlates artist-origin country population with scrobble counts.

    Requires artist_info.parquet (for country) and pypopulation.
    Returns ranked lists by absolute scrobble count and per-capita rate.
    """
    import pypopulation
    stats_df = artist_country_stats()
    if stats_df.empty:
        return {"error": "No enriched country data available."}
    # Loading country-code → English name from c.parquet
    name_map: dict[str, str] = {}
    if C_PQ.exists():
        con = _con()
        try:
            names_df = con.execute(f"""
                SELECT "ISO-2" AS cc, en_name FROM {_pq(C_PQ)}
            """).df()
            name_map = dict(zip(names_df["cc"], names_df["en_name"]))
        finally:
            con.close()
    rows: list[dict] = []
    for _, row in stats_df.iterrows():
        cc = row["country"]
        if not cc or cc == "None" or len(cc) != 2:
            continue
        pop = pypopulation.get_population(cc)
        if not pop:
            continue
        plays = int(row["play_count"])
        per_million = round(plays / (pop / 1_000_000), 2)
        rows.append({
            "country": cc,
            "name": name_map.get(cc, ""),
            "play_count": plays,
            "artist_count": int(row["artist_count"]),
            "population": pop,
            "per_million": per_million,
        })
    if not rows:
        return {"error": "No population data matched."}
    by_absolute = sorted(rows, key=lambda r: r["play_count"], reverse=True)[:top_n]
    by_per_capita = sorted(rows, key=lambda r: r["per_million"], reverse=True)[:top_n]
    return {
        "by_absolute": by_absolute,
        "by_per_capita": by_per_capita,
        "total_countries": len(rows),
    }


# ── User-country breakdown ─────────────────────────────────────────────────
def user_country_profile(top_n: int = 10) -> dict:
    """
    Aggregates scrobbles by the user's physical country at play time.

    Uses uc.parquet (country timeline) interval-matched against scrobbles.
    Returns ranked entries with scrobble counts and share percentages.
    """
    df = user_country_scrobble_counts()
    if df.empty:
        return {"error": "No user-country data available (uc.parquet missing or empty)."}
    # Loading country names from c.parquet
    name_map: dict[str, str] = {}
    if C_PQ.exists():
        con = _con()
        try:
            names_df = con.execute(f"""
                SELECT "ISO-2" AS cc, en_name FROM {_pq(C_PQ)}
            """).df()
            name_map = dict(zip(names_df["cc"], names_df["en_name"]))
        finally:
            con.close()
    grand_total = int(df["scrobble_count"].sum())
    rows: list[dict] = []
    for _, row in df.head(top_n).iterrows():
        cc = row["country_code"]
        cnt = int(row["scrobble_count"])
        rows.append({
            "country": cc,
            "name": name_map.get(cc, ""),
            "scrobble_count": cnt,
            "pct": round(100.0 * cnt / grand_total, 2) if grand_total else 0.0,
        })
    return {
        "countries": rows,
        "total_scrobbles_matched": grand_total,
        "unique_countries": int(df["country_code"].nunique()),
    }


# ── User-country medal table ────────────────────────────────────────────────
def _country_name_map() -> dict[str, str]:
    """Loads ISO-2 → English country name mapping from c.parquet."""
    if not C_PQ.exists():
        return {}
    con = _con()
    try:
        df = con.execute(f"""
            SELECT "ISO-2" AS cc, en_name FROM {_pq(C_PQ)}
        """).df()
        return dict(zip(df["cc"], df["en_name"]))
    finally:
        con.close()


def _df_to_medal_entries(df: pd.DataFrame, cc: str, category: str) -> list[dict]:
    """Converts ranked rows for one country into a list of medal dicts."""
    subset = df[df["country_code"] == cc]
    entries: list[dict] = []
    for _, row in subset.iterrows():
        entry: dict = {"rank": int(row["rank"]), "plays": int(row["play_count"])}
        if category == "artists":
            entry["name"] = row["artist_name"]
        elif category == "albums":
            entry["name"] = f"{row['artist_name']}: {row['album_title']}"
        else:
            album = row["album_title"] or ""
            album_part = f" ({album})" if album else ""
            entry["name"] = f"{row['artist_name']}: {row['track_title']}{album_part}"
        entries.append(entry)
    return entries


def user_country_medal_profile(
    top_n: int = 3,
    ucn: int = 5,
    country_codes: list[str] | None = None,
) -> dict:
    """
    Builds a per-country medal table of top artists, albums, and tracks.

    Uses uc.parquet to assign scrobbles to user-countries, then ranks
    entities within each of the top *ucn* countries (by scrobble volume).
    When *country_codes* is given, only those countries are included
    (sorted by scrobble volume), and *ucn* is ignored.

    Parameters
    ----------
    top_n : int
        Number of entries per category per country.
    ucn : int
        Number of top user-countries to include.
    country_codes : list[str] | None
        Optional whitelist of ISO-2 country codes to restrict the output to.
    """
    counts_df = user_country_scrobble_counts()
    if counts_df.empty:
        return {"error": "No user-country data available (uc.parquet missing or empty)."}
    # Selecting countries: explicit whitelist or top-ucn by scrobble volume
    if country_codes:
        upper = [c.upper() for c in country_codes]
        filtered = counts_df[counts_df["country_code"].isin(upper)]
        top_codes = filtered["country_code"].tolist()
        top_counts = dict(zip(filtered["country_code"], filtered["scrobble_count"].astype(int)))
    else:
        top_codes = counts_df.head(ucn)["country_code"].tolist()
        top_counts = dict(zip(
            counts_df.head(ucn)["country_code"],
            counts_df.head(ucn)["scrobble_count"].astype(int),
        ))
    name_map = _country_name_map()
    entities = user_country_top_entities(top_n=top_n)
    countries: list[dict] = []
    for cc in top_codes:
        countries.append({
            "country": cc,
            "name": name_map.get(cc, ""),
            "scrobble_count": top_counts.get(cc, 0),
            "artists": _df_to_medal_entries(entities["artists"], cc, "artists"),
            "albums": _df_to_medal_entries(entities["albums"], cc, "albums"),
            "tracks": _df_to_medal_entries(entities["tracks"], cc, "tracks"),
        })
    return {"countries": countries, "top_n": top_n, "ucn": len(top_codes)}
