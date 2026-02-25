"""
Queries the local MusicBrainz PostgreSQL mirror for bulk artist enrichment.

Requires musicbrainz-docker to be running.  Communicates with the DB
container via ``docker exec`` and ``psql``, so no additional Python
dependencies are needed.
"""
from __future__ import annotations
import io
import logging
import os
import subprocess
from typing import Sequence
import pandas as pd
from helpers.io import (
    ARTIST_INFO_COLS,
    ARTIST_INFO_PQ,
    SCROBBLE_PQ,
    append_to_parquet,
    dump_parquet,
    read_parquet,
)

log = logging.getLogger(__name__)

# ── Docker / PostgreSQL constants ─────────────────────────────────────────────
MB_CONTAINER = os.getenv("MB_DOCKER_CONTAINER", "musicbrainz-docker-db-1")
MB_USER = os.getenv("MB_DOCKER_USER", "musicbrainz")
MB_DB = os.getenv("MB_DOCKER_DB", "musicbrainz_db")

# Batch sizes for IN-clause queries
_MBID_BATCH = 500
_NAME_BATCH = 200

# ── SQL template ──────────────────────────────────────────────────────────────
_ARTIST_SELECT = """\
SELECT
    a.name          AS artist_name,
    a.gid::text     AS mbid,
    COALESCE(iso.code, '')  AS country,
    COALESCE(a.comment, '') AS disambiguation_comment,
    COALESCE(
        (SELECT string_agg(aa.name, ',' ORDER BY aa.name)
         FROM musicbrainz.artist_alias aa
         WHERE aa.artist = a.id),
        ''
    ) AS aliases
FROM musicbrainz.artist a
LEFT JOIN musicbrainz.area        ar  ON a.area = ar.id
LEFT JOIN musicbrainz.iso_3166_1  iso ON ar.id  = iso.area"""


# ── Connectivity check ────────────────────────────────────────────────────────
def check_local_mb() -> bool:
    """Returns True if the local MusicBrainz mirror is reachable."""
    try:
        result = subprocess.run(
            ["docker", "exec", MB_CONTAINER, "psql",
             "-U", MB_USER, "-d", MB_DB, "-c", "SELECT 1"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ── Low-level helpers ─────────────────────────────────────────────────────────
def _psql_csv(sql: str) -> pd.DataFrame:
    """
    Runs *sql* wrapped in COPY … TO STDOUT CSV HEADER via docker exec.

    Passes the SQL via stdin to avoid shell-escaping headaches with
    artist names that contain quotes, backslashes, or Unicode.
    """
    copy_sql = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"
    result = subprocess.run(
        ["docker", "exec", "-i", MB_CONTAINER,
         "psql", "-U", MB_USER, "-d", MB_DB],
        input=copy_sql, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"psql error: {result.stderr.strip()}")
    stdout = result.stdout.strip()
    if not stdout:
        return pd.DataFrame(columns=ARTIST_INFO_COLS)
    return pd.read_csv(io.StringIO(stdout))


def _escape_pg(value: str) -> str:
    """Escapes a string for a PostgreSQL single-quoted literal."""
    return value.replace("'", "''")


# ── Batched queries ───────────────────────────────────────────────────────────
def _enrich_by_mbid(mbids: Sequence[str]) -> pd.DataFrame:
    """Batch-queries the local MB mirror by MBID."""
    frames: list[pd.DataFrame] = []
    mbid_list = list(mbids)
    for i in range(0, len(mbid_list), _MBID_BATCH):
        batch = mbid_list[i : i + _MBID_BATCH]
        values = ",".join(f"'{m}'" for m in batch)
        sql = f"{_ARTIST_SELECT}\nWHERE a.gid IN ({values})"
        df = _psql_csv(sql)
        if not df.empty:
            frames.append(df)
        log.info(
            "MBID batch %d–%d: %d resolved.",
            i + 1, min(i + _MBID_BATCH, len(mbid_list)), len(df),
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ARTIST_INFO_COLS)


def _enrich_by_name(names: Sequence[str]) -> pd.DataFrame:
    """Batch-queries the local MB mirror by exact artist name."""
    frames: list[pd.DataFrame] = []
    name_list = list(names)
    for i in range(0, len(name_list), _NAME_BATCH):
        batch = name_list[i : i + _NAME_BATCH]
        values = ",".join(f"'{_escape_pg(n)}'" for n in batch)
        sql = f"{_ARTIST_SELECT}\nWHERE a.name IN ({values})"
        df = _psql_csv(sql)
        if not df.empty:
            # When multiple MB artists share the same name, keeping
            # the entry with the lowest artist ID (longest-standing).
            df = df.drop_duplicates(subset=["artist_name"], keep="first")
            frames.append(df)
        log.info(
            "Name batch %d–%d: %d resolved.",
            i + 1, min(i + _NAME_BATCH, len(name_list)), len(df),
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ARTIST_INFO_COLS)


# ── Main orchestrator ─────────────────────────────────────────────────────────
def enrich_from_local_mb(*, rebuild: bool = False) -> int:
    """
    Bulk-enriches artist_info.parquet from the local MusicBrainz mirror.

    Parameters
    ----------
    rebuild : bool
        When True, overwrites artist_info.parquet instead of appending.

    Returns
    -------
    int
        Number of artist rows written.
    """
    if not check_local_mb():
        raise RuntimeError(
            f"Cannot reach local MB mirror (container: {MB_CONTAINER}). "
            "Is musicbrainz-docker running?"
        )
    scrobbles = read_parquet(SCROBBLE_PQ)
    if scrobbles is None or scrobbles.empty:
        log.warning("No scrobbles found — nothing to enrich.")
        return 0
    # Getting unique (artist_name, artist_mbid) pairs
    pairs = (
        scrobbles[["artist_name", "artist_mbid"]]
        .drop_duplicates(subset=["artist_name"])
        .copy()
    )
    # Filtering out already-known artists (unless rebuilding)
    if not rebuild:
        known = read_parquet(ARTIST_INFO_PQ)
        if known is not None and not known.empty:
            known_names = set(known["artist_name"])
            pairs = pairs[~pairs["artist_name"].isin(known_names)]
    if pairs.empty:
        log.info("All artists already enriched.")
        return 0
    log.info("Enriching %d artists from local MB mirror.", len(pairs))
    # Splitting into artists with and without MBIDs
    valid_mbid = pairs["artist_mbid"].notna() & (pairs["artist_mbid"].str.len() == 36)
    with_mbid = pairs[valid_mbid]
    without_mbid = pairs[~valid_mbid]
    result_frames: list[pd.DataFrame] = []
    # ── Tier 1: resolve by MBID (exact, fast) ─────────────────────────────
    if not with_mbid.empty:
        log.info("Tier 1: querying %d artists by MBID.", len(with_mbid))
        unique_mbids = with_mbid["artist_mbid"].unique().tolist()
        mbid_results = _enrich_by_mbid(unique_mbids)
        if not mbid_results.empty:
            # Mapping MB rows back to every scrobble artist_name that
            # shares the same MBID (preserving the scrobble-level name).
            mbid_to_names = (
                with_mbid.groupby("artist_mbid")["artist_name"]
                .apply(list)
                .to_dict()
            )
            expanded: list[dict] = []
            for _, mb_row in mbid_results.iterrows():
                scrobble_names = mbid_to_names.get(
                    mb_row["mbid"], [mb_row["artist_name"]]
                )
                for sname in scrobble_names:
                    expanded.append({
                        "artist_name": sname,
                        "mbid": mb_row["mbid"],
                        "country": mb_row["country"],
                        "disambiguation_comment": mb_row["disambiguation_comment"],
                        "aliases": mb_row["aliases"],
                    })
            result_frames.append(pd.DataFrame(expanded))
            found_mbids = set(mbid_results["mbid"])
            log.info(
                "Tier 1 resolved %d unique MBIDs → %d artist rows.",
                len(found_mbids), len(expanded),
            )
        else:
            found_mbids = set()
        # MBIDs not present in the local mirror → fall through to Tier 2
        unfound = with_mbid[~with_mbid["artist_mbid"].isin(found_mbids)]
        if not unfound.empty:
            without_mbid = pd.concat([without_mbid, unfound])
    # ── Tier 2: resolve by exact name match ───────────────────────────────
    if not without_mbid.empty:
        log.info("Tier 2: querying %d artists by name.", len(without_mbid))
        name_results = _enrich_by_name(without_mbid["artist_name"].tolist())
        if not name_results.empty:
            result_frames.append(name_results)
            log.info("Tier 2 resolved %d artists.", len(name_results))
        # Stub rows for artists with no MB match at all
        resolved_names = (
            set(name_results["artist_name"]) if not name_results.empty else set()
        )
        unresolved = without_mbid[~without_mbid["artist_name"].isin(resolved_names)]
        if not unresolved.empty:
            stubs = pd.DataFrame({
                "artist_name": unresolved["artist_name"].values,
                "mbid": "",
                "country": "",
                "disambiguation_comment": "",
                "aliases": "",
            })
            result_frames.append(stubs)
            log.info("Created %d stub rows for unresolved artists.", len(stubs))
    if not result_frames:
        log.warning("No enrichment results.")
        return 0
    # ── Combine & write ───────────────────────────────────────────────────
    combined = pd.concat(result_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["artist_name"], keep="first")
    combined = combined[ARTIST_INFO_COLS]
    if rebuild:
        dump_parquet(combined, ARTIST_INFO_PQ)
    else:
        append_to_parquet(combined, ARTIST_INFO_PQ, dedup_cols=["artist_name"])
    log.info("artist_info.parquet: %d rows written.", len(combined))
    return len(combined)
