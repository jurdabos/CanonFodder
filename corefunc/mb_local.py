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
import re
import subprocess
from collections import Counter, defaultdict
from typing import Sequence

import pandas as pd

from helpers.io import (
    ALIAS_SEP,
    ARTIST_INFO_COLS,
    ARTIST_INFO_PQ,
    PQ_DIR,
    append_to_parquet,
    dump_parquet,
    read_parquet,
    read_scrobble_df,
)

log = logging.getLogger(__name__)

# ── Docker / PostgreSQL constants ─────────────────────────────────────────────
MB_CONTAINER = os.getenv("MB_DOCKER_CONTAINER", "musicbrainz-docker-db-1")
MB_USER = os.getenv("MB_DOCKER_USER", "musicbrainz")
MB_DB = os.getenv("MB_DOCKER_DB", "musicbrainz_db")

# Batch sizes for IN-clause queries
_MBID_BATCH = 500
_NAME_BATCH = 200
# Tier 3 catalogue-based resolution constants
_COLLAB_RE = re.compile(r"[,&]| feat[\.\s]| with | vs[\.\s]", re.IGNORECASE)
_MIN_TRACK_LEN = 3
_AMBIGUITY_LIMIT = 20
_MIN_CATALOGUE_HITS = 2
_DOMINANCE_THRESHOLD = 0.6
_SOLO_DISCO_PQ = PQ_DIR / "mbdb_discography_solo.parquet"

# ── SQL template ──────────────────────────────────────────────────────────────
# Aliases are aggregated with the canonical ALIAS_SEP ('{'), NOT a comma: many
# artist names legitimately contain commas (e.g. "Lustmord, Bohren & Der Club Of
# Gore"), so a comma separator would be indistinguishable from an in-name comma
# and would break alias parsing in helpers.query._canonical_cte.
_ARTIST_SELECT = f"""\
SELECT
    a.name          AS artist_name,
    a.gid::text     AS mbid,
    COALESCE(iso.code, '')  AS country,
    COALESCE(a.comment, '') AS disambiguation_comment,
    COALESCE(
        (SELECT string_agg(aa.name, '{ALIAS_SEP}' ORDER BY aa.name)
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
            ["docker", "exec", MB_CONTAINER, "psql", "-U", MB_USER, "-d", MB_DB, "-c", "SELECT 1"],
            capture_output=True,
            text=True,
            timeout=10,
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
        ["docker", "exec", "-i", MB_CONTAINER, "psql", "-U", MB_USER, "-d", MB_DB],
        input=copy_sql,
        capture_output=True,
        text=True,
        timeout=300,
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
            i + 1,
            min(i + _MBID_BATCH, len(mbid_list)),
            len(df),
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
            i + 1,
            min(i + _NAME_BATCH, len(name_list)),
            len(df),
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ARTIST_INFO_COLS)


# ── Gender lookup ─────────────────────────────────────────────────────────────
def lookup_artist_genders(mbids: Sequence[str]) -> dict[str, str]:
    """Batch-queries the local MB mirror for artist gender by MBID.

    Returns mbid → gender name, with "(unknown)" when the artist row exists
    without a gender; MBIDs absent from the mirror are simply not keys.
    """
    mapping: dict[str, str] = {}
    mbid_list = list(dict.fromkeys(mbids))
    for i in range(0, len(mbid_list), _MBID_BATCH):
        batch = mbid_list[i : i + _MBID_BATCH]
        values = ",".join(f"'{m}'" for m in batch)
        sql = f"""\
SELECT a.gid::text AS mbid, COALESCE(g.name, '(unknown)') AS gender
FROM musicbrainz.artist a
LEFT JOIN musicbrainz.gender g ON a.gender = g.id
WHERE a.gid IN ({values})"""
        df = _psql_csv(sql)
        if not df.empty:
            mapping.update(zip(df["mbid"], df["gender"]))
        log.info(
            "Gender batch %d–%d: %d resolved.",
            i + 1,
            min(i + _MBID_BATCH, len(mbid_list)),
            len(df),
        )
    return mapping


# ── Tier 3: catalogue-based MBID resolution ──────────────────────────────────
def _build_track_reverse_index(
    scrobbles: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Builds a case-insensitive track_name → {mbid: count} reverse index.

    Combines two sources:
    1. Scrobble data — tracks from artists that already have a valid MBID.
    2. MBDB solo-credit discography cache — if mbdb_discography_solo.parquet
       exists (built by Exp 14+).
    Only tracks with > _MIN_TRACK_LEN characters are included.
    Ambiguous tracks (appearing under > _AMBIGUITY_LIMIT MBIDs) are excluded.
    """
    index: dict[str, dict[str, int]] = defaultdict(Counter)
    # ── Source 1: scrobble data ────────────────────────────────────────────
    has_mbid = scrobbles["artist_mbid"].notna() & (scrobbles["artist_mbid"].str.len() == 36)
    valid = scrobbles.loc[has_mbid, ["track_title", "artist_mbid"]].dropna()
    for _, row in valid.iterrows():
        t = str(row["track_title"]).strip().lower()
        if len(t) > _MIN_TRACK_LEN:
            index[t][row["artist_mbid"]] += 1
    n_scrobble = len(index)
    # ── Source 2: MBDB solo-credit cache (if available) ────────────────────
    if _SOLO_DISCO_PQ.exists():
        disco = read_parquet(_SOLO_DISCO_PQ)
        for _, row in disco.iterrows():
            ts = row.get("tracks_str", "")
            mbid = row["mbid"]
            if ts and str(ts) != "nan":
                for t in ts.split("{"):
                    t_lower = t.strip().lower()
                    if len(t_lower) > _MIN_TRACK_LEN:
                        index[t_lower][mbid] += 1
        log.info(
            "Track reverse index: %d from scrobbles, %d total (with MBDB cache).",
            n_scrobble,
            len(index),
        )
    else:
        log.info("Track reverse index: %d entries from scrobbles (no MBDB cache).", n_scrobble)
    # Filtering out ambiguous tracks
    return {t: dict(mbids) for t, mbids in index.items() if len(mbids) <= _AMBIGUITY_LIMIT}


def _resolve_by_catalogue(
    unresolved_names: Sequence[str],
    scrobbles: pd.DataFrame,
    track_index: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Resolves unresolved artist names by matching their scrobble tracks.

    For each non-collaborative unresolved artist with >= 2 tracks:
      1. Looks up each track in the reverse index.
      2. Counts which MBIDs appear across matched tracks.
      3. Assigns the dominant MBID if it has >= _MIN_CATALOGUE_HITS hits
         and > _DOMINANCE_THRESHOLD share of total hits.
    Returns a DataFrame with columns matching _ARTIST_SELECT output.
    """
    # Filtering out collaborative credit names
    candidates = [n for n in unresolved_names if not _COLLAB_RE.search(n)]
    if not candidates:
        return pd.DataFrame(columns=ARTIST_INFO_COLS)
    # Collecting tracks per unresolved artist
    cand_set = set(candidates)
    unres_sc = scrobbles[scrobbles["artist_name"].isin(cand_set)]
    tracks_by_artist = (
        unres_sc.groupby("artist_name")["track_title"]
        .apply(lambda x: list({t for t in x.dropna().unique() if str(t).strip()}))
        .to_dict()
    )
    # Attempting resolution
    resolved: list[tuple[str, str]] = []  # (artist_name, mbid)
    for name, tracks in tracks_by_artist.items():
        if len(tracks) < 2:
            continue
        mbid_hits: Counter = Counter()
        for t in tracks:
            key = str(t).strip().lower()
            if key in track_index:
                for mbid in track_index[key]:
                    mbid_hits[mbid] += 1
        if not mbid_hits:
            continue
        best_mbid, best_count = mbid_hits.most_common(1)[0]
        total = sum(mbid_hits.values())
        dominance = best_count / total
        if best_count >= _MIN_CATALOGUE_HITS and dominance > _DOMINANCE_THRESHOLD:
            resolved.append((name, best_mbid))
    if not resolved:
        return pd.DataFrame(columns=ARTIST_INFO_COLS)
    # Fetching full artist metadata for resolved MBIDs
    unique_mbids = list({mbid for _, mbid in resolved})
    mb_data = _enrich_by_mbid(unique_mbids)
    if mb_data.empty:
        return pd.DataFrame(columns=ARTIST_INFO_COLS)
    mbid_to_info = {}
    for _, row in mb_data.iterrows():
        mbid_to_info[row["mbid"]] = row
    # Building result rows (preserving scrobble-level artist_name)
    rows: list[dict] = []
    for name, mbid in resolved:
        if mbid not in mbid_to_info:
            continue
        info = mbid_to_info[mbid]
        rows.append(
            {
                "artist_name": name,
                "mbid": mbid,
                "country": info["country"],
                "disambiguation_comment": info["disambiguation_comment"],
                "aliases": info["aliases"],
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=ARTIST_INFO_COLS)


# ── Main orchestrator ─────────────────────────────────────────────────────────
def enrich_from_local_mb(*, rebuild: bool = False) -> int:
    """Bulk-enriches artist_info.parquet from the local MusicBrainz mirror.

    Three resolution tiers:
      - Tier 1: exact MBID lookup (fast, highest confidence).
      - Tier 2: exact artist name match.
      - Tier 3: catalogue-based resolution — matches scrobble tracks against
        a reverse index built from known MBID→track associations.  Only
        non-collaborative artist names with ≥ 2 discriminative track hits
        pointing to a dominant MBID are resolved.

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
        raise RuntimeError(f"Cannot reach local MB mirror (container: {MB_CONTAINER}). Is musicbrainz-docker running?")
    scrobbles = read_scrobble_df()
    if scrobbles is None or scrobbles.empty:
        log.warning("No scrobbles found — nothing to enrich.")
        return 0
    # Getting unique (artist_name, artist_mbid) pairs
    pairs = scrobbles[["artist_name", "artist_mbid"]].drop_duplicates(subset=["artist_name"]).copy()
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
            mbid_to_names = with_mbid.groupby("artist_mbid")["artist_name"].apply(list).to_dict()
            expanded: list[dict] = []
            for _, mb_row in mbid_results.iterrows():
                scrobble_names = mbid_to_names.get(mb_row["mbid"], [mb_row["artist_name"]])
                for sname in scrobble_names:
                    expanded.append(
                        {
                            "artist_name": sname,
                            "mbid": mb_row["mbid"],
                            "country": mb_row["country"],
                            "disambiguation_comment": mb_row["disambiguation_comment"],
                            "aliases": mb_row["aliases"],
                        }
                    )
            result_frames.append(pd.DataFrame(expanded))
            found_mbids = set(mbid_results["mbid"])
            log.info(
                "Tier 1 resolved %d unique MBIDs → %d artist rows.",
                len(found_mbids),
                len(expanded),
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
        # Tracking still-unresolved names for Tier 3
        resolved_names = set(name_results["artist_name"]) if not name_results.empty else set()
        unresolved = without_mbid[~without_mbid["artist_name"].isin(resolved_names)]
    else:
        unresolved = pd.DataFrame(columns=without_mbid.columns)
    # ── Tier 3: catalogue-based resolution (track overlap) ─────────────
    if not unresolved.empty:
        log.info("Tier 3: attempting catalogue-based resolution for %d artists.", len(unresolved))
        track_index = _build_track_reverse_index(scrobbles)
        catalogue_results = _resolve_by_catalogue(
            unresolved["artist_name"].tolist(),
            scrobbles,
            track_index,
        )
        if not catalogue_results.empty:
            result_frames.append(catalogue_results)
            log.info("Tier 3 resolved %d artists by catalogue overlap.", len(catalogue_results))
            resolved_by_cat = set(catalogue_results["artist_name"])
            unresolved = unresolved[~unresolved["artist_name"].isin(resolved_by_cat)]
    # Stub rows for artists with no match at all
    if not unresolved.empty:
        stubs = pd.DataFrame(
            {
                "artist_name": unresolved["artist_name"].values,
                "mbid": "",
                "country": "",
                "disambiguation_comment": "",
                "aliases": "",
            }
        )
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
