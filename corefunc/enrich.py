"""
Orchestrates artist enrichment over Parquet files.

Three backends are supported:
- **local** (default): local MusicBrainz PostgreSQL mirror via docker exec
- **mbapi**: remote MusicBrainz JSON API
- **lastfmapi**: Last.fm artist.getInfo for MBIDs + remote MB API for metadata

All paths write to artist_info.parquet (metadata) and backfill MBIDs
into scrobble.parquet.
"""
from __future__ import annotations
import logging
import time
import pandas as pd
from helpers.io import (
    ARTIST_INFO_PQ, read_parquet, append_to_parquet, read_scrobble_df, dump_scrobble_df,
)

log = logging.getLogger(__name__)


# ── Remote MB API enrichment ──────────────────────────────────────────────────
def enrich_artist_country(*, batch: int = 100) -> int:
    """
    Looks up country/MBID for scrobbled artists not yet in artist_info.parquet.

    Uses the remote MusicBrainz JSON API (via HTTP/mbAPI.py).
    Returns the number of artists enriched.
    """
    from HTTP.mbAPI import search_artist, _cache_artist
    scrobbles = read_scrobble_df()
    if scrobbles is None or scrobbles.empty:
        return 0
    known = read_parquet(ARTIST_INFO_PQ)
    known_names = set(known["artist_name"].tolist()) if known is not None and not known.empty else set()
    # Finding artists in scrobbles that are not yet cached
    pairs = (
        scrobbles[["artist_name", "artist_mbid"]]
        .drop_duplicates(subset=["artist_name"])
    )
    unknown = pairs[~pairs["artist_name"].isin(known_names)]
    if unknown.empty:
        log.info("All artists already in artist_info.parquet.")
        return 0
    log.info("Enriching %d unknown artists via remote MB API.", len(unknown))
    rows: list[dict] = []
    for _, rec in unknown.iterrows():
        name, mbid = rec["artist_name"], rec["artist_mbid"]
        if not mbid or pd.isna(mbid):
            hit = search_artist(name, limit=1)
            if not hit:
                continue
            cand = hit[0]
            _cache_artist(cand)  # to also persist via mbAPI
            rows.append({
                "artist_name": name,
                "mbid": cand.get("id", ""),
                "country": cand.get("country", ""),
                "disambiguation_comment": cand.get("disambiguation", ""),
                "aliases": ",".join(cand.get("aliases", [])) if isinstance(cand.get("aliases"), list) else "",
            })
        else:
            rows.append({
                "artist_name": name,
                "mbid": mbid,
                "country": "",
                "disambiguation_comment": "",
                "aliases": "",
            })
        time.sleep(0.25)  # to respect MusicBrainz rate limits
    if rows:
        df_new = pd.DataFrame(rows)
        append_to_parquet(df_new, ARTIST_INFO_PQ, dedup_cols=["artist_name"])
    log.info("Enriched %d artists.", len(rows))
    return len(rows)


# ── MBID backfill ─────────────────────────────────────────────────────────────
def backfill_mbids() -> int:
    """
    Patches missing artist_mbid values in scrobble.parquet from artist_info.parquet.

    Returns the number of scrobble rows updated.
    """
    scrobbles = read_scrobble_df()
    if scrobbles is None or scrobbles.empty:
        return 0
    artist_info = read_parquet(ARTIST_INFO_PQ)
    if artist_info is None or artist_info.empty:
        return 0
    # Building artist_name → mbid mapping (only non-empty MBIDs)
    info_with_mbid = artist_info[artist_info["mbid"].notna() & (artist_info["mbid"] != "")]
    if info_with_mbid.empty:
        return 0
    mbid_map = dict(zip(info_with_mbid["artist_name"], info_with_mbid["mbid"]))
    # Finding scrobble rows with missing MBIDs
    missing = scrobbles["artist_mbid"].isna() | (scrobbles["artist_mbid"] == "")
    if not missing.any():
        return 0
    # Applying the mapping
    filled = scrobbles.loc[missing, "artist_name"].map(mbid_map)
    updated_mask = filled.notna()
    n_updated = int(updated_mask.sum())
    if n_updated > 0:
        scrobbles.loc[filled[updated_mask].index, "artist_mbid"] = filled[updated_mask]
        dump_scrobble_df(scrobbles)
        log.info("Backfilled %d MBIDs into scrobble.parquet.", n_updated)
    return n_updated


# ── Unified orchestrator ──────────────────────────────────────────────────────
def enrich_all(
    *,
    backend: str = "local",
    rebuild: bool = False,
) -> dict[str, int]:
    """
    Runs the full enrichment pipeline.

    Parameters
    ----------
    backend : str
        One of 'local' (default), 'mbapi', or 'lastfmapi'.
    rebuild : bool
        When True, overwrites artist_info.parquet instead of appending.

    Returns
    -------
    dict with keys 'artist_info_rows' and 'mbids_backfilled'.
    """
    result: dict[str, int] = {"artist_info_rows": 0, "mbids_backfilled": 0}
    if backend == "local":
        from corefunc.mb_local import enrich_from_local_mb
        result["artist_info_rows"] = enrich_from_local_mb(rebuild=rebuild)
    elif backend == "mbapi":
        result["artist_info_rows"] = enrich_artist_country()
    elif backend == "lastfmapi":
        # Step 1: Last.fm for MBIDs in scrobble.parquet
        from HTTP.lfAPI import enrich_artist_mbids
        mbid_result = enrich_artist_mbids()
        log.info("Last.fm MBID enrichment: %s", mbid_result.get("message", ""))
        # Step 2: remote MB API for metadata in artist_info.parquet
        result["artist_info_rows"] = enrich_artist_country()
    else:
        raise ValueError(f"Unknown enrichment backend: {backend!r}")
    # Backfilling MBIDs from artist_info → scrobble
    result["mbids_backfilled"] = backfill_mbids()
    return result
