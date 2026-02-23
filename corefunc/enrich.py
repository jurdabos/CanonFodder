"""
Provides artist-enrichment helpers over Parquet files.
"""
from __future__ import annotations
import logging
import time
import pandas as pd
from helpers.io import (
    ARTIST_INFO_PQ, SCROBBLE_PQ,
    read_parquet, append_to_parquet, dump_parquet,
)
from HTTP.mbAPI import search_artist, _cache_artist  # noqa: private ok inside project

log = logging.getLogger(__name__)


def enrich_artist_country(*, batch: int = 100) -> int:
    """
    Looks up country/MBID for scrobbled artists not yet in artist_info.parquet.

    Returns the number of artists enriched.
    """
    scrobbles = read_parquet(SCROBBLE_PQ)
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
    log.info("Enriching %d unknown artists.", len(unknown))
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
