"""
Provides the core data-gathering workflow for c9r.

All persistence goes through helpers.io (Parquet).  No DB dependency.
"""
from __future__ import annotations
import logging
from helpers.io import ingest_scrobbles, latest_scrobble_ts
from HTTP import lfAPI

log = logging.getLogger(__name__)


def run_data_gathering_workflow(
        username: str,
        *,
        full: bool = False,
        source: str = "lastfm",
) -> int:
    """
    Fetches scrobbles from the chosen source and appends them to scrobble.parquet.

    Parameters
    ----------
    username : str
        Username for the chosen source.
    full : bool
        When True, ignores existing data and fetches the full history.
    source : str
        Data source — ``"lastfm"`` or ``"listenbrainz"``.

    Returns
    -------
    int
        Number of scrobbles ingested (0 when nothing new).
    """
    since = None if full else latest_scrobble_ts()
    log.info(
        "Fetching scrobbles for %s from %s%s",
        username,
        source,
        f" since uts={since}" if since else " (full history)",
    )
    try:
        if source == "lastfm":
            df = lfAPI.fetch_scrobbles_since(username, since)
        else:
            from HTTP.lblink import fetch_scrobbles_since as lb_fetch
            df = lb_fetch(username, since)
    except Exception as exc:
        log.error("API fetch failed: %s", exc)
        raise
    if df.empty:
        log.info("No new scrobbles.")
        return 0
    n = ingest_scrobbles(df)
    log.info("Ingested %d scrobbles.", n)
    # Syncing user country (Last.fm only — LB has no user-country concept)
    if source == "lastfm":
        try:
            lfAPI.sync_user_country(username, ask=False)
        except Exception as exc:
            log.warning("Country sync skipped: %s", exc)
    return n
