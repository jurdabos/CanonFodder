"""mbAPI.py – MusicBrainz connector for CanonFodder
-------------------------------------------------
High‑level wrapper around **musicbrainzngs** with automatic
* env‑driven configuration (`.env`)
* proper User‑Agent & retry logic
* transparent on‑disk cache via SQLAlchemy (`artist_info` table)
* (optional) authenticated write helpers (add alias, tag, …)
Usage from Python
-----------------
```python
from HTTP import mbAPI
mbAPI.init()
country = mbAPI.fetch_country("Bohren & der Club of Gore")
```
CLI smoke‑test
--------------
```bash
python mbAPI.py --artist "Bohren"
```
Environment variables (add them to **.env**)
-------------------------------------------
```
# compulsory
MB_APP_NAME=CanonFodder
MB_APP_VERSION=1.0
MB_CONTACT=balazs.torda@example.com

# optional – only needed for write‑requests
MB_USERNAME=my_mb_username
MB_PASSWORD=my_mb_password
```
"""
from __future__ import annotations
import logging
import os
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
import musicbrainzngs as mb
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from HTTP.client import USER_AGENT as DEFAULT_UA
from helpers.io import ARTIST_INFO_PQ, read_parquet, append_to_parquet
from tenacity import retry, stop_after_attempt, wait_exponential
_RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
_UA_RE = re.compile(r"(?P<app>[^/]+)/(?P<ver>[^ ]+) \((?P<contact>[^)]+)\)")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("mbAPI")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
DEFAULT_RATE_LIMIT = 2.0
_last_call: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mb_call(function, /, *mbcallargs, **kw):
    """
    Wrapper for MusicBrainz API calls with retry and rate limiting.

    Parameters
    ----------
    function : callable
        The MusicBrainz API function to call
    *mbcallargs : tuple
        Positional arguments for the function
    **kw : dict
        Keyword arguments for the function

    Returns
    -------
    Any
        The result of the function call
    """

    @_RETRY
    @_rate_limited
    def _inner():
        return function(*mbcallargs, **kw)

    return _inner()


def _rate_limited(funktor):
    """
    Decorator to rate limit function calls to avoid MusicBrainz throttling.

    Parameters
    ----------
    funktor : callable
        The function to rate limit

    Returns
    -------
    callable
        The rate-limited function
    """

    @wraps(funktor)
    def wrapper(*wrapper_args, **wrapper_kw):
        global _last_call
        elapsed = time.time() - _last_call
        if elapsed < DEFAULT_RATE_LIMIT:
            time.sleep(DEFAULT_RATE_LIMIT - elapsed)
        result = funktor(*wrapper_args, **wrapper_kw)
        _last_call = time.time()
        return result

    return wrapper


def _split_user_agent(ua: str = DEFAULT_UA) -> tuple[str, str, str]:
    """
    Split a user agent string into app name, version, and contact.

    Parameters
    ----------
    ua : str, optional
        The user agent string, by default DEFAULT_UA

    Returns
    -------
    tuple[str, str, str]
        (app_name, version, contact)
    """
    m = _UA_RE.fullmatch(ua)
    if not m:
        raise ValueError(f"Invalid USER_AGENT: {ua!r}")
    return m["app"], m["ver"], m["contact"]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init(*, user_agent: str | None = None) -> None:
    """
    Configures **musicbrainzngs** once (idempotent).

    Parameters
    ----------
    user_agent : str, optional
        User agent string, by default None
    """
    if getattr(init, "_done", False):
        return
    app, ver, contact = _split_user_agent(user_agent or DEFAULT_UA)
    mb.set_useragent(app, ver, contact)
    mb_username = os.getenv("MB_USERNAME")
    mb_password = os.getenv("MB_PASSWORD")
    if mb_username and mb_password:
        mb.auth(mb_username, mb_password)
    init._done = True  # type: ignore[attr-defined]


def _cache_artist(data: Dict[str, Any]) -> None:
    """
    Caches artist data into artist_info.parquet.

    Parameters
    ----------
    data : Dict[str, Any]
        Artist data from MusicBrainz API
    """
    artist_name = data.get("name")
    if not artist_name:
        return
    # Processing aliases
    aliases: list[str] = []
    if "aliases" in data and isinstance(data["aliases"], list):
        aliases = data["aliases"]
    elif "alias-list" in data and isinstance(data["alias-list"], list):
        aliases = [a.get("alias") if isinstance(a, dict) else a for a in data["alias-list"]]
    aliases = [str(a) for a in aliases if a]
    aliases_str = ",".join(aliases)
    row = pd.DataFrame([{
        "artist_name": artist_name,
        "mbid": data.get("id") or "",
        "country": data.get("country") or "",
        "disambiguation_comment": data.get("disambiguation") or "",
        "aliases": aliases_str,
    }])
    append_to_parquet(row, ARTIST_INFO_PQ, dedup_cols=["artist_name"])
    LOGGER.debug(f"Cached {artist_name} with {len(aliases)} aliases")


def add_alias(mbid: str, alias: str, *, sort_name: str | None = None) -> None:
    """
    Add an alias to a MusicBrainz artist.

    Parameters
    ----------
    mbid : str
        MusicBrainz ID
    alias : str
        Alias to add
    sort_name : str, optional
        Sort name for the alias, by default None
    """
    init()
    mb.add_artist_alias(  # type: ignore[attr-defined]
        gid=mbid,
        alias=alias,
        sort_name=sort_name or alias,
        locale=None, primary=None
    )


def fetch_country(artist_name: str) -> str | None:
    """
    Convenience helper – return ISO‑country for *first* search match.

    Parameters
    ----------
    artist_name : str
        Artist name to search for

    Returns
    -------
    str | None
        ISO country code or None if not found
    """
    init()
    candidates = search_artist(artist_name, limit=1)
    if not candidates:
        return None
    data = candidates[0]
    _cache_artist(data)
    return data.get("country")


def lookup_mb_for(artist_name: str) -> str | None:
    """
    Look up MusicBrainz ID for an artist name.

    Parameters
    ----------
    artist_name : str
        Artist name to look up

    Returns
    -------
    str | None
        MusicBrainz ID or None if not found
    """
    try:
        LOGGER.info(f"Looking up MusicBrainz ID for artist: {artist_name}")
        # Clean up the artist name to improve search results
        cleaned_name = artist_name.strip()
        LOGGER.debug(f"Cleaned artist name: '{cleaned_name}'")
        # Search for the artist
        hit = search_artist(cleaned_name, limit=1)
        if hit:
            mbid = hit[0]["id"]
            LOGGER.info(f"Found MBID for '{artist_name}': {mbid}")
            # Log additional data for debugging
            hit_name = hit[0].get("name", "Unknown")
            hit_country = hit[0].get("country", "Unknown")
            hit_disambig = hit[0].get("disambiguation", "None")
            LOGGER.debug(f"Artist details: Name: '{hit_name}', Country: {hit_country}, Disambiguation: {hit_disambig}")
            # Also call lookup_artist to ensure we get complete data cached
            lookup_artist(mbid)
            return mbid
        LOGGER.warning(f"No MusicBrainz results found for artist: {artist_name}")
        return None
    except Exception as e:
        LOGGER.error(f"Error looking up MBID for '{artist_name}': {e}")
        return None


@_rate_limited
def search_artist(
        artist: str | None = None,
        alias: str | None = None,
        primary_alias: str | None = None,
        country: str | None = None,
        *,
        limit: int = 10,
) -> list[dict]:
    """
    Search for artists in MusicBrainz.

    Parameters
    ----------
    artist : str | None, optional
        Artist name to search for, by default None
    alias : str | None, optional
        Alias to search for, by default None
    primary_alias : str | None, optional
        Primary alias to search for, by default None
    country : str | None, optional
        Country to filter by, by default None
    limit : int, optional
        Maximum number of results to return, by default 10

    Returns
    -------
    list[dict]
        List of artist dictionaries
    """
    init()
    LOGGER.info(f"Searching for artist: '{artist}', limit: {limit}")
    try:
        # Filter out None values to avoid type errors
        params = {}
        if artist is not None:
            params["query"] = artist  # Using query instead of artist can improve results
            params["artist"] = artist
        if alias is not None:
            params["alias"] = alias
        if primary_alias is not None:
            params["primaryalias"] = primary_alias
        if country is not None:
            params["country"] = country
        params["limit"] = limit

        # Use a more robust search with additional parameters
        result = _mb_call(
            mb.search_artists,
            **params
        )["artist-list"]
        LOGGER.info(f"Found {len(result)} results for '{artist}'")
        if result:
            first_result = result[0]
            LOGGER.debug(f"Top match: {first_result.get('name')} ({first_result.get('id')})")
            if 'disambiguation' in first_result:
                LOGGER.debug(f"Disambiguation: {first_result.get('disambiguation')}")
        return result
    except Exception as e:
        LOGGER.error(f"Error searching for artist '{artist}': {e}")
        return []


@_rate_limited
def lookup_artist(mbid: str, with_aliases: bool = True) -> dict[str, Any]:
    """
    Look up an artist in MusicBrainz by MBID.

    Parameters
    ----------
    mbid : str
        MusicBrainz ID
    with_aliases : bool, optional
        Whether to fetch aliases, by default True

    Returns
    -------
    dict
        {
            "id": <mbid>,
            "name": <str>,
            "country": <str | None>,
            "aliases": [<str>, ...],
            "disambiguation": <str | None>,
        }
    """
    init()  # ensure client is ready
    LOGGER.info(f"Looking up artist by MBID: {mbid}")
    try:
        # Get artist data with includes
        includes = ["url-rels"]
        if with_aliases:
            includes.append("aliases")
        # Create params dictionary for proper type handling
        params = {"id": mbid, "includes": includes}
        data = _mb_call(mb.get_artist_by_id, **params)["artist"]

        # Extract aliases and ensure it's a list
        aliases = []
        if with_aliases:
            if "alias-list" in data:
                aliases = [a["alias"] for a in data.get("alias-list", [])]
                LOGGER.debug(f"Found {len(aliases)} aliases for {mbid}: {aliases[:3] if aliases else []}")
            elif "aliases" in data and isinstance(data["aliases"], list):
                aliases = data["aliases"]
                LOGGER.debug(f"Found {len(aliases)} aliases for {mbid}")

        # Create the result dictionary
        result = {
            "id": data["id"],
            "name": data["name"],
            "country": data.get("country"),
            "aliases": aliases,
            "disambiguation": data.get("disambiguation"),
        }

        # Cache the artist data
        _cache_artist(data)

        return result
    except Exception as e:
        LOGGER.error(f"Error fetching artist {mbid} from MusicBrainz: {e}")
        raise


def get_aliases(mbid: str) -> list[str]:
    """
    Get list of aliases for an artist.

    Parameters
    ----------
    mbid : str
        MusicBrainz ID

    Returns
    -------
    list[str]
        List of alias names
    """
    try:
        artist_data = lookup_artist(mbid, with_aliases=True)
        if "aliases" in artist_data and artist_data["aliases"]:
            return artist_data["aliases"]
        return []
    except Exception as e:
        LOGGER.error(f"Error getting aliases for {mbid}: {e}")
        return []


# ───────────────────────────────────────────────────────────────────────────
# Public high-level helper
# ───────────────────────────────────────────────────────────────────────────


@_rate_limited
def get_complete_artist_info(artist_identifier: str = None, **kwargs) -> dict[str, Any]:
    """
    Returns a fully-fledged artist record.

    Parameters
    ----------
    artist_identifier : str, optional
        Either a MusicBrainz UUID (mbid) **or** a human-readable artist name.
    **kwargs : dict
        Alternative keys: artist_mbid, artist_name, mbid.

    Returns
    -------
    dict
        {"id", "name", "country", "aliases", "disambiguation"}
    """
    try:
        if artist_identifier is None:
            artist_identifier = kwargs.get("artist_mbid") or kwargs.get("mbid") or kwargs.get("artist_name")
        if not artist_identifier:
            raise ValueError("No artist identifier provided")
        LOGGER.info(f"Getting complete artist info for: {artist_identifier}")
        init()
        is_mbid = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", artist_identifier))
        # ── Parquet cache lookup ──────────────────────────────────────────
        cached_df = read_parquet(ARTIST_INFO_PQ)
        if cached_df is not None and not cached_df.empty:
            if is_mbid:
                hit = cached_df.loc[cached_df["mbid"] == artist_identifier]
            else:
                hit = cached_df.loc[cached_df["artist_name"].str.lower() == artist_identifier.lower()]
            if not hit.empty:
                row = hit.iloc[-1]  # to take the latest in case of dupes
                if row.get("country") and row.get("aliases"):
                    LOGGER.info(f"Cache hit for {artist_identifier}")
                    return {
                        "id": row["mbid"] or None,
                        "name": row["artist_name"],
                        "country": row["country"] or None,
                        "aliases": str(row["aliases"]).split(",") if row["aliases"] else [],
                        "disambiguation": row["disambiguation_comment"] or None,
                    }
        # ── Remote calls ──────────────────────────────────────────────────
        try:
            if is_mbid:
                data = lookup_artist(artist_identifier, with_aliases=True)
            else:
                found_mbid = lookup_mb_for(artist_identifier)
                if found_mbid is None:
                    LOGGER.warning(f"No MusicBrainz ID found for {artist_identifier}")
                    _cache_artist({"name": artist_identifier})
                    return {"id": None, "name": artist_identifier, "country": None, "aliases": [], "disambiguation": None}
                data = lookup_artist(found_mbid, with_aliases=True)
            return data
        except Exception as exc:
            LOGGER.error(f"MusicBrainz API error for {artist_identifier}: {exc}")
            _cache_artist({"name": artist_identifier if not is_mbid else "Unknown Artist"})
            return {
                "id": artist_identifier if is_mbid else None,
                "name": artist_identifier if not is_mbid else "Unknown Artist",
                "country": None, "aliases": [], "disambiguation": None,
            }
    except Exception as exc:
        LOGGER.error(f"Unexpected error in get_complete_artist_info: {exc}")
        return {
            "id": None,
            "name": str(artist_identifier) if artist_identifier else "Unknown Artist",
            "country": None, "aliases": [], "disambiguation": None,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Quick MusicBrainz test")
    parser.add_argument("--artist", required=True, help="search term")
    args = parser.parse_args()
    for cand in search_artist(args.artist):
        print(json.dumps({
            "name": cand.get("name"),
            "id": cand.get("id"),
            "country": cand.get("country"),
            "disambig": cand.get("disambiguation"),
        }, ensure_ascii=False))
