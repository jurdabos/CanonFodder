"""mbAPI.py – MusicBrainz connector for CanonFodder
-------------------------------------------------
High‑level wrapper around **musicbrainzngs** with automatic
* env‑driven configuration (`.env`)
* proper User‑Agent & retry logic
* transparent on‑disk cache via SQLAlchemy (`mb_artists` table)
* (optional) authenticated write helpers (add alias, tag, …)
Usage from Python
-----------------
```python
from mbAPI import lookup_artist, fetch_country
country = fetch_country("Bohren & der Club of Gore")
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

from dotenv import load_dotenv

load_dotenv()
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
import musicbrainzngs as mb

try:
    from DB import engine as _default_engine, SessionLocal  # type: ignore
    from DB.models import ArtistInfo  # Import ArtistInfo model
except ModuleNotFoundError:  # running outside full project – OK for CLI test
    _default_engine = None
    SessionLocal = None
    ArtistInfo = None
from HTTP.client import USER_AGENT as DEFAULT_UA
import re
from sqlalchemy import create_engine, select
from sqlalchemy.orm import declarative_base, Session
from tenacity import retry, stop_after_attempt, wait_exponential

_RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
_UA_RE = re.compile(r"(?P<app>[^/]+)/(?P<ver>[^ ]+) \((?P<contact>[^)]+)\)")
Base = declarative_base()

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("mbAPI")
# Configure basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Increase rate limit to avoid MusicBrainz throttling
# MusicBrainz allows 1 request per second for non-authenticated users
# Use an even more conservative limit to be safe (more wait time)
DEFAULT_RATE_LIMIT = 2.0  # 2 seconds between requests to stay well within limits
_RETRY_STATUS_CODES = {502, 503, 504}
_last_call: float = 0.0
_session_maker: Optional[Callable[[], Session]] = None

RETRIES = 4
BACKOFF = 2
last_call = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mb_call(function, /, *mbcallargs, **kw):
    @_RETRY
    @_rate_limited
    def _inner():
        return function(*mbcallargs, **kw)

    return _inner()


def _rate_limited(funktor):
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
    m = _UA_RE.fullmatch(ua)
    if not m:
        raise ValueError(f"Invalid USER_AGENT: {ua!r}")
    return m["app"], m["ver"], m["contact"]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init(*, engine=None, session_maker=None, user_agent: str | None = None) -> None:
    """Configure **musicbrainzngs** *once* and prepare connection to database."""
    if getattr(init, "_done", False):
        return  # idempotent
    # ── MusicBrainz client config ─────────────────────────────────────────
    app, ver, contact = _split_user_agent(user_agent or DEFAULT_UA)
    mb.set_useragent(app, ver, contact)
    mb_username = os.getenv("MB_USERNAME")
    mb_password = os.getenv("MB_PASSWORD")
    if mb_username and mb_password:
        mb.auth(mb_username, mb_password)
    # ── DB config  ────────────────────────────────────────────────────────
    global _session_maker
    if session_maker is not None:
        _session_maker = session_maker
    elif SessionLocal is not None:
        _session_maker = SessionLocal
    else:
        eng = engine or _default_engine or create_engine("sqlite:///canonfodder.db", echo=False)
        from sqlalchemy.orm import sessionmaker  # local import
        _session_maker = sessionmaker(bind=eng, expire_on_commit=False)
    # ArtistInfo table is managed by alembic migrations, no need to create here
    init._done = True  # type: ignore[attr-defined]


def _cache_artist(data: Dict[str, Any]) -> None:
    if _session_maker is None or ArtistInfo is None:
        return

    mbid = data.get("id")
    artist_name = data.get("name")
    if not artist_name:
        return

    with _get_session() as sess:
        # Check if artist already exists
        existing = None
        if mbid:
            existing = sess.execute(
                select(ArtistInfo).where(ArtistInfo.mbid == mbid)
            ).scalar_one_or_none()

        if not existing and artist_name:
            existing = sess.execute(
                select(ArtistInfo).where(ArtistInfo.artist_name == artist_name)
            ).scalar_one_or_none()

        # Process aliases - ensure we're handling both direct list and nested 'alias-list'
        aliases = []
        if "aliases" in data and isinstance(data["aliases"], list):
            aliases = data["aliases"]
        elif "alias-list" in data and isinstance(data["alias-list"], list):
            # Handle the MB API format where aliases are in an alias-list with name in 'alias' field
            aliases = [a.get("alias") if isinstance(a, dict) else a for a in data["alias-list"]]
        
        # Filter out any null/empty aliases and ensure they're all strings
        aliases = [str(a) for a in aliases if a]
        aliases_str = ",".join(aliases) if aliases else ""
        
        LOGGER.debug(f"Caching {len(aliases)} aliases for {artist_name}: {aliases_str[:100]}")
        
        # If exists, update it
        if existing:
            existing.artist_name = artist_name
            existing.mbid = mbid
            existing.country = data.get("country")
            existing.disambiguation_comment = data.get("disambiguation")
            # Always update aliases, even if empty - this ensures we're not keeping stale data
            existing.aliases = aliases_str
            LOGGER.debug(f"Updated artist record for {artist_name} with {len(aliases)} aliases")
        else:
            # Otherwise create new entry
            new_artist = ArtistInfo(
                artist_name=artist_name,
                mbid=mbid,
                country=data.get("country"),
                disambiguation_comment=data.get("disambiguation"),
                aliases=aliases_str
            )
            sess.add(new_artist)
            LOGGER.debug(f"Created new artist record for {artist_name} with {len(aliases)} aliases")

        sess.commit()


def _get_session() -> Session:
    init()  # idempotent; guarantees factory
    if _session_maker is None:  # should never happen, but mypy/pylint
        raise RuntimeError("DB not initialised")
    return _session_maker()


def add_alias(mbid: str, alias: str, *, sort_name: str | None = None) -> None:
    init()
    mb.add_artist_alias(  # type: ignore[attr-defined]
        gid=mbid,
        alias=alias,
        sort_name=sort_name or alias,
        locale=None, primary=None
    )


def fetch_country(artist_name: str) -> str | None:
    """Convenience helper – return ISO‑country for *first* search match."""
    init()
    candidates = search_artist(artist_name, limit=1)
    if not candidates:
        return None
    data = candidates[0]
    _cache_artist(data)
    return data.get("country")


def get_aliases(mbid: str) -> list[str]:
    """Return *current* aliases for an MBID as a plain list."""
    try:
        artist_data = lookup_artist(mbid, with_aliases=True)
        # Check if aliases is a list of dictionaries (typical MB API response)
        if "aliases" in artist_data and isinstance(artist_data["aliases"], list):
            # Extract the name from each alias dictionary
            aliases = []
            for alias in artist_data["aliases"]:
                # Handle both possible formats: direct string or dict with name key
                if isinstance(alias, str):
                    aliases.append(alias)
                elif isinstance(alias, dict) and "name" in alias:
                    aliases.append(alias["name"])
            return aliases
        # If aliases is already a list of strings
        elif "aliases" in artist_data and artist_data["aliases"]:
            return artist_data["aliases"]
        else:
            return []
    except Exception as e:
        import logging
        logging.getLogger("mbAPI").error(f"Error getting aliases for {mbid}: {e}")
        return []


@_rate_limited
def lookup_artist(mbid: str, *, with_aliases: bool = True) -> dict[str, Any]:
    includes = ["aliases"] if with_aliases else []
    data = _mb_call(mb.get_artist_by_id, mbid, includes=includes)["artist"]

    # Log the raw data for debugging
    LOGGER.debug(f"Raw MusicBrainz data for {mbid}: {data}")

    # Extract aliases and ensure it's a list
    aliases = []
    if with_aliases and "alias-list" in data:
        aliases = [a["alias"] for a in data.get("alias-list", [])]
        LOGGER.info(f"Found {len(aliases)} aliases for {mbid}: {aliases[:3]}...")

    result = {
        "id": data["id"],
        "name": data["name"],
        "country": data.get("country"),
        "aliases": aliases,
        "disambiguation": data.get("disambiguation"),
    }

    # Cache the artist data
    _cache_artist(data)

    LOGGER.debug(f"Returning processed data: {result}")
    return result


def lookup_mb_for(artist_name: str) -> str | None:
    """Look up MusicBrainz ID for an artist name."""
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
            LOGGER.info(f"Artist details: Name: '{hit_name}', Country: {hit_country}, Disambiguation: {hit_disambig}")
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
    init()
    LOGGER.info(f"Searching for artist: '{artist}', limit: {limit}")
    try:
        # Use a more robust search with additional parameters
        result = _mb_call(
            mb.search_artists,
            query=artist,  # Using query instead of artist can improve results
            artist=artist,
            alias=alias,
            primaryalias=primary_alias,
            country=country,
            limit=limit,
        )["artist-list"]
        LOGGER.info(f"Found {len(result)} results for '{artist}'")
        if result:
            first_result = result[0]
            LOGGER.info(f"Top match: {first_result.get('name')} ({first_result.get('id')})")
            if 'disambiguation' in first_result:
                LOGGER.info(f"Disambiguation: {first_result.get('disambiguation')}")
        return result
    except Exception as e:
        LOGGER.error(f"Error searching for artist '{artist}': {e}")
        return []


# ───────────────────────────────────────────────────────────────────────────
# Public high-level helper
# ───────────────────────────────────────────────────────────────────────────
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
        # Try to get complete data including aliases
        inc = ["url-rels"]
        if with_aliases:
            inc.append("aliases")
        artist = mb.get_artist_by_id(mbid, includes=inc)
        # Extract aliases
        aliases = []
        if with_aliases and hasattr(artist, 'aliases') and artist.aliases:
            LOGGER.debug(f"Found {len(artist.aliases)} aliases for {artist.name}")
            for alias in artist.aliases:
                if hasattr(alias, 'name') and alias.name and alias.name != artist.name:
                    aliases.append(alias.name)
        # Create the result dictionary
        result = {
            "id": mbid,
            "name": artist.name,
            "country": artist.country,
            "aliases": aliases,
            "disambiguation": artist.disambiguation if hasattr(artist, 'disambiguation') else None,
        }
        # Cache the result
        _cache_artist(result)
        return result
    except Exception as e:
        LOGGER.error(f"Error fetching artist {mbid} from MusicBrainz: {e}")
        raise


@_rate_limited
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
    init()  # ensure client is ready
    LOGGER.debug(f"Getting aliases for MBID: {mbid}")
    try:
        artist = mb.get_artist_by_id(mbid, includes=["aliases"])
        aliases = []
        if hasattr(artist, 'aliases') and artist.aliases:
            for alias in artist.aliases:
                if hasattr(alias, 'name') and alias.name and alias.name != artist.name:
                    aliases.append(alias.name)
                    
        LOGGER.debug(f"Found {len(aliases)} aliases for {artist.name}")
        return aliases
        
    except Exception as e:
        LOGGER.error(f"Error getting aliases for {mbid}: {e}")
        return []


@_rate_limited
def get_complete_artist_info(artist_identifier: str = None, **kwargs) -> dict[str, Any]:
    """
    Return a fully-fledged artist record.
    Parameters
    ----------
    artist_identifier : str, optional
        Either a MusicBrainz UUID (mbid) **or** a human-readable artist name.
    **kwargs : dict
        Alternative ways to specify the artist:
        - artist_mbid: MusicBrainz UUID
        - artist_name: Human-readable artist name
        - mbid: MusicBrainz UUID (alias for artist_mbid)
        
    Returns
    -------
    dict
        {
            "id": <mbid>,
            "name": <str>,
            "country": <str | None>,
            "aliases": [<str>, …],
            "disambiguation": <str | None>,
        }
    """
    def _cache_artist_locally(data: dict) -> None:
        """
        Cache artist data in the database.
        
        Parameters
        ----------
        data : dict
            Artist data from MusicBrainz API
        """
        if _session_maker is None or ArtistInfo is None:
            LOGGER.debug("Database not available for caching artist")
            return
        
        try:
            with _get_session() as sess:
                # Prepare aliases as comma-separated string
                aliases_str = None
                if "aliases" in data and data["aliases"]:
                    # Filter out any empty aliases
                    valid_aliases = [a for a in data["aliases"] if a]
                    if valid_aliases:
                        aliases_str = ",".join(valid_aliases)
                        LOGGER.debug(f"Storing {len(valid_aliases)} aliases for {data.get('name')}: {aliases_str}")
                
                # Check if artist already exists by MBID
                artist_record = None
                if "id" in data and data["id"]:
                    artist_record = sess.execute(
                        select(ArtistInfo).where(ArtistInfo.mbid == data["id"])
                    ).scalar_one_or_none()
                
                # If not found by MBID, try to find by name
                if not artist_record and "name" in data and data["name"]:
                    artist_record = sess.execute(
                        select(ArtistInfo).where(ArtistInfo.artist_name == data["name"])
                    ).scalar_one_or_none()
                
                # Update existing or create new
                if artist_record:
                    # Update existing record
                    artist_record.artist_name = data.get("name", artist_record.artist_name)
                    if "id" in data and data["id"]:
                        artist_record.mbid = data["id"]
                    if "country" in data:
                        artist_record.country = data["country"]
                    if "disambiguation" in data:
                        artist_record.disambiguation_comment = data["disambiguation"]
                    if aliases_str is not None:
                        artist_record.aliases = aliases_str
                    LOGGER.debug(f"Updated artist record for {data.get('name')}")
                else:
                    # Create new record
                    artist_record = ArtistInfo(
                        artist_name=data.get("name", "Unknown"),
                        mbid=data.get("id"),
                        country=data.get("country"),
                        disambiguation_comment=data.get("disambiguation"),
                        aliases=aliases_str
                    )
                    sess.add(artist_record)
                    LOGGER.debug(f"Created new artist record for {data.get('name')}")
                    
                sess.commit()
                
                # Debug: Verify data was saved correctly
                saved_artist = sess.execute(
                    select(ArtistInfo).where(ArtistInfo.artist_name == data.get("name", "Unknown"))
                ).scalar_one_or_none()
                if saved_artist:
                    LOGGER.debug(f"Verified saved data: artist_name={saved_artist.artist_name}, aliases={saved_artist.aliases}")
        except Exception as e:
            LOGGER.error(f"Error caching artist: {e}")
    
    try:
        # Handle different ways of passing identifiers
        if artist_identifier is None:
            artist_identifier = kwargs.get('artist_mbid') or kwargs.get('mbid') or kwargs.get('artist_name')
    
        if not artist_identifier:
            raise ValueError("No artist identifier provided")
    
        LOGGER.info(f"Getting complete artist info for: {artist_identifier}")
        init()  # ensure client and DB are ready
    
        # ── 1) decide whether this is an MBID or a plain name ────────────────
        is_mbid = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", artist_identifier))
        LOGGER.debug(f"Identifier type: {'MBID' if is_mbid else 'Name'}")
    
        # ── 2) hit local cache as fast as possible ────────────────────────────
        if _session_maker is not None and ArtistInfo is not None:
            with _get_session() as sess:
                if is_mbid:
                    cached = sess.execute(
                        select(ArtistInfo).where(ArtistInfo.mbid == artist_identifier)
                    ).scalar_one_or_none()
                else:
                    cached = sess.execute(
                        select(ArtistInfo)
                        .where(ArtistInfo.artist_name.ilike(artist_identifier))
                        .order_by(ArtistInfo.id.desc())
                    ).scalar_one_or_none()
    
                if cached and cached.country and cached.aliases:  # ensure both country and aliases are present
                    LOGGER.info(f"Cache hit for {artist_identifier}: {cached.artist_name} ({cached.country})")
                    # Process aliases from database correctly
                    aliases = cached.aliases.split(",") if cached.aliases else []
                    return {
                        "id": cached.mbid,
                        "name": cached.artist_name,
                        "country": cached.country,
                        "aliases": aliases,
                        "disambiguation": cached.disambiguation_comment,
                    }
                else:
                    if cached:
                        reason = []
                        if not cached.country: reason.append("missing country")
                        if not cached.aliases: reason.append("missing aliases")
                        LOGGER.info(f"Cache incomplete for {artist_identifier}: {', '.join(reason)}")
                    else:
                        LOGGER.info(f"Cache miss for {artist_identifier}")
                    LOGGER.info("Querying MusicBrainz for complete data")
    
        # ── 3) remote calls ──────────────────────────────────────────────────
        try:
            if is_mbid:
                LOGGER.info(f"Looking up artist by MBID: {artist_identifier}")
                data = lookup_artist(artist_identifier, with_aliases=True)
            else:
                LOGGER.info(f"Looking up MBID for artist name: {artist_identifier}")
                mbid = lookup_mb_for(artist_identifier)
                if mbid is None:  # no MB hit ‑ still persist minimal row
                    LOGGER.warning(f"No MusicBrainz ID found for {artist_identifier}")
                    _cache_artist_locally({"name": artist_identifier})
                    return {
                        "id": None,
                        "name": artist_identifier,
                        "country": None,
                        "aliases": [],
                        "disambiguation": None,
                    }
                LOGGER.info(f"Found MBID {mbid} for {artist_identifier}, fetching complete data")
                data = lookup_artist(mbid, with_aliases=True)
    
            LOGGER.info(f"Successfully retrieved data for {artist_identifier}: {data.get('name')} ({data.get('country')})")
            if data.get('aliases'):
                LOGGER.info(f"Found {len(data['aliases'])} aliases for {data.get('name')}")
            # data has already been cached by lookup_artist()
            return data
    
        except Exception as e:
            LOGGER.error(f"Error during MusicBrainz API call for {artist_identifier}: {e}")
            # Still return minimal data to avoid breaking the application
            return {
                "id": artist_identifier if is_mbid else None,
                "name": artist_identifier if not is_mbid else "Unknown Artist",
                "country": None,
                "aliases": [],
                "disambiguation": None,
            }
    
    except Exception as e:
        LOGGER.error(f"Unexpected error in get_complete_artist_info: {e}")
        # Return minimal data to avoid breaking application
        return {
            "id": None,
            "name": str(artist_identifier) if artist_identifier else "Unknown Artist",
            "country": None,
            "aliases": [],
            "disambiguation": None,
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
