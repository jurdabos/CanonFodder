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
except ModuleNotFoundError:  # running outside full project – OK for CLI test
    _default_engine = None
    SessionLocal = None
from HTTP.client import USER_AGENT as DEFAULT_UA
import re
from sqlalchemy import Column, String, DateTime, func, create_engine
from sqlalchemy.orm import declarative_base, Session
from tenacity import retry, stop_after_attempt, wait_exponential

_RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
_UA_RE = re.compile(r"(?P<app>[^/]+)/(?P<ver>[^ ]+) \((?P<contact>[^)]+)\)")
Base = declarative_base()


class MBArtistCache(Base):
    __tablename__ = "mb_artists"
    mbid = Column(String(36), primary_key=True)  # MusicBrainz UUID
    artist_name = Column(String(350), nullable=False, index=True)
    country = Column(String(2))  # ISO‑3166‑1 alpha‑2
    disambiguation = Column(String(558))
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("mbAPI")
DEFAULT_RATE_LIMIT = 1.2
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
    """Configure **musicbrainzngs** *once* and prepare cache table."""
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
    Base.metadata.create_all(bind=_session_maker.kw["bind"], tables=[MBArtistCache.__table__])  # type: ignore[arg-type]
    init._done = True  # type: ignore[attr-defined]


def _cache_artist(data: Dict[str, Any]) -> None:
    if _session_maker is None:
        return
    with _get_session() as sess:
        sess.merge(MBArtistCache(
            mbid=data.get("id"),  # may be None
            artist_name=data.get("name"),
            country=data.get("country"),
            disambiguation=data.get("disambiguation"),
        ))
        sess.commit()


def _get_session() -> Session:
    init()  # idempotent; guarantees factory
    if _session_maker is None:  # should never happen, but mypy/pylint
        raise RuntimeError("DB not initialised")
    return _session_maker()


def add_alias(mbid: str, alias: str, *, sort_name: str | None = None) -> None:
    init()
    mb.add_artist_alias(   # type: ignore[attr-defined]
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
    return lookup_artist(mbid, with_aliases=True)["aliases"]


@_rate_limited
def lookup_artist(mbid: str, *, with_aliases: bool = True) -> dict[str, Any]:
    includes = ["aliases"] if with_aliases else []
    data = _mb_call(mb.get_artist_by_id, mbid, includes=includes)["artist"]
    _cache_artist(data)
    return {
        "id": data["id"],
        "name": data["name"],
        "country": data.get("country"),
        "aliases": [a["alias"] for a in data.get("alias-list", [])],
        "disambiguation": data.get("disambiguation"),
    }


def lookup_mb_for(artist_name: str) -> str | None:
    hit = search_artist(artist_name, limit=1)
    return hit[0]["id"] if hit else None


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
    return _mb_call(
        mb.search_artists,
        artist=artist,
        alias=alias,
        primaryalias=primary_alias,
        country=country,
        limit=limit,
    )["artist-list"]


# ───────────────────────────────────────────────────────────────────────────
# Public high-level helper
# ───────────────────────────────────────────────────────────────────────────
def get_complete_artist_info(identifier: str) -> dict[str, Any]:
    """
    Return a fully-fledged artist record.
    Parameters
    ----------
    identifier : str
        Either a MusicBrainz UUID (mbid) **or** a human-readable artist name.
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
    init()  # ensure client and DB are ready
    # ── 1) decide whether this is an MBID or a plain name ────────────────
    is_mbid = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", identifier))
    # ── 2) hit local cache as fast as possible ────────────────────────────
    if _session_maker is not None:
        with _get_session() as sess:
            if is_mbid:
                cached = sess.get(MBArtistCache, identifier)
            else:
                cached = (
                    sess.query(MBArtistCache)
                    .filter(MBArtistCache.artist_name.ilike(identifier))
                    .order_by(MBArtistCache.fetched_at.desc())
                    .first()
                )
            if cached and cached.country:  # basic sanity
                return {
                    "id": cached.mbid,
                    "name": cached.artist_name,
                    "country": cached.country,
                    "aliases": [],  # cache table has no aliases yet
                    "disambiguation": cached.disambiguation,
                }
    # ── 3) remote calls ──────────────────────────────────────────────────
    if is_mbid:
        data = lookup_artist(identifier)
    else:
        mbid = lookup_mb_for(identifier)
        if mbid is None:  # no MB hit ‑ still persist minimal row
            _cache_artist({"name": identifier})
            return {
                "id": None,
                "name": identifier,
                "country": None,
                "aliases": [],
                "disambiguation": None,
            }
        data = lookup_artist(mbid)
    # data has already been cached by lookup_artist()
    return data


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