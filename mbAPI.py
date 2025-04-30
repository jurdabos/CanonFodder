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

import os
import sys
import time
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import musicbrainzngs as mb
from requests import HTTPError
from sqlalchemy.orm import Session

# ────────────────────────────────────────────────────────────────────────────
# DB plumbing (lazy import to avoid circulars)
# ────────────────────────────────────────────────────────────────────────────
try:
    from DB import engine as _default_engine, SessionLocal  # type: ignore
except ModuleNotFoundError:  # running outside full project – ok for CLI test
    _default_engine = None
    SessionLocal = None

from sqlalchemy import Column, String, DateTime, func, create_engine
from sqlalchemy.orm import declarative_base, Session

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


def safe_mb_call(fun, *a, **kw):
    global last_call
    for k in range(RETRIES):
        wait = DEFAULT_RATE_LIMIT - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
            try:
                last_call = time.time()
                return fun(*a, **kw)
            except mb.NetworkError as exc:
                if k == RETRIES - 1:
                    raise time.sleep(BACKOFF * (k + 1))
# example use
# country = safe_mb_call(mb.search_artists, artist="Bohren und der Club of Gore", limit=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _last_call
        elapsed = time.time() - _last_call
        if elapsed < DEFAULT_RATE_LIMIT:
            time.sleep(DEFAULT_RATE_LIMIT - elapsed)
        result = func(*args, **kwargs)
        _last_call = time.time()
        return result

    return wrapper


def _retryable(func):
    @wraps(func)
    def inner(*args, **kwargs):
        backoff = 1
        for attempt in range(5):
            try:
                return func(*args, **kwargs)
            except mb.NetworkError as exc:
                if exc.cause and getattr(exc.cause, "status_code", None) in _RETRY_STATUS_CODES:
                    LOGGER.warning("MB temporary error %s – retry #%d in %ds", exc, attempt + 1, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
        raise RuntimeError("MusicBrainz request failed after retries")

    return inner


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init(*, engine=None, session_maker=None, user_agent: str | None = None) -> None:
    """Configure **musicbrainzngs** *once* and prepare cache table."""
    if getattr(init, "_done", False):
        return  # idempotent
    # ── MusicBrainz client config ─────────────────────────────────────────
    app = os.getenv("MB_APP_NAME", "CanonFodder")
    ver = os.getenv("MB_APP_VERSION", "dev")
    email = os.getenv("MB_CONTACT", "contact@example.com")
    mb.set_useragent(app, ver, email)
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
    Base.metadata.create_all(bind=_session_maker.kw["bind"], tables=[MBArtistCache.__table__])
    init._done = True  # type: ignore[attr-defined]


@_rate_limited
def search_artist(query: str, *, limit: int = 10):
    """
    Full-text search → list[dict].
    """
    init()  # init() from earlier
    return mb.search_artists(query=query, limit=limit)["artist-list"]


def _get_session() -> Session:
    init()  # idempotent; guarantees factory
    if _session_maker is None:  # should never happen, but mypy/pylint
        raise RuntimeError("DB not initialised")
    return _session_maker()  # ← one place that actually calls it


@_rate_limited
def lookup_artist(mbid: str) -> dict[str, Any]:
    with _get_session() as sess:  # ← uses helper
        hit = sess.get(MBArtistCache, mbid)
        if hit:
            return {
                "id": hit.mbid,
                "name": hit.artist_name,
                "country": hit.country,
                "disambiguation": hit.disambiguation,
                "from_cache": True,
            }
    data = mb.get_artist_by_id(mbid, includes=[])
    _cache_artist(data)
    return data


def fetch_country(artist_name: str) -> str | None:
    """Convenience helper – return ISO‑country for *first* search match."""
    init()
    candidates = search_artist(artist_name, limit=1)
    if not candidates:
        return None
    data = candidates[0]
    _cache_artist(data)
    return data.get("country")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_artist(data: Dict[str, Any]) -> None:
    if "id" not in data or _session_maker is None:
        return
    with _get_session() as sess:
        if sess.get(MBArtistCache, data["id"]):
            return
        sess.add(MBArtistCache(
            mbid=data["id"],
            artist_name=data.get("name"),
            country=data.get("country"),
            disambiguation=data.get("disambiguation"),
        ))
        sess.commit()


# ---------------------------------------------------------------------------
# (Experimental) write helper – adding an alias
# ---------------------------------------------------------------------------
def add_alias(mbid: str, alias: str, *, sort_name: str | None = None) -> None:
    """Add *alias* to artist – **requires authenticated user**."""
    init()
    mb.add_alias(entity="artist", gid=mbid, alias=alias, sort_name=sort_name or alias)


def _get_cached(mbid: str) -> dict | None:
    if not _session_maker:
        return None
    with _session_maker() as s:
        row = s.get(MBArtistCache, mbid)
        if row:
            return {
                "id": row.mbid,
                "name": row.artist_name,
                "country": row.country,
                "disambiguation": row.disambiguation,
                "from_cache": True,
            }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

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
