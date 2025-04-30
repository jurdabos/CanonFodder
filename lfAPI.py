from dotenv import load_dotenv
load_dotenv()
import hashlib
from HTTP.client import make_request, USER_AGENT
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

ENDPOINT = "https://ws.audioscrobbler.com/2.0/"
HERE = Path(__file__).resolve().parent
# Constants
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
LASTFM_SECRET = os.environ.get("LASTFM_SECRET")
if not LASTFM_API_KEY:
    raise RuntimeError(
        "LASTFM_API_KEY not found – did you forget to put it into .env "
        "or call load_dotenv(<path>) before importing lfAPI?")


def _clean_track(rec: object) -> dict | None:
    """Returns a normalised track‐dict or None."""
    if not isinstance(rec, dict):  # ← to guard against plain strings
        return None
    # rows for the *currently playing* track having no 'date'
    uts = int(rec.get("date", {}).get("uts", 0) or 0)
    return {
        "artist_name": rec["artist"]["#text"],
        "album_title": rec["album"]["#text"],
        "track_title": rec["name"],
        "uts": uts,
    }


# --------------------------------------------------------------
# Misc data fetch
# --------------------------------------------------------------
def fetch_misc_data_from_lastfmapi(user: str | None = None) -> None:
    """
    One-screen “profile snapshot”.
    Fetch ‘top’/friends/info/loved/recent’ for *user* and print a compact digest.
    """
    if not user:
        user = input("Enter your Last.fm username: ").strip()
    # ── API calls ───────────────────────────────────────────────────────────────
    top_artists = lastfm_request("user.getTopArtists", user=user)
    top_albums = lastfm_request("user.getTopAlbums", user=user)
    top_tracks = lastfm_request("user.getTopTracks", user=user)
    try:
        friends = lastfm_request("user.getFriends", user=user)
    except LastFMError as exc:  # own wrapper around RuntimeError
        if exc.code == 6:  # -> not fatal, continue workflow
            logging.info("Friend list private – skipping")
            friends = []
        else:
            raise  # re-raise unknown problems
    infos = lastfm_request("user.getInfo", user=user)
    lovedtracks = lastfm_request("user.getLovedTracks", user=user)
    # ── Top artists ────────────────────────────────────────────────────────────
    if top_artists and "topartists" in top_artists:
        print("Top Artists:")
        for artist in top_artists["topartists"]["artist"][:2]:
            print("   ", artist.get("name", "N/A"))
    # ── Top albums (title  +  “by ‹artist›”) ───────────────────────────────────
    if top_albums and "topalbums" in top_albums:
        print("\nTop Albums:")
        for album in top_albums["topalbums"]["album"][:2]:
            print(f"   {album.get('name', 'N/A')}  by {album.get('artist', {}).get('name', '?')}")
    # ── Top tracks (title  +  “by ‹artist›”) ───────────────────────────────────
    if top_tracks and "toptracks" in top_tracks:
        print("\nTop Tracks:")
        for track in top_tracks["toptracks"]["track"][:2]:
            title = track.get("name", "N/A")
            artist = track.get("artist", {}).get("name", "N/A")
            print(f"   {title} by {artist}")
    # ── User info ──────────────────────────────────────────────────────────────
    if infos and "user" in infos:
        u = infos["user"]
        print("\nUser Info:")
        print("   Name:", u.get("name", "N/A"))
        print("   Country:", u.get("country", "N/A"))
        print("   Playcount:", u.get("playcount", "N/A"))
    # ── Loved tracks ───────────────────────────────────────────────────────────
    if lovedtracks and "lovedtracks" in lovedtracks:
        print("\n2 loved tracks:")
        for lt in lovedtracks["lovedtracks"]["track"][:2]:
            title = lt.get("name", "N/A")
            artist = lt.get("artist", {}).get("name", "N/A")
            print(f"   {title} by {artist}")


# --------------------------------------------------------------
# Get all pages of recent tracks from last.fm API to df
# --------------------------------------------------------------
def fetch_recent_tracks_all_pages(user: str) -> pd.DataFrame:
    """Return all scrobbles for *user* as a DataFrame."""
    bag: list[dict] = []
    pageno = 1
    while True:
        rsp = lastfm_request("user.getRecentTracks",
                             user=user, page=pageno, limit=200)
        if not rsp or "recenttracks" not in rsp:
            logging.warning("Empty / invalid reply for page %s – stopping", pageno)
            break
        raw = rsp["recenttracks"].get("track", [])
        batch = [_clean_track(t) for t in raw]
        batch = [b for b in batch if b]  # to drop Nones
        bag.extend(batch)
        # Are we on the last page?
        meta = rsp["recenttracks"].get("@attr", {})
        if int(meta.get("page", 0)) >= int(meta.get("totalPages", 0)):
            break
        pageno += 1
    return pd.DataFrame(bag)


# --------------------------------------------------------------
# Authentication function for Last.fm API
# --------------------------------------------------------------
def generate_lastfm_signature(params, secret):
    """
    Generate signature for last.fm requests.
    """
    sorted_params = "".join(f"{k}{v}" for k, v in sorted(params.items()))
    signature = hashlib.md5((sorted_params + secret).encode("utf-8")).hexdigest()
    return signature


# --------------------------------------------------------------
# Thin last.fm-specific wrapper around HTTP.client's make_request
# --------------------------------------------------------------
def lastfm_request(
        method: str,
        *,
        authed: bool = False,
        user: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        timeout: int = 10,
) -> Dict[str, Any]:
    """
    Thin wrapper around the Last.fm REST endpoint.
    Raises
    ------
    LastFMError   for HTTP failures or Last.fm JSON errors
    """
    # ---------- query construction ------------------------------------------------
    q: dict[str, Any] = {
        "method": method,
        "format": "json",
    }
    # auth — either API key or `sk` (session key) for write-calls
    if authed:
        if not LASTFM_SESSION_KEY:
            raise LastFMError(-2, "No SESSION KEY in env", ENDPOINT)
        q["sk"] = LASTFM_SESSION_KEY
    else:
        if not LASTFM_API_KEY:
            raise LastFMError(-3, "No API KEY in env", ENDPOINT)
        q["api_key"] = LASTFM_API_KEY
    # optional parameters ----------------------------------------------------------
    if user is not None:
        q["user"] = user
    if page is not None:
        q["page"] = page
    if limit is not None:
        q["limit"] = limit
    # --- HTTP request ---------------------------------------------------------
    r = make_request(url=ENDPOINT,
                     params=q,
                     headers={"User-Agent": USER_AGENT},
                     max_retries=10)
    if r is None:  # network totally failed
        raise LastFMError(-1, "empty HTTP response", ENDPOINT)
    # Try JSON decoding no matter what the status code was
    json_body: Any | None = None
    try:
        json_body = r.json()
    except ValueError:
        pass  # not JSON → leave at None
    # Transport layer error?
    if r.status_code != 200:
        # Did Last.fm still give us a structured error?
        if isinstance(json_body, dict) and "error" in json_body:
            raise LastFMError(json_body["error"],
                              json_body.get("message", "No message"),
                              r.url)
        raise LastFMError(-1, f"HTTP {r.status_code}", r.url)
    # 200 OK but application-level error?
    if isinstance(json_body, dict) and "error" in json_body:
        raise LastFMError(json_body["error"],
                          json_body.get("message", "No message"),
                          r.url)
    return json_body  # type: ignore[return-value]


class LastFMError(RuntimeError):
    """
    Raised for any non-200 response from the Last.fm API.
    Attributes
    ----------
    code        Last.fm numeric error code (int)
    message     Last.fm error message (str)
    url         full request URL (str) – handy when debugging
    """

    def __init__(self, code: int, message: str, url: str):
        super().__init__(f"Last.fm API error {code}: {message}\n{url}")
        self.code = code
        self.message = message
        self.url = url
