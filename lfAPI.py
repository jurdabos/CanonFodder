from dotenv import load_dotenv
load_dotenv()
from DB import engine, SessionLocal
import hashlib
from HTTP.client import make_request, USER_AGENT
import logging
import os
import pandas as pd
from pathlib import Path

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
    if not isinstance(rec, dict):          # ← to guard against plain strings
        return None
    # rows for the *currently playing* track having no 'date'
    uts = int(rec.get("date", {}).get("uts", 0) or 0)
    return {
        "artist_name":   rec["artist"]["#text"],
        "album_title":   rec["album"]["#text"],
        "track_title":   rec["name"],
        "uts":           uts,
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
    top_artists = lastfm_request("user.gettopartists", user=user)
    top_albums = lastfm_request("user.gettopalbums", user=user)
    top_tracks = lastfm_request("user.gettoptracks", user=user)
    friends = lastfm_request("user.getfriends", user=user)
    infos = lastfm_request("user.getinfo", user=user)
    lovedtracks = lastfm_request("user.getlovedtracks", user=user)
    # ── Top artists ────────────────────────────────────────────────────────────
    if top_artists and "topartists" in top_artists:
        print("Top Artists:")
        for artist in top_artists["topartists"]["artist"][:3]:
            print("   ", artist.get("name", "N/A"))
    # ── Top albums (title  +  “by ‹artist›”) ───────────────────────────────────
    if top_albums and "topalbums" in top_albums:
        print("\nTop Albums:")
        for album in top_albums["topalbums"]["album"][:3]:
            print(f"   {album.get('name', 'N/A')}  by {album.get('artist', {}).get('name', '?')}")
    # ── Top tracks (title  +  “by ‹artist›”) ───────────────────────────────────
    if top_tracks and "toptracks" in top_tracks:
        print("\nTop Tracks:")
        for track in top_tracks["toptracks"]["track"][:3]:
            title = track.get("name", "N/A")
            artist = track.get("artist", {}).get("name", "N/A")
            print(f"   {title} by {artist}")
    # ── Friends ────────────────────────────────────────────────────────────────
    if friends and "friends" in friends:
        print("\n3 friends:")
        for friend in friends["friends"]["user"][:3]:
            print("   ", friend.get("name", "N/A"),
                  "=", friend.get("realname", "N/A"))
    # ── User info ──────────────────────────────────────────────────────────────
    if infos and "user" in infos:
        u = infos["user"]
        print("\nUser Info:")
        print("   Name:", u.get("name", "N/A"))
        print("   Country:", u.get("country", "N/A"))
        print("   Playcount:", u.get("playcount", "N/A"))
    # ── Loved tracks ───────────────────────────────────────────────────────────
    if lovedtracks and "lovedtracks" in lovedtracks:
        print("\n3 loved tracks:")
        for lt in lovedtracks["lovedtracks"]["track"][:3]:
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
        batch = [b for b in batch if b]           # to drop Nones
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
def lastfm_request(method: str,
                   *,
                   user: str | None = None,
                   page: int | None = None,
                   limit: int | None = None):
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    }
    if user:
        params["user"] = user
    if page:
        params["page"] = page
    if limit:
        params["limit"] = limit
    rsp = make_request(url, params=params,
                       headers={"User-Agent": USER_AGENT},
                       max_retries=10)
    if rsp is None:
        raise RuntimeError("Last.fm: empty HTTP response")
    try:
        data = rsp.json()
    except ValueError:
        raise RuntimeError("Last.fm: response is not JSON")
    # ── reject anything that is not a JSON *object* ─────────────────
    if not isinstance(data, dict):
        # log ‘data’ for diagnostics, then
        return None  # signal to the caller → stop
    if "error" in data:  # real API-level error
        raise RuntimeError(
            f"Last.fm API error {data['error']}: {data['message']}"
        )
    return data
