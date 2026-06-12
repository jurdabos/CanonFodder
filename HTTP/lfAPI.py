from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from helpers.io import C_PQ, UC_PQ, dump_parquet, dump_scrobble_df, read_parquet, read_scrobble_df  # noqa: E402
from HTTP.client import USER_AGENT, make_request  # noqa: E402

log = logging.getLogger("lfAPI")
LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
HERE = Path(__file__).resolve().parent
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
LASTFM_SECRET = os.environ.get("LASTFM_SECRET")
USERNAME = os.getenv("LASTFM_USER")
PER_PAGE = 200


def _clean_track(rec: object) -> dict | None:
    """
    Return a normalised track-dict or None for malformed rows.
    """
    if not isinstance(rec, dict):  # guard against stray strings
        return None
    # ― timestamp ────────────────────────────────────────────────
    uts = int(rec.get("date", {}).get("uts", 0) or 0)  # 0 for "now-playing"
    # ― nested helpers ────────────────────────────────────────────
    artist = rec.get("artist", {}) or {}
    album = rec.get("album", {}) or {}
    return {
        "artist_name": artist.get("#text", ""),
        "artist_mbid": artist.get("mbid") or None,
        "album_title": album.get("#text", ""),
        "track_title": rec.get("name", ""),
        "uts": uts,
    }


def _fetch_country_from_lastfm(username: str) -> str:
    """
    Calls Last.fm and returns the ISO country code.
    Raises RuntimeError if the API call fails or the country cannot be mapped.
    """
    payload = lastfm_request("user.getInfo", user=username)
    country_name = payload["user"].get("country")
    if not country_name:
        raise RuntimeError("Last.fm did not return a country for that user.")
    code = iso2_for_en_name(country_name)
    if code is None:
        raise RuntimeError(f"Unknown country «{country_name}» – extend mapping.")
    return code


def _normalise_tracks(tracks: list[dict]) -> list[dict]:
    """
    Normalize a list of tracks from Last.fm API into a consistent format.
    Filters out tracks without timestamps (now playing).
    Parameters
    ----------
    tracks : list[dict]
        List of track dictionaries from Last.fm API
    Returns
    -------
    list[dict]
        List of normalized track dictionaries with consistent keys
    """
    normalized = []
    for track in tracks:
        # Skip "now playing" tracks (no timestamp)
        if "date" not in track:
            continue
        artist = (
            track.get("artist", {}) if isinstance(track.get("artist"), dict) else {"#text": track.get("artist", "")}
        )
        album = track.get("album", {}) if isinstance(track.get("album"), dict) else {"#text": track.get("album", "")}
        normalized.append(
            {
                "Artist": artist.get("#text", ""),
                "Song": track.get("name", ""),
                "Album": album.get("#text", ""),
                "uts": int(track.get("date", {}).get("uts", 0)),
                "artist_mbid": artist.get("mbid") if isinstance(artist, dict) else None,
            }
        )
    return normalized


# --------------------------------------------------------------
# Thin last.fm-specific wrapper around HTTP.client's make_request
# --------------------------------------------------------------
def _paginate(method: str, progress_callback: Optional[Callable[[int, int, str], None]] = None, **params) -> list[dict]:
    """
    Handle pagination for Last.fm API calls.
    Returns all pages of results combined into a single list.

    Parameters
    ----------
    method : str
        Last.fm API method name
    progress_callback : callable, optional
        Callback for progress updates (current, total, message)
    **params
        Any parameters to pass to the Last.fm API

    Returns
    -------
    list[dict]
        Combined list of items from all paginated responses

    Notes
    -----
    This is a helper function used by other functions in this module.
    It automatically handles pagination for Last.fm API calls and combines
    the results into a single list.
    """
    page = 1
    results = []
    total_pages = 1  # Will be updated after first request
    while page <= total_pages:
        # Update progress if callback provided
        if progress_callback:
            progress_callback(page, total_pages, f"Fetching page {page} of {total_pages}")
        params_with_page = params.copy()
        params_with_page["page"] = page
        # Use lastfm_request with progress_callback
        response = lastfm_request(method, page=page, progress_callback=progress_callback, **params)
        # Extracting the result key based on method
        result_key = method.split(".")[-1]
        if method == "user.getRecentTracks":
            items = response.get("recenttracks", {}).get("track", [])
            attr = response.get("recenttracks", {}).get("@attr", {})
        else:
            # For other methods, try to determine the key dynamically
            items_key = f"{result_key.lower()}s"  # e.g., "artists" for "getArtists"
            items = response.get(items_key, {}).get(result_key.lower(), [])
            attr = response.get(items_key, {}).get("@attr", {})
        # Adding items to our results
        if items:
            results.extend(items)
        # Updating total pages from response metadata
        total_pages = int(attr.get("totalPages", "1"))
        # If we're on the last page, or there are no more pages, break
        if page >= total_pages or not items:
            break
        page += 1
        # Small delay to avoid rate limiting
        time.sleep(0.2)
    # Final progress update if callback provided
    if progress_callback and total_pages > 0:
        progress_callback(total_pages, total_pages, f"Completed fetching {len(results)} items")
    return results


def _update_user_country(new_code: str) -> bool:
    """
    Persists the new country if it differs from today's entry in uc.parquet.
    Returns True when the file was changed, False otherwise.
    """
    today = str(date.today())
    df = read_parquet(UC_PQ)
    if df is not None and not df.empty:
        # Finding the current row (start ≤ today, end is null or > today)
        mask = (df["start_date"] <= today) & (df["end_date"].isna() | (df["end_date"] > today))
        current = df[mask].sort_values("start_date", ascending=False).head(1)
        if not current.empty and current.iloc[0]["country_code"] == new_code:
            return False
        # Closing old row
        if not current.empty:
            df.at[current.index[0], "end_date"] = today
    else:
        df = pd.DataFrame(columns=["country_code", "start_date", "end_date"])
    # Adding new row
    new_row = pd.DataFrame([{"country_code": new_code, "start_date": today, "end_date": None}])
    df = pd.concat([df, new_row], ignore_index=True)
    dump_parquet(df, UC_PQ)
    return True


def enrich_artist_mbids(progress_callback: Optional[Callable] = None) -> dict:
    """
    Fetches artist MBIDs from Last.fm API and updates scrobble.parquet.

    Looks up each artist by name via artist.getInfo — no username needed.

    Parameters
    ----------
    progress_callback : callable, optional
        Callback for progress updates (task_name, percentage, message)

    Returns
    -------
    dict
        Status information about the operation
    """
    try:
        if progress_callback:
            progress_callback("Initializing", 0, "Starting artist MBID enrichment")
        df = read_scrobble_df()
        if df is None or df.empty:
            return {"status": "success", "message": "No scrobbles found", "enriched": 0}
        # Finding artists without MBIDs
        missing_mask = df["artist_mbid"].isna() | (df["artist_mbid"] == "")
        artists = df.loc[missing_mask, "artist_name"].unique().tolist()
        if not artists:
            return {"status": "success", "message": "No artists missing MBIDs", "enriched": 0}
        if progress_callback:
            progress_callback("Analyzing", 5, f"Found {len(artists)} artists without MBIDs")
        # Fetching MBIDs from Last.fm in batches
        artist_mbid_map: dict[str, str] = {}
        batch_size = 50
        for i in range(0, len(artists), batch_size):
            batch = artists[i : i + batch_size]
            if progress_callback:
                pct = 5 + (i / len(artists)) * 45
                progress_callback("Fetching", pct, f"Getting MBIDs {i + 1}-{i + len(batch)} of {len(artists)}")
            for artist_name in batch:
                try:
                    data = lastfm_request("artist.getInfo", artist=artist_name)
                    if "artist" in data and data["artist"].get("mbid"):
                        artist_mbid_map[artist_name] = data["artist"]["mbid"]
                except (LastFMError, Exception) as exc:
                    log.warning(f"Error fetching MBID for '{artist_name}': {exc}")
                    continue
                time.sleep(0.2)
        # Applying MBIDs to the DataFrame and writing back
        if artist_mbid_map:
            if progress_callback:
                progress_callback("Updating", 70, f"Updating {len(artist_mbid_map)} artist MBIDs")
            for name, mbid in artist_mbid_map.items():
                mask = (df["artist_name"] == name) & (df["artist_mbid"].isna() | (df["artist_mbid"] == ""))
                df.loc[mask, "artist_mbid"] = mbid
            dump_scrobble_df(df)
        if progress_callback:
            progress_callback("Complete", 100, f"Enriched {len(artist_mbid_map)} artists with MBIDs")
        return {
            "status": "success",
            "message": f"Enriched {len(artist_mbid_map)} artists with MBIDs",
            "enriched": len(artist_mbid_map),
        }
    except Exception as exc:
        if progress_callback:
            progress_callback("Error", 100, f"Error: {exc}")
        return {"status": "error", "message": f"Error enriching artist MBIDs: {exc}"}


# --------------------------------------------------------------
# Misc data fetch
# --------------------------------------------------------------
def fetch_misc_data_from_lastfmapi(user: str | None = None) -> None:
    """
    One-screen "profile snapshot".
    Fetch 'top/friends/info/loved/recent' for *user* and print a compact digest.
    """
    if not user:
        user = input("Enter your Last.fm username: ").strip()
    # ── API calls ───────────────────────────────────────────────────────────────
    top_artists = lastfm_request("user.getTopArtists", user=user)
    top_albums = lastfm_request("user.getTopAlbums", user=user)
    top_tracks = lastfm_request("user.getTopTracks", user=user)
    infos = lastfm_request("user.getInfo", user=user)
    # ── Top artists ────────────────────────────────────────────────────────────
    if top_artists and "topartists" in top_artists:
        print("Top Artist")
        for artist in top_artists["topartists"]["artist"][:1]:
            print("   ", artist.get("name", "N/A"))
    # ── Top albums (title  +  "by ‹artist›") ───────────────────────────────────
    if top_albums and "topalbums" in top_albums:
        print("\nTop Album")
        for album in top_albums["topalbums"]["album"][:1]:
            print(f"   {album.get('name', 'N/A')}  by {album.get('artist', {}).get('name', '?')}")
    # ── Top tracks (title  +  "by ‹artist›") ───────────────────────────────────
    if top_tracks and "toptracks" in top_tracks:
        print("\nTop Track")
        for track in top_tracks["toptracks"]["track"][:1]:
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


# --------------------------------------------------------------
# Get pages of recent tracks from last.fm API until first DB hit
# --------------------------------------------------------------
def fetch_recent(
    limit: int = 1000, progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Return the user's most recent scrobbles as a DataFrame.
    Columns match the naming expected by `_prepare_scrobble_rows`.

    Parameters
    ----------
    limit : int, optional
        Maximum number of scrobbles to fetch, by default 1000
    progress_callback : callable, optional
        Callback for progress updates (current, total, message)

    Returns
    -------
    pd.DataFrame
        DataFrame with scrobble data
    """
    # Use USERNAME from environment if available
    username = USERNAME
    if not username:
        raise ValueError("No Last.fm username provided. Set LASTFM_USER environment variable.")

    # Use get_recent_tracks_with_progress which already handles pagination and progress updates
    tracks = get_recent_tracks_with_progress(
        username=username,
        limit=min(200, limit),  # Last.fm API limit per page is 200
        progress_callback=progress_callback,
    )

    # Normalize tracks and truncate to requested limit
    normalized_tracks = _normalise_tracks(tracks[:limit])

    return pd.DataFrame.from_records(normalized_tracks)


# --------------------------------------------------------------
# Get all pages of recent tracks from last.fm API to df
# --------------------------------------------------------------
def fetch_recent_tracks_all_pages(
    user: str, progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Return all scrobbles for *user* as a DataFrame.

    Parameters
    ----------
    user : str
        Last.fm username
    progress_callback : callable, optional
        Callback for progress updates (current, total, message)

    Returns
    -------
    pd.DataFrame
        DataFrame with scrobble data
    """
    # Use get_recent_tracks_with_progress which already handles pagination and progress updates
    tracks = get_recent_tracks_with_progress(
        username=user,
        limit=200,  # Last.fm API limit per page is 200
        progress_callback=progress_callback,
    )

    if not tracks:
        logging.warning("No tracks returned for user %s", user)
        return pd.DataFrame()

    # Applying _clean_track to each track
    batch = [_clean_track(t) for t in tracks]
    batch = [b for b in batch if b]  # to drop Nones

    return pd.DataFrame(batch)


def fetch_scrobbles_since(
    username: str, since: Optional[int] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Fetch every scrobble after `since` (Unix time, exclusive).
    When `since` is None we fetch the complete history.
    Returns a tidy DataFrame with the same column set that
    bulk_insert_scrobbles() expects.

    Parameters
    ----------
    username : str
        Last.fm username
    since : int, optional
        Unix timestamp to fetch scrobbles from (exclusive)
    progress_callback : callable, optional
        Callback for progress updates (current, total, message)

    Returns
    -------
    pd.DataFrame
        DataFrame with scrobble data
    """
    # Use get_recent_tracks_with_progress which already handles pagination and progress updates
    tracks = get_recent_tracks_with_progress(
        username=username,
        limit=PER_PAGE,
        from_timestamp=since + 1 if since is not None else None,
        progress_callback=progress_callback,
    )

    # Process the tracks into the expected format
    rows = []
    for t in tracks:
        # Skip "now-playing" pseudo-track (no date key)
        if "date" not in t:
            continue

        # Safely handle artist and album data which might not be dictionaries
        artist = t.get("artist", {}) if isinstance(t.get("artist"), dict) else {"#text": t.get("artist", "")}
        album = t.get("album", {}) if isinstance(t.get("album"), dict) else {"#text": t.get("album", "")}

        rows.append(
            {
                "Artist": artist.get("#text", ""),
                "Song": t.get("name", ""),
                "Album": album.get("#text", ""),
                "uts": int(t.get("date", {}).get("uts", 0)),
                "artist_mbid": artist.get("mbid") if isinstance(artist, dict) else None,
            }
        )

    # Create DataFrame with the expected columns
    df = pd.DataFrame(rows, columns=["Artist", "Song", "Album", "uts", "artist_mbid"])
    return df


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


def iso2_for_en_name(en_name: str) -> str | None:
    """
    Translates an English country name into ISO-3166 alpha-2 using c.parquet.

    The comparison is case-insensitive and falls back to a single-edit-distance
    fuzzy match.
    """
    df = read_parquet(C_PQ)
    if df is None or df.empty:
        return None
    # Exact (case-insensitive) match
    match = df.loc[df["en_name"].str.lower() == en_name.lower(), "ISO2"]
    if not match.empty:
        return str(match.iloc[0])

    # Fuzzy fallback (edit distance ≤ 1)
    def _very_close(a: str, b: str) -> bool:
        if abs(len(a) - len(b)) > 1:
            return False
        return sum(c1 != c2 for c1, c2 in zip(a.lower(), b.lower())) <= 1

    for _, row in df.iterrows():
        if _very_close(str(row["en_name"]), en_name):
            return str(row["ISO2"])
    return None


def lastfm_request(
    method: str,
    *,
    authed: bool = False,
    user: Optional[str] = None,
    page: Optional[int] = None,
    limit: Optional[int] = None,
    from_ts: Optional[int] = None,  # Added parameter to handle 'from' timestamp
    timeout: int = 10,  # Unused parameter kept for API consistency  # noqa: ARG001
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    **kwargs,  # Additional parameters
) -> Dict[str, Any]:
    """
    Thin wrapper around the Last.fm REST endpoint.
    Parameters
    ----------
    method : str
        Last.fm API method name
    authed : bool
        Whether to use authenticated call
    user : str, optional
        Last.fm username
    page : int, optional
        Page number for paginated results
    limit : int, optional
        Number of results per page
    from_ts : int, optional
        Unix timestamp to fetch scrobbles from (used instead of 'from' to avoid Python keyword)
    timeout : int
        Request timeout in seconds
    progress_callback : callable, optional
        Function that receives progress updates as (current, total, message)
        Used to report progress during long-running operations like pagination
    **kwargs
        Any additional parameters to pass to the Last.fm API
    Returns
    -------
    Dict[str, Any]
        Parsed JSON response from Last.fm API
    Raises
    ------
    LastFMError for HTTP failures or Last.fm JSON errors
    """
    # ---------- query construction ------------------------------------------------
    q: dict[str, Any] = {
        "method": method,
        "format": "json",
    }
    # auth — either API key or session key for write-calls
    if authed:
        # For authenticated calls, we need the session key from environment
        session_key = os.environ.get("LASTFM_SESSION_KEY")
        if not session_key:
            raise LastFMError(-2, "No LASTFM_SESSION_KEY in environment", LASTFM_API_URL)
        q["sk"] = session_key
    else:
        if not LASTFM_API_KEY:
            raise LastFMError(
                -3,
                "LASTFM_API_KEY not found in environment — set it in .env or use --source listenbrainz",
                LASTFM_API_URL,
            )
        q["api_key"] = LASTFM_API_KEY
    # optional parameters ----------------------------------------------------------
    if user is not None:
        q["user"] = user
    if page is not None:
        q["page"] = page
    if limit is not None:
        q["limit"] = limit
    # Handle the 'from' parameter specially (Python reserved keyword)
    if from_ts is not None:
        q["from"] = from_ts
    # Add any additional parameters
    q.update(kwargs)
    # --- HTTP request ---------------------------------------------------------
    # Report progress if callback provided
    if progress_callback and method == "user.getRecentTracks" and page is not None:
        # For scrobble fetching, report which page we're requesting
        msg = f"Requesting page {page}"
        if limit:
            msg += f" (limit: {limit})"
        if "from" in q:
            msg += f" from timestamp: {q['from']}"
        progress_callback(page, page, msg)

    r = make_request(url=LASTFM_API_URL, params=q, headers={"User-Agent": USER_AGENT}, max_retries=10)
    if r is None:  # network totally failed
        raise LastFMError(-1, "empty HTTP response", LASTFM_API_URL)
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
            raise LastFMError(json_body["error"], json_body.get("message", "No message"), r.url)
        raise LastFMError(-1, f"HTTP {r.status_code}", r.url)
    # 200 OK, but application-level error?
    if isinstance(json_body, dict) and "error" in json_body:
        raise LastFMError(json_body["error"], json_body.get("message", "No message"), r.url)
    return json_body  # type: ignore[return-value]


def get_recent_tracks_with_progress(
    username: str,
    limit: int = 200,
    from_timestamp: Optional[int] = None,
    to_timestamp: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Enhanced version of get_recent_tracks that provides detailed progress updates.
    Parameters
    ----------
    username : str
        Last.fm username
    limit : int
        Number of tracks per page (max 200)
    from_timestamp : int, optional
        Unix timestamp to fetch scrobbles from
    to_timestamp : int, optional
        Unix timestamp to fetch scrobbles to
    progress_callback : callable, optional
        Callback function to report progress (current, total, message)
    Returns
    -------
    List[Dict[str, Any]]
        List of scrobble records
    """
    # Initializing parameters for first request
    params = {"extended": 1}
    if from_timestamp:
        params["from"] = from_timestamp
    if to_timestamp:
        params["to"] = to_timestamp
    # First request to get total pages
    response = lastfm_request(
        "user.getRecentTracks", user=username, page=1, limit=limit, progress_callback=progress_callback, **params
    )
    # Extracting metadata
    metadata = response.get("recenttracks", {}).get("@attr", {})
    total_pages = int(metadata.get("totalPages", 0))
    total_tracks = int(metadata.get("total", 0))
    if progress_callback:
        progress_callback(1, total_pages, f"Fetching page 1 of {total_pages} ({total_tracks} tracks total)")
    # Storing all tracks in this list
    all_tracks = []
    # Extracting tracks from the first page
    tracks = response.get("recenttracks", {}).get("track", [])
    if not isinstance(tracks, list):
        tracks = [tracks]
    all_tracks.extend(tracks)
    # Fetching remaining pages
    for page in range(2, total_pages + 1):
        if progress_callback:
            progress_callback(
                page, total_pages, f"Fetching page {page} of {total_pages} ({len(all_tracks)}/{total_tracks} tracks)"
            )
        response = lastfm_request(
            "user.getRecentTracks", user=username, page=page, limit=limit, progress_callback=progress_callback, **params
        )
        tracks = response.get("recenttracks", {}).get("track", [])
        if not isinstance(tracks, list):
            tracks = [tracks]
        all_tracks.extend(tracks)
        # Introducing a small delay to avoid hitting rate limits
        time.sleep(0.2)
    if progress_callback:
        progress_callback(total_pages, total_pages, f"Completed fetching {len(all_tracks)} tracks")
    return all_tracks


class LastFMError(RuntimeError):
    """
    Raised for any non-200 response from the Last.fm API.
    Attributes
    ----------
    code: Last.fm numeric error code (int)
    message: Last.fm error message (str)
    url: full request URL (str) – handy when debugging
    """

    def __init__(self, code: int, message: str, url: str):
        super().__init__(f"Last.fm API error {code}: {message}\n{url}")
        self.code = code
        self.message = message
        self.url = url


def sync_user_country(lastfm_username: str, *, ask: bool = True) -> bool:
    """
    Fetches the country for *lastfm_username* via Last.fm, translates it
    to ISO-2, and updates uc.parquet with a timeline row.
    Returns True when the file was changed.
    """
    payload = lastfm_request("user.getInfo", user=lastfm_username)
    raw_country = payload["user"].get("country") or ""
    if not raw_country:
        raise RuntimeError("Last.fm did not return a country for that user.")
    code = iso2_for_en_name(raw_country)
    if code is None:
        raise RuntimeError(f"Country «{raw_country}» not found in reference table.")
    if ask:
        answer = input(f"Last.fm says your country is «{code}».  Save it? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return False
    return _update_user_country(code)
