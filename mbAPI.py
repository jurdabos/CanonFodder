from HTTP.client import make_request, USER_AGENT
from dotenv import load_dotenv
load_dotenv()
import musicbrainzngs
from pathlib import Path
import requests
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/artist/"
HERE = Path(__file__).resolve().parent

# We should inject a session/engine or perform a lazy import inside the helper that writes to the DB!

# --------------------------------------------------------------
# MusicBrainz country fetch
# --------------------------------------------------------------
def fetch_country(artist_name):
    """
    Fetch country information from MusicBrainz API for an artist.
    """
    try:
        resp = musicbrainz_request("artist", query=artist_name)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('artists'):
                return data['artists'][0].get('country', 'Unknown')
        return 'Unknown'
    except Exception as e:
        print(f"Error fetching country for {artist_name}: {e}")
        return 'Unknown'


# --------------------------------------------------------------
# Thin MB-specific wrapper around HTTP.client's make_request
# --------------------------------------------------------------
def musicbrainz_request(entity: str, **query):
    url = f"https://musicbrainz.org/ws/2/{entity}"
    params = {"fmt": "json", **query}
    return make_request(url, params=params,
                        headers={"User-Agent": "CanonFodder/1.0 (contact@example.com)"},
                        max_retries=10)