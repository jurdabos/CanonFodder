"""
Additional unit tests for HTTP.lfAPI — higher-level functions with mocking.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from HTTP.lfAPI import (
    LastFMError,
    _fetch_country_from_lastfm,
    enrich_artist_mbids,
    fetch_scrobbles_since,
    lastfm_request,
)


class TestLastfmRequest:
    """Tests the lastfm_request wrapper."""

    @patch("HTTP.lfAPI.LASTFM_API_KEY", "fake-key")
    @patch("HTTP.lfAPI.make_request")
    def test_success(self, mock_req):
        """Returns JSON payload on 200 OK."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"user": {"name": "test"}}
        mock_req.return_value = mock_resp
        result = lastfm_request("user.getInfo", user="test")
        assert result["user"]["name"] == "test"

    @patch("HTTP.lfAPI.LASTFM_API_KEY", "fake-key")
    @patch("HTTP.lfAPI.make_request")
    def test_raises_on_none_response(self, mock_req):
        """Raises LastFMError when make_request returns None."""
        mock_req.return_value = None
        with pytest.raises(LastFMError, match="empty HTTP response"):
            lastfm_request("user.getInfo", user="test")

    @patch("HTTP.lfAPI.make_request")
    def test_raises_on_http_error(self, mock_req):
        """Raises LastFMError on non-200 status with JSON error body."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"error": 10, "message": "Forbidden"}
        mock_resp.url = "https://example.com"
        mock_req.return_value = mock_resp
        with pytest.raises(LastFMError):
            lastfm_request("user.getInfo", user="test")

    @patch("HTTP.lfAPI.make_request")
    def test_raises_on_api_error(self, mock_req):
        """Raises LastFMError on 200 with application-level error field."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": 6, "message": "User not found"}
        mock_resp.url = "https://example.com"
        mock_req.return_value = mock_resp
        with pytest.raises(LastFMError):
            lastfm_request("user.getInfo", user="nonexistent")


class TestFetchScrobblesSince:
    """Tests fetch_scrobbles_since with mocked HTTP."""

    @patch("HTTP.lfAPI.get_recent_tracks_with_progress")
    def test_returns_dataframe(self, mock_fetch):
        """Returns a DataFrame with expected columns."""
        mock_fetch.return_value = [
            {
                "artist": {"#text": "Bohren", "mbid": "abc"},
                "album": {"#text": "Sunset"},
                "name": "Prowler",
                "date": {"uts": "1700000000"},
            },
        ]
        df = fetch_scrobbles_since("testuser")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Artist" in df.columns

    @patch("HTTP.lfAPI.get_recent_tracks_with_progress")
    def test_skips_now_playing(self, mock_fetch):
        """Skips tracks without a 'date' key (now-playing)."""
        mock_fetch.return_value = [
            {"artist": {"#text": "A"}, "album": {"#text": "Al"}, "name": "T"},  # no date
        ]
        df = fetch_scrobbles_since("testuser")
        assert len(df) == 0

    @patch("HTTP.lfAPI.get_recent_tracks_with_progress")
    def test_since_parameter(self, mock_fetch):
        """Passes from_timestamp as since+1."""
        mock_fetch.return_value = []
        fetch_scrobbles_since("testuser", since=1000)
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs["from_timestamp"] == 1001


class TestEnrichArtistMbids:
    """Tests enrich_artist_mbids with mocked API calls."""

    @patch("HTTP.lfAPI.dump_scrobble_df")
    @patch("HTTP.lfAPI.lastfm_request")
    @patch("HTTP.lfAPI.read_scrobble_df")
    def test_no_scrobbles(self, mock_read, mock_req, mock_dump):
        """Returns success with 0 enriched when no scrobbles exist."""
        mock_read.return_value = None
        result = enrich_artist_mbids()
        assert result["status"] == "success"
        assert result["enriched"] == 0

    @patch("HTTP.lfAPI.time.sleep")
    @patch("HTTP.lfAPI.dump_scrobble_df")
    @patch("HTTP.lfAPI.lastfm_request")
    @patch("HTTP.lfAPI.read_scrobble_df")
    def test_enriches_missing_mbids(self, mock_read, mock_req, mock_dump, mock_sleep):
        """Enriches artists missing MBIDs via Last.fm API."""
        df = pd.DataFrame(
            {
                "artist_name": ["A", "B"],
                "album_title": ["Al1", "Al2"],
                "track_title": ["T1", "T2"],
                "artist_mbid": [None, "existing-mbid"],
                "play_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            }
        )
        mock_read.return_value = df
        mock_req.return_value = {"artist": {"name": "A", "mbid": "new-mbid"}}
        result = enrich_artist_mbids()
        assert result["status"] == "success"
        assert result["enriched"] == 1


class TestFetchCountryFromLastfm:
    """Tests the internal country-fetch helper."""

    @patch("HTTP.lfAPI.iso2_for_en_name", return_value="HU")
    @patch("HTTP.lfAPI.lastfm_request")
    def test_returns_country_code(self, mock_req, mock_iso):
        """Returns ISO-2 code for a valid country response."""
        mock_req.return_value = {"user": {"country": "Hungary"}}
        assert _fetch_country_from_lastfm("testuser") == "HU"

    @patch("HTTP.lfAPI.lastfm_request")
    def test_raises_when_no_country(self, mock_req):
        """Raises RuntimeError when Last.fm returns no country."""
        mock_req.return_value = {"user": {"country": ""}}
        with pytest.raises(RuntimeError, match="did not return"):
            _fetch_country_from_lastfm("testuser")
