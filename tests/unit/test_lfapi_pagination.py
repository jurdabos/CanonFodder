"""
Unit tests for HTTP.lfAPI pagination and higher-level fetch helpers.
"""
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytest
from HTTP.lfAPI import (
    _paginate,
    fetch_recent,
    fetch_recent_tracks_all_pages,
    get_recent_tracks_with_progress,
    sync_user_country,
)


class TestGetRecentTracksWithProgress:
    """Tests the progress-enabled track fetcher."""

    @patch("HTTP.lfAPI.time.sleep")
    @patch("HTTP.lfAPI.lastfm_request")
    def test_single_page(self, mock_req, mock_sleep):
        """Returns all tracks from a single-page response."""
        mock_req.return_value = {
            "recenttracks": {
                "track": [
                    {"artist": {"#text": "A", "mbid": ""}, "name": "T1",
                     "album": {"#text": "Al"}, "date": {"uts": "100"}},
                ],
                "@attr": {"totalPages": "1", "total": "1"},
            },
        }
        result = get_recent_tracks_with_progress("user1", limit=200)
        assert len(result) == 1
        assert result[0]["name"] == "T1"

    @patch("HTTP.lfAPI.time.sleep")
    @patch("HTTP.lfAPI.lastfm_request")
    def test_multi_page(self, mock_req, mock_sleep):
        """Fetches across multiple pages."""
        page1 = {
            "recenttracks": {
                "track": [{"name": "T1", "date": {"uts": "100"}}],
                "@attr": {"totalPages": "2", "total": "2"},
            },
        }
        page2 = {
            "recenttracks": {
                "track": [{"name": "T2", "date": {"uts": "200"}}],
                "@attr": {"totalPages": "2", "total": "2"},
            },
        }
        mock_req.side_effect = [page1, page2]
        result = get_recent_tracks_with_progress("user1", limit=1)
        assert len(result) == 2

    @patch("HTTP.lfAPI.time.sleep")
    @patch("HTTP.lfAPI.lastfm_request")
    def test_progress_callback_called(self, mock_req, mock_sleep):
        """Invokes the progress callback during fetching."""
        mock_req.return_value = {
            "recenttracks": {
                "track": [{"name": "T1"}],
                "@attr": {"totalPages": "1", "total": "1"},
            },
        }
        cb = MagicMock()
        get_recent_tracks_with_progress("user1", progress_callback=cb)
        assert cb.call_count >= 1


class TestFetchRecent:
    """Tests the simplified fetch_recent wrapper."""

    @patch("HTTP.lfAPI.get_recent_tracks_with_progress")
    @patch("HTTP.lfAPI.USERNAME", "testuser")
    def test_returns_dataframe(self, mock_fetch):
        """Returns a normalised DataFrame."""
        mock_fetch.return_value = [
            {"artist": {"#text": "A", "mbid": ""}, "album": {"#text": "Al"},
             "name": "T1", "date": {"uts": "100"}},
        ]
        df = fetch_recent(limit=10)
        assert isinstance(df, pd.DataFrame)
        assert "Artist" in df.columns
        assert len(df) == 1

    @patch("HTTP.lfAPI.USERNAME", None)
    def test_raises_without_username(self):
        """Raises ValueError when LASTFM_USER is missing."""
        with pytest.raises(ValueError, match="username"):
            fetch_recent()


class TestFetchRecentTracksAllPages:
    """Tests the all-pages fetcher."""

    @patch("HTTP.lfAPI.get_recent_tracks_with_progress")
    def test_returns_cleaned_df(self, mock_fetch):
        """Returns a DataFrame after _clean_track processing."""
        mock_fetch.return_value = [
            {"artist": {"#text": "A", "mbid": ""}, "album": {"#text": "Al"},
             "name": "T1", "date": {"uts": "100"}},
        ]
        df = fetch_recent_tracks_all_pages("user1")
        assert isinstance(df, pd.DataFrame)
        assert "artist_name" in df.columns

    @patch("HTTP.lfAPI.get_recent_tracks_with_progress", return_value=[])
    def test_returns_empty_on_no_tracks(self, mock_fetch):
        """Returns empty DataFrame when no tracks found."""
        df = fetch_recent_tracks_all_pages("user1")
        assert df.empty


class TestPaginate:
    """Tests the generic _paginate helper."""

    @patch("HTTP.lfAPI.time.sleep")
    @patch("HTTP.lfAPI.lastfm_request")
    def test_single_page_result(self, mock_req, mock_sleep):
        """Returns items from a single page."""
        mock_req.return_value = {
            "recenttracks": {
                "track": [{"name": "T1"}],
                "@attr": {"totalPages": "1"},
            },
        }
        result = _paginate("user.getRecentTracks", user="u1")
        assert len(result) == 1


class TestSyncUserCountry:
    """Tests the country sync flow."""

    @patch("HTTP.lfAPI._update_user_country", return_value=True)
    @patch("HTTP.lfAPI.iso2_for_en_name", return_value="HU")
    @patch("HTTP.lfAPI.lastfm_request")
    @patch("builtins.input", return_value="y")
    def test_syncs_country(self, mock_input, mock_req, mock_iso, mock_update):
        """Updates the country when user confirms."""
        mock_req.return_value = {"user": {"country": "Hungary"}}
        assert sync_user_country("testuser") is True
        mock_update.assert_called_once_with("HU")

    @patch("HTTP.lfAPI.iso2_for_en_name", return_value="HU")
    @patch("HTTP.lfAPI.lastfm_request")
    @patch("builtins.input", return_value="n")
    def test_skips_when_declined(self, mock_input, mock_req, mock_iso):
        """Returns False when user declines."""
        mock_req.return_value = {"user": {"country": "Hungary"}}
        assert sync_user_country("testuser") is False

    @patch("HTTP.lfAPI.lastfm_request")
    def test_raises_on_missing_country(self, mock_req):
        """Raises RuntimeError when Last.fm returns no country."""
        mock_req.return_value = {"user": {"country": ""}}
        with pytest.raises(RuntimeError, match="did not return"):
            sync_user_country("testuser")

    @patch("HTTP.lfAPI.iso2_for_en_name", return_value=None)
    @patch("HTTP.lfAPI.lastfm_request")
    def test_raises_on_unknown_country(self, mock_req, mock_iso):
        """Raises RuntimeError when the country cannot be mapped."""
        mock_req.return_value = {"user": {"country": "Narnia"}}
        with pytest.raises(RuntimeError, match="not found"):
            sync_user_country("testuser")
