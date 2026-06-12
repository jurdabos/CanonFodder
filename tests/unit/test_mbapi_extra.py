"""
Additional unit tests for HTTP.mbAPI — higher-level functions with mocking.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from HTTP.mbAPI import (
    _rate_limited,
    fetch_country,
    get_complete_artist_info,
    init,
    lookup_artist,
    lookup_mb_for,
    search_artist,
)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """Disables rate-limit sleeps and resets init._done between tests."""
    monkeypatch.setattr("HTTP.mbAPI.time.sleep", lambda _: None)
    monkeypatch.setattr("HTTP.mbAPI._last_call", 0.0)
    init._done = False  # type: ignore[attr-defined]
    yield
    init._done = False  # type: ignore[attr-defined]


class TestSearchArtist:
    """Tests the search_artist wrapper."""

    @patch("HTTP.mbAPI._mb_call")
    @patch("HTTP.mbAPI.init")
    def test_returns_results(self, mock_init, mock_call):
        """Returns a list of artist dicts."""
        mock_call.return_value = {
            "artist-list": [{"id": "abc", "name": "Bohren", "country": "DE"}],
        }
        results = search_artist("Bohren", limit=1)
        assert len(results) == 1
        assert results[0]["name"] == "Bohren"

    @patch("HTTP.mbAPI._mb_call", side_effect=Exception("API down"))
    @patch("HTTP.mbAPI.init")
    def test_returns_empty_on_error(self, mock_init, mock_call):
        """Returns empty list on API failure."""
        results = search_artist("Unknown")
        assert results == []


class TestLookupArtist:
    """Tests the lookup_artist MBID-based lookup."""

    @patch("HTTP.mbAPI._cache_artist")
    @patch("HTTP.mbAPI._mb_call")
    @patch("HTTP.mbAPI.init")
    def test_returns_artist_data(self, mock_init, mock_call, mock_cache):
        """Returns a dict with id, name, country, aliases, disambiguation."""
        mock_call.return_value = {
            "artist": {
                "id": "abc-123",
                "name": "Bohren",
                "country": "DE",
                "alias-list": [{"alias": "B&dCoG"}],
                "disambiguation": "doom jazz",
            },
        }
        result = lookup_artist("abc-123")
        assert result["name"] == "Bohren"
        assert result["country"] == "DE"
        assert "B&dCoG" in result["aliases"]
        mock_cache.assert_called_once()


class TestFetchCountry:
    """Tests the fetch_country convenience helper."""

    @patch("HTTP.mbAPI._cache_artist")
    @patch("HTTP.mbAPI.search_artist")
    @patch("HTTP.mbAPI.init")
    def test_returns_country(self, mock_init, mock_search, mock_cache):
        """Returns the country from the first search hit."""
        mock_search.return_value = [{"id": "abc", "name": "Bohren", "country": "DE"}]
        assert fetch_country("Bohren") == "DE"

    @patch("HTTP.mbAPI.search_artist", return_value=[])
    @patch("HTTP.mbAPI.init")
    def test_returns_none_when_no_hit(self, mock_init, mock_search):
        """Returns None when no search results."""
        assert fetch_country("Unknown") is None


class TestLookupMbFor:
    """Tests the name→MBID resolver."""

    @patch("HTTP.mbAPI.lookup_artist")
    @patch("HTTP.mbAPI.search_artist")
    def test_returns_mbid(self, mock_search, mock_lookup):
        """Returns the MBID from the first search hit."""
        mock_search.return_value = [{"id": "abc-123", "name": "Test"}]
        mock_lookup.return_value = {"id": "abc-123", "name": "Test"}
        result = lookup_mb_for("Test")
        assert result == "abc-123"

    @patch("HTTP.mbAPI.search_artist", return_value=[])
    def test_returns_none_when_not_found(self, mock_search):
        """Returns None when no artist is found."""
        assert lookup_mb_for("Nonexistent") is None


class TestGetCompleteArtistInfo:
    """Tests the high-level artist info getter."""

    @patch("HTTP.mbAPI.read_parquet")
    @patch("HTTP.mbAPI.lookup_artist")
    @patch("HTTP.mbAPI.init")
    def test_cache_hit(self, mock_init, mock_lookup, mock_read):
        """Returns cached data without calling remote API."""
        cached = pd.DataFrame(
            {
                "artist_name": ["Test"],
                "mbid": ["abc-123"],
                "country": ["DE"],
                "disambiguation_comment": ["rock"],
                "aliases": ["Alt1,Alt2"],
            }
        )
        mock_read.return_value = cached
        result = get_complete_artist_info("Test")
        assert result["name"] == "Test"
        assert result["country"] == "DE"
        mock_lookup.assert_not_called()

    @patch("HTTP.mbAPI.read_parquet", return_value=None)
    @patch("HTTP.mbAPI.lookup_mb_for", return_value=None)
    @patch("HTTP.mbAPI._cache_artist")
    @patch("HTTP.mbAPI.init")
    def test_no_match_returns_stub(self, mock_init, mock_cache, mock_lookup, mock_read):
        """Returns a stub dict when no MusicBrainz match exists."""
        result = get_complete_artist_info("Completely Unknown")
        assert result["id"] is None
        assert result["name"] == "Completely Unknown"

    @patch("HTTP.mbAPI.init")
    def test_no_identifier_returns_stub(self, mock_init):
        """Returns a stub dict when no identifier is provided."""
        result = get_complete_artist_info(None)
        assert result["id"] is None
        assert result["name"] == "Unknown Artist"


class TestRateLimited:
    """Tests the rate-limiting decorator."""

    def test_rate_limited_returns_result(self):
        """Decorated function returns its result."""

        @_rate_limited
        def dummy():
            return 42

        assert dummy() == 42
