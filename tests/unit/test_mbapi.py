"""
Unit tests for HTTP.mbAPI (MusicBrainz API helpers).
"""
from unittest.mock import patch
import pandas as pd
import pytest
from HTTP.mbAPI import _split_user_agent, _cache_artist, init


class TestSplitUserAgent:
    """Tests the User-Agent string parser."""

    def test_valid_ua(self):
        """Parses a standard UA string into (app, version, contact)."""
        app, ver, contact = _split_user_agent("CanonFodder/1.3 (balazs.torda@iu-study.org)")
        assert app == "CanonFodder"
        assert ver == "1.3"
        assert contact == "balazs.torda@iu-study.org"

    def test_invalid_ua_raises(self):
        """Raises ValueError for malformed UA strings."""
        with pytest.raises(ValueError, match="Invalid USER_AGENT"):
            _split_user_agent("just-a-string")


class TestCacheArtist:
    """Tests _cache_artist writing to artist_info.parquet."""

    def test_caches_basic_data(self, tmp_pq_dir):
        """Writes artist data to Parquet."""
        import helpers.io as io_mod
        data = {"name": "TestArtist", "id": "mbid-1", "country": "DE", "disambiguation": "rock"}
        _cache_artist(data)
        df = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert len(df) == 1
        assert df.iloc[0]["artist_name"] == "TestArtist"
        assert df.iloc[0]["country"] == "DE"

    def test_handles_alias_list(self, tmp_pq_dir):
        """Processes aliases from 'alias-list' format."""
        import helpers.io as io_mod
        data = {
            "name": "AliasArtist", "id": "mbid-2", "country": "",
            "disambiguation": "",
            "alias-list": [{"alias": "Alt1"}, {"alias": "Alt2"}],
        }
        _cache_artist(data)
        df = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert "Alt1" in df.iloc[0]["aliases"]

    def test_skips_when_no_name(self, tmp_pq_dir):
        """Does nothing when artist name is missing."""
        import helpers.io as io_mod
        _cache_artist({"id": "mbid-3"})
        assert not io_mod.ARTIST_INFO_PQ.exists()


class TestInit:
    """Tests the musicbrainzngs init wrapper."""

    @patch("HTTP.mbAPI.mb")
    def test_idempotent(self, mock_mb):
        """Calling init twice only configures musicbrainzngs once."""
        # Resetting the _done flag
        init._done = False  # type: ignore[attr-defined]
        init()
        init()
        mock_mb.set_useragent.assert_called_once()
        # Cleaning up
        init._done = False  # type: ignore[attr-defined]
