"""
Unit tests for corefunc.enrich (artist enrichment via MusicBrainz).
"""
from unittest.mock import patch
import pandas as pd
from corefunc.enrich import enrich_artist_country


class TestEnrichArtistCountry:
    """Tests enrich_artist_country with mocked external calls."""

    def test_no_scrobbles(self, tmp_pq_dir):
        """Returns 0 when scrobble.parquet does not exist."""
        assert enrich_artist_country() == 0

    def test_all_artists_known(self, tmp_pq_dir):
        """Returns 0 when every scrobbled artist is already in artist_info."""
        import helpers.io as io_mod
        scrobbles = pd.DataFrame({
            "artist_name": ["A"], "album_title": ["Al"], "track_title": ["T"],
            "artist_mbid": ["mbid-a"],
            "play_time": pd.to_datetime(["2024-01-01"], utc=True),
        })
        artist_info = pd.DataFrame({
            "artist_name": ["A"], "mbid": ["mbid-a"], "country": ["DE"],
            "disambiguation_comment": [""], "aliases": [""],
        })
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        artist_info.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        assert enrich_artist_country() == 0

    @patch("corefunc.enrich.time.sleep")
    @patch("HTTP.mbAPI._cache_artist")
    @patch("HTTP.mbAPI.search_artist")
    def test_enriches_unknown_artist(self, mock_search, mock_cache, mock_sleep, tmp_pq_dir):
        """Enriches an unknown artist via mocked MusicBrainz search."""
        import helpers.io as io_mod
        scrobbles = pd.DataFrame({
            "artist_name": ["NewBand"], "album_title": ["Debut"], "track_title": ["Song"],
            "artist_mbid": [None],
            "play_time": pd.to_datetime(["2024-06-01"], utc=True),
        })
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        mock_search.return_value = [{"id": "mb-123", "country": "GB", "disambiguation": "punk", "aliases": []}]
        result = enrich_artist_country()
        assert result == 1
        mock_search.assert_called_once()
        mock_cache.assert_called_once()
        mock_sleep.assert_called()

    @patch("corefunc.enrich.time.sleep")
    @patch("HTTP.mbAPI.search_artist")
    def test_artist_with_mbid_skips_search(self, mock_search, mock_sleep, tmp_pq_dir):
        """Skips MB search for artists that already have an MBID in scrobbles."""
        import helpers.io as io_mod
        scrobbles = pd.DataFrame({
            "artist_name": ["Known"], "album_title": ["Al"], "track_title": ["T"],
            "artist_mbid": ["existing-mbid"],
            "play_time": pd.to_datetime(["2024-06-01"], utc=True),
        })
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        result = enrich_artist_country()
        assert result == 1
        mock_search.assert_not_called()
