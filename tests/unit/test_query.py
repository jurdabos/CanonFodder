"""
Unit tests for helpers.query (DuckDB analytics layer).
"""
import pandas as pd
import pytest
from helpers.query import (
    artist_country_stats,
    artist_info_df,
    scrobble_count,
    scrobbles_between,
    top_artists,
    unique_artists,
)


class TestTopArtists:
    """Tests top_artists query."""

    def test_returns_top_n(self, populated_pq):
        """Returns a DataFrame with the top N artists by play count."""
        df = top_artists(n=5)
        assert not df.empty
        assert "artist_name" in df.columns
        assert "play_count" in df.columns
        # Bohren has 2 scrobbles, Ry Cooder has 1
        assert df.iloc[0]["artist_name"] == "Bohren & der Club of Gore"
        assert df.iloc[0]["play_count"] == 2


class TestScrobbleCount:
    """Tests scrobble_count query."""

    def test_counts_scrobbles(self, populated_pq):
        """Returns the total number of scrobbles."""
        assert scrobble_count() == 3


class TestUniqueArtists:
    """Tests unique_artists query."""

    def test_counts_unique_artists(self, populated_pq):
        """Returns the number of distinct artist names."""
        assert unique_artists() == 2


class TestArtistInfoDf:
    """Tests artist_info_df query."""

    def test_returns_artist_info(self, populated_pq):
        """Returns the full artist_info table."""
        df = artist_info_df()
        assert len(df) == 2
        assert "country" in df.columns

    def test_returns_empty_when_missing(self, tmp_pq_dir):
        """Returns empty DataFrame when artist_info.parquet is absent."""
        df = artist_info_df()
        assert df.empty


class TestScrobblesBetween:
    """Tests scrobbles_between date-range query."""

    def test_filters_by_date_range(self, populated_pq):
        """Returns only scrobbles within the given range."""
        df = scrobbles_between("2024-01-15T20:00:00+00:00", "2024-01-15T20:06:00+00:00")
        assert len(df) == 2  # 20:00 and 20:05 included, 20:10 excluded


class TestArtistCountryStats:
    """Tests artist_country_stats join query."""

    def test_returns_stats(self, populated_pq):
        """Returns country-level aggregation."""
        df = artist_country_stats()
        assert not df.empty
        assert set(df.columns) == {"country", "play_count", "artist_count"}

    def test_returns_empty_when_no_info(self, tmp_pq_dir, sample_scrobble_df):
        """Returns empty when artist_info is missing."""
        import helpers.io as io_mod
        sample_scrobble_df.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        df = artist_country_stats()
        assert df.empty
