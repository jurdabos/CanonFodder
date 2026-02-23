"""
Unit tests for helpers.io edge cases not covered by test_io.py.
"""
from collections import Counter
import pandas as pd
import pytest
from helpers.io import normalise_scrobble_df, register_custom_palette, sanitize


class TestNormaliseScrobbleDfEdgeCases:
    """Tests edge cases in normalise_scrobble_df."""

    def test_timestamp_column(self):
        """Handles a 'Timestamp' column instead of 'uts'."""
        raw = pd.DataFrame({
            "artist_name": ["X"],
            "album_title": ["A"],
            "track_title": ["T"],
            "Timestamp": pd.to_datetime(["2024-06-01 12:00:00"], utc=True),
        })
        result = normalise_scrobble_df(raw)
        assert "play_time" in result.columns
        assert pd.notna(result["play_time"].iloc[0])

    def test_missing_artist_mbid_column(self):
        """Adds artist_mbid as None when column is absent."""
        raw = pd.DataFrame({
            "artist_name": ["X"],
            "album_title": ["A"],
            "track_title": ["T"],
            "uts": [1705348800],
        })
        result = normalise_scrobble_df(raw)
        assert "artist_mbid" in result.columns

    def test_missing_columns_filled(self):
        """Fills missing optional columns with None."""
        raw = pd.DataFrame({
            "Artist": ["X"],
            "uts": [1705348800],
        })
        result = normalise_scrobble_df(raw)
        assert "album_title" in result.columns
        assert "track_title" in result.columns


class TestSanitize:
    """Tests the column-name sanitiser."""

    def test_replaces_operators(self):
        """Replaces arithmetic operators with safe tokens."""
        seen = Counter()
        assert sanitize("partial_ratio - QRatio", seen) == "partial_ratio_minus_QRatio"

    def test_dedup_suffix(self):
        """Adds a numeric suffix on collision."""
        seen = Counter()
        first = sanitize("col_a", seen)
        second = sanitize("col_a", seen)
        assert first == "col_a"
        assert second == "col_a_1"

    def test_strips_special_chars(self):
        """Replaces non-word characters with underscores."""
        seen = Counter()
        assert sanitize("hello world!", seen) == "hello_world"


class TestRegisterCustomPalette:
    """Tests the Seaborn palette registration helper."""

    def test_not_found_raises(self):
        """Raises ValueError when the palette name doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            register_custom_palette("nonexistent", [])

    def test_registers_palette(self):
        """Registers a known palette and returns hex colours."""
        palettes = [{
            "paletteName": "test_pal",
            "colors": [
                {"hex": "FF0000", "position": 0},
                {"hex": "#00FF00", "position": 1},
            ],
        }]
        colours = register_custom_palette("test_pal", palettes)
        assert len(colours) == 2
        assert colours[0] == "#FF0000"
        assert colours[1] == "#00FF00"
