"""Unit tests for corefunc.data_cleaning (Parquet-based artist dedup and encoding repair)."""
import pandas as pd
import pytest
from corefunc.data_cleaning import (
    _repair_text,
    clean_artist_info,
    fix_encoding,
)


# ── _repair_text unit tests ───────────────────────────────────────────────────
class TestRepairText:
    """Tests the _repair_text encoding-repair cascade."""

    def test_clean_passthrough(self):
        """Returns clean text unchanged."""
        assert _repair_text("Hello World") == "Hello World"

    def test_cp1252_smart_quote(self):
        """Repairs CP1252 right single quote (\\x92 → \u2019)."""
        assert _repair_text("Don\x92t Cry") == "Don\u2019t Cry"

    def test_cp1252_ellipsis(self):
        """Repairs CP1252 horizontal ellipsis (\\x85 → \u2026)."""
        assert _repair_text("In the Lap of the Gods\x85revisit") == "In the Lap of the Gods\u2026revisit"

    def test_utf8_mojibake_apostrophe(self):
        """Repairs double-decoded UTF-8 right single quote."""
        # UTF-8 bytes E2 80 99 decoded as Latin-1 → \u00e2\x80\x99
        assert _repair_text("Don\u00e2\x80\x99t") == "Don\u2019t"

    def test_latin2_mojibake(self):
        """Repairs UTF-8 decoded as Latin-2 (\u0102\\x81 → \u00c1)."""
        assert _repair_text("\u0102\x81kos") == "\u00c1kos"

    def test_shift_jis(self):
        """Repairs partially decoded Shift-JIS text."""
        # \x83V = Shift-JIS for \u30b7, \x83^ = \u30bf, \x81[ = \u30fc
        raw = "\x83V\x83^\x81["
        result = _repair_text(raw)
        assert "\x83" not in result
        assert "\x81" not in result

    def test_fallback_strips_unmappable(self):
        """Falls back to stripping C1 chars with no codec match."""
        # \x81 is undefined in CP1252 → stripped
        assert _repair_text("A\x81B") == "AB"

    def test_non_latin_untouched(self):
        """Does not corrupt valid non-Latin text."""
        text = "\u05d0\u05e8\u05d9\u05e7 \u05d0\u05d9\u05d9\u05e0\u05e9\u05d8\u05d9\u05d9\u05df"
        assert _repair_text(text) == text


# ── fix_encoding integration tests ────────────────────────────────────────────
class TestFixEncoding:
    """Tests the fix_encoding end-to-end on a temp Parquet file."""

    def test_empty_parquet(self, tmp_pq_dir):
        """Returns (0, 0) when scrobble.parquet does not exist."""
        fixed, total = fix_encoding()
        assert fixed == 0
        assert total == 0

    def test_no_bad_chars(self, tmp_pq_dir):
        """Returns (0, N) when all text is clean."""
        import helpers.io as io_mod
        df = pd.DataFrame({
            "artist_name": ["Artist A", "Artist B"],
            "album_title": ["Album A", "Album B"],
            "track_title": ["Track A", "Track B"],
            "artist_mbid": [None, None],
            "play_time": pd.date_range("2024-06-01", periods=2, freq="5min", tz="UTC"),
        })
        df.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        fixed, total = fix_encoding()
        assert fixed == 0
        assert total == 2

    def test_repairs_bad_chars(self, tmp_pq_dir):
        """Repairs rows with encoding corruption and writes back."""
        import helpers.io as io_mod
        df = pd.DataFrame({
            "artist_name": ["Good Artist", "Good Artist 2"],
            "album_title": ["Clean Album", "Clean Album 2"],
            "track_title": ["Don\x92t Cry", "Clean Track"],
            "artist_mbid": [None, None],
            "play_time": pd.date_range("2024-06-01", periods=2, freq="5min", tz="UTC"),
        })
        df.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        fixed, total = fix_encoding()
        assert fixed == 1
        assert total == 2
        # Verifying the repaired value
        result = pd.read_parquet(io_mod.SCROBBLE_PQ)
        assert result.loc[0, "track_title"] == "Don\u2019t Cry"


# ── clean_artist_info tests ───────────────────────────────────────────────────
class TestCleanArtistInfo:
    """Tests the clean_artist_info deduplication routine."""

    def test_empty_parquet(self, tmp_pq_dir):
        """Returns (0, 0) when artist_info.parquet does not exist."""
        removed, remaining = clean_artist_info()
        assert removed == 0
        assert remaining == 0

    def test_no_duplicates(self, tmp_pq_dir):
        """Keeps all rows when there are no duplicate artist_names."""
        import helpers.io as io_mod
        df = pd.DataFrame({
            "artist_name": ["A", "B"],
            "mbid": ["id-a", "id-b"],
            "country": ["DE", "US"],
            "disambiguation_comment": ["", ""],
            "aliases": ["", ""],
        })
        df.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        removed, remaining = clean_artist_info()
        assert removed == 0
        assert remaining == 2

    def test_dedup_keeps_most_complete(self, tmp_pq_dir):
        """Keeps the row with the highest completeness score per artist."""
        import helpers.io as io_mod
        df = pd.DataFrame({
            "artist_name": ["A", "A"],
            "mbid": [None, "mbid-1"],
            "country": [None, "DE"],
            "disambiguation_comment": [None, "rock band"],
            "aliases": [None, "alias"],
        })
        df.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        removed, remaining = clean_artist_info()
        assert removed == 1
        assert remaining == 1
        # Verifying the kept row has the complete data
        result = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert result.iloc[0]["mbid"] == "mbid-1"
