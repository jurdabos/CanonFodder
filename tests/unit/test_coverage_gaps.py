"""
Tests targeting specific uncovered lines across several modules.

Closes the gap between 79.56% and the 80% coverage target by exercising:
- helpers.io: migrate_scrobble_to_partitioned, dump_scrobble_df (empty),
  scrobble_duckdb_from (partitioned), ingest_scrobbles (existing partition)
- corefunc.enrich: backfill_mbids when info MBIDs are all empty
- corefunc.data_cleaning: _to_raw_bytes CP1252 reverse, _fix_df_encoding
  when column is absent
- helpers.stats: length_stats with non-string input
"""

import pandas as pd
import pytest


# ── helpers.io ────────────────────────────────────────────────────────────────
class TestMigrateScrobbleToPartitioned:
    """Tests migrate_scrobble_to_partitioned."""

    def test_no_legacy_file(self, tmp_pq_dir):
        """Returns 0 when legacy scrobble.parquet does not exist."""
        from helpers.io import migrate_scrobble_to_partitioned

        assert migrate_scrobble_to_partitioned() == 0

    def test_empty_legacy_file(self, tmp_pq_dir, sample_scrobble_df):
        """Returns 0 when legacy file exists but is empty."""
        from helpers.io import SCROBBLE_PQ, migrate_scrobble_to_partitioned

        empty = sample_scrobble_df.head(0)
        empty.to_parquet(SCROBBLE_PQ, index=False)
        assert migrate_scrobble_to_partitioned() == 0

    def test_migrates_data(self, tmp_pq_dir, sample_scrobble_df):
        """Migrates rows from legacy file to year-partitioned layout."""
        from helpers.io import (
            SCROBBLE_PQ,
            SCROBBLE_PQ_DIR,
            migrate_scrobble_to_partitioned,
            read_scrobble_df,
        )

        sample_scrobble_df.to_parquet(SCROBBLE_PQ, index=False)
        count = migrate_scrobble_to_partitioned()
        assert count == len(sample_scrobble_df)
        assert SCROBBLE_PQ_DIR.exists()
        df = read_scrobble_df()
        assert df is not None
        assert len(df) == len(sample_scrobble_df)


class TestDumpScrobbleDfEmpty:
    """Tests dump_scrobble_df with an empty DataFrame."""

    def test_empty_df_returns_early(self, tmp_pq_dir, sample_scrobble_df):
        """Returns the directory path without writing when df is empty."""
        from helpers.io import SCROBBLE_PQ_DIR, dump_scrobble_df

        result = dump_scrobble_df(sample_scrobble_df.head(0))
        assert result == SCROBBLE_PQ_DIR


class TestScrobbleDuckdbFromPartitioned:
    """Tests scrobble_duckdb_from when partitioned data exists."""

    def test_partitioned_source(self, tmp_pq_dir, sample_scrobble_df):
        """Returns a read_parquet(..., hive_partitioning=true) expression."""
        from helpers.io import ingest_scrobbles, scrobble_duckdb_from

        ingest_scrobbles(sample_scrobble_df)
        expr = scrobble_duckdb_from()
        assert "hive_partitioning" in expr
        assert "read_parquet" in expr


class TestIngestScrobblesExistingPartition:
    """Tests ingest_scrobbles when the year partition already exists."""

    def test_appends_to_existing_partition(self, tmp_pq_dir, sample_scrobble_df):
        """Appending to an existing year partition deduplicates correctly."""
        from helpers.io import ingest_scrobbles, read_scrobble_df

        # Ingesting the initial batch
        n1 = ingest_scrobbles(sample_scrobble_df)
        assert n1 == 3
        # Ingesting a second batch with one new row and two duplicates
        extra = pd.DataFrame(
            {
                "artist_name": ["New Artist"],
                "album_title": ["New Album"],
                "track_title": ["New Track"],
                "artist_mbid": [None],
                "play_time": pd.to_datetime(["2024-01-15 21:00:00"], utc=True),
            }
        )
        combined = pd.concat([sample_scrobble_df, extra], ignore_index=True)
        n2 = ingest_scrobbles(combined)
        assert n2 == 4  # all 4 normalised rows counted
        df = read_scrobble_df()
        assert df is not None
        assert len(df) == 4  # deduplication kept 3 original + 1 new


# ── corefunc.enrich ──────────────────────────────────────────────────────────
class TestBackfillMbidsAllEmpty:
    """Tests backfill_mbids when artist_info has only empty-string MBIDs."""

    def test_returns_zero_when_mbids_empty(self, tmp_pq_dir, sample_scrobble_df):
        """Returns 0 when artist_info exists but all mbid values are empty."""
        from helpers.io import ARTIST_INFO_PQ, ingest_scrobbles

        ingest_scrobbles(sample_scrobble_df)
        # Writing artist_info with empty mbid strings
        ai = pd.DataFrame(
            {
                "artist_name": ["Bohren & der Club of Gore", "Ry Cooder"],
                "mbid": ["", ""],
                "country": ["DE", "US"],
                "disambiguation_comment": ["", ""],
                "aliases": ["", ""],
            }
        )
        ai.to_parquet(ARTIST_INFO_PQ, index=False)
        from corefunc.enrich import backfill_mbids

        assert backfill_mbids() == 0


# ── corefunc.data_cleaning ───────────────────────────────────────────────────
class TestToRawBytesCP1252:
    """Tests _to_raw_bytes with CP1252 reverse-mapped characters."""

    def test_reverse_maps_smart_quote(self):
        """Maps a right single quotation mark (U+2019) back to 0x92."""
        from corefunc.data_cleaning import _to_raw_bytes

        result = _to_raw_bytes("\u2019")
        assert result == bytes([0x92])

    def test_raises_on_unmappable(self):
        """Raises ValueError for a character outside the Latin-1/CP1252 range."""
        from corefunc.data_cleaning import _to_raw_bytes

        with pytest.raises(ValueError, match="Cannot map"):
            _to_raw_bytes("\u4e16")  # CJK character


class TestFixDfEncodingMissingCol:
    """Tests _fix_df_encoding when a specified column is absent."""

    def test_skips_missing_column(self):
        """Returns 0 repaired rows when the column does not exist."""
        from corefunc.data_cleaning import _fix_df_encoding

        df = pd.DataFrame({"other": ["hello"]})
        result_df, count = _fix_df_encoding(df, ["nonexistent_col"])
        assert count == 0
        assert len(result_df) == 1


# ── helpers.stats ─────────────────────────────────────────────────────────────
class TestLengthStatsNonString:
    """Tests length_stats with non-string input."""

    def test_none_returns_zeros(self):
        """Returns zero-valued Series for None input."""
        from helpers.stats import length_stats

        result = length_stats(None)
        assert result["sig_len"] == 0
        assert result["n_variants"] == 0

    def test_int_returns_zeros(self):
        """Returns zero-valued Series for integer input."""
        from helpers.stats import length_stats

        result = length_stats(42)
        assert result["sig_len"] == 0
        assert result["n_variants"] == 0
