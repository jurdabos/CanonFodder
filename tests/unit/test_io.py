"""
Unit tests for helpers.io (Parquet I/O layer).
"""

import pandas as pd

from helpers.io import (
    append_to_parquet,
    dump_parquet,
    ingest_scrobbles,
    latest_scrobble_ts,
    normalise_scrobble_df,
    read_parquet,
)


class TestReadParquet:
    """Tests read_parquet behaviour."""

    def test_returns_none_when_missing(self, tmp_pq_dir):
        """Returns None when the file does not exist."""
        result = read_parquet(tmp_pq_dir / "nonexistent.parquet")
        assert result is None

    def test_reads_existing_file(self, tmp_pq_dir, sample_scrobble_df):
        """Reads an existing Parquet file correctly."""
        path = tmp_pq_dir / "scrobble.parquet"
        sample_scrobble_df.to_parquet(path, index=False)
        result = read_parquet(path)
        assert len(result) == len(sample_scrobble_df)
        assert list(result.columns) == list(sample_scrobble_df.columns)


class TestDumpParquet:
    """Tests dump_parquet behaviour."""

    def test_creates_file(self, tmp_pq_dir, sample_scrobble_df):
        """Creates a new Parquet file."""
        path = tmp_pq_dir / "out.parquet"
        dump_parquet(sample_scrobble_df, path)
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(sample_scrobble_df)

    def test_overwrites_existing(self, tmp_pq_dir, sample_scrobble_df):
        """Overwrites an existing file with new data."""
        path = tmp_pq_dir / "out.parquet"
        dump_parquet(sample_scrobble_df, path)
        # Writing a smaller DF
        small = sample_scrobble_df.head(1)
        dump_parquet(small, path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 1


class TestAppendToParquet:
    """Tests append_to_parquet with deduplication."""

    def test_creates_when_missing(self, tmp_pq_dir, sample_scrobble_df):
        """Creates the file when it does not exist yet."""
        path = tmp_pq_dir / "new.parquet"
        append_to_parquet(sample_scrobble_df, path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(sample_scrobble_df)

    def test_appends_new_rows(self, tmp_pq_dir, sample_scrobble_df):
        """Appends new, non-duplicate rows."""
        path = tmp_pq_dir / "app.parquet"
        sample_scrobble_df.head(2).to_parquet(path, index=False)
        # Appending the last row
        append_to_parquet(
            sample_scrobble_df.tail(1),
            path,
            dedup_cols=["artist_name", "track_title", "play_time"],
        )
        loaded = pd.read_parquet(path)
        assert len(loaded) == 3

    def test_dedup_keeps_last(self, tmp_pq_dir, sample_scrobble_df):
        """Deduplication keeps the last occurrence."""
        path = tmp_pq_dir / "dedup.parquet"
        sample_scrobble_df.to_parquet(path, index=False)
        # Appending same data — should still be 3 rows
        append_to_parquet(
            sample_scrobble_df,
            path,
            dedup_cols=["artist_name", "track_title", "play_time"],
        )
        loaded = pd.read_parquet(path)
        assert len(loaded) == 3

    def test_empty_df_noop(self, tmp_pq_dir):
        """Appending an empty DataFrame is a no-op."""
        path = tmp_pq_dir / "empty.parquet"
        append_to_parquet(pd.DataFrame(), path)
        assert not path.exists()


class TestNormaliseScrobbleDf:
    """Tests normalise_scrobble_df column aliasing and cleaning."""

    def test_renames_lastfm_columns(self):
        """Renames Last.fm API column names to canonical names."""
        raw = pd.DataFrame(
            {
                "Artist": ["Test"],
                "Album": ["A"],
                "Song": ["S"],
                "mbid": ["a4074512-87e0-4820-b609-0c4a18142a70"],
                "uts": [1705348800],
            }
        )
        result = normalise_scrobble_df(raw)
        assert list(result.columns) == [
            "artist_name",
            "album_title",
            "track_title",
            "artist_mbid",
            "play_time",
        ]
        assert result["artist_name"].iloc[0] == "Test"

    def test_cleans_invalid_mbid(self):
        """Replaces non-UUID artist_mbid with NaN."""
        raw = pd.DataFrame(
            {
                "artist_name": ["X"],
                "album_title": ["A"],
                "track_title": ["T"],
                "artist_mbid": ["not-a-uuid"],
                "uts": [1705348800],
            }
        )
        result = normalise_scrobble_df(raw)
        assert pd.isna(result["artist_mbid"].iloc[0])

    def test_deduplicates_rows(self):
        """Drops duplicate scrobbles based on artist+track+time."""
        raw = pd.DataFrame(
            {
                "artist_name": ["A", "A"],
                "album_title": ["Al", "Al"],
                "track_title": ["T", "T"],
                "artist_mbid": [None, None],
                "uts": [1705348800, 1705348800],
            }
        )
        result = normalise_scrobble_df(raw)
        assert len(result) == 1


class TestIngestScrobbles:
    """Tests the high-level ingest_scrobbles helper."""

    def test_ingests_and_returns_count(self, tmp_pq_dir, sample_scrobble_df):
        """Normalises and writes scrobbles to partitioned layout; returns row count."""
        from helpers.io import scrobble_data_exists

        n = ingest_scrobbles(sample_scrobble_df)
        assert n == 3
        assert scrobble_data_exists()


class TestLatestScrobbleTs:
    """Tests latest_scrobble_ts."""

    def test_returns_none_when_empty(self, tmp_pq_dir):
        """Returns None when no scrobble file exists."""
        assert latest_scrobble_ts() is None

    def test_returns_max_timestamp(self, tmp_pq_dir, sample_scrobble_df):
        """Returns the Unix timestamp of the newest scrobble."""
        import helpers.io as io_mod

        sample_scrobble_df.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        ts = latest_scrobble_ts()
        assert isinstance(ts, int)
        assert ts > 0
