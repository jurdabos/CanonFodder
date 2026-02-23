"""
Unit tests for corefunc.data_cleaning (Parquet-based artist dedup).
"""
import pandas as pd
from corefunc.data_cleaning import clean_artist_info


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
