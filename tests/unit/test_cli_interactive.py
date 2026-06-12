"""
Unit tests for helpers.cli interactive and pure helper functions.
"""

import pandas as pd

from helpers.cli import _apply_canonical, _overlaps, _remember_artist_variant


class TestApplyCanonical:
    """Tests the in-place canonical name application."""

    def test_replaces_variants(self):
        """Replaces all variant names with the canonical name."""
        data = pd.DataFrame({"Artist": ["Beatles", "The Beatles", "Radiohead"]})
        artcounts = pd.DataFrame({"Artist": ["Beatles", "The Beatles", "Radiohead"], "Count": [10, 5, 20]})
        _apply_canonical("The Beatles", ["Beatles", "The Beatles"], data, artcounts)
        assert (data["Artist"] == "The Beatles").sum() == 2
        assert (artcounts["Artist"] == "The Beatles").sum() == 2


class TestOverlaps:
    """Tests the interval overlap checker."""

    def test_no_overlap(self):
        """Returns False for non-overlapping intervals."""
        df = pd.DataFrame(
            {
                "start_date": [pd.Timestamp("2020-01-01")],
                "end_date": [pd.Timestamp("2020-06-01")],
            }
        )
        assert _overlaps(df, pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")) is False

    def test_overlap_detected(self):
        """Returns True when intervals overlap."""
        df = pd.DataFrame(
            {
                "start_date": [pd.Timestamp("2020-01-01")],
                "end_date": [pd.Timestamp("2020-12-31")],
            }
        )
        assert _overlaps(df, pd.Timestamp("2020-06-01"), pd.Timestamp("2021-06-01")) is True

    def test_open_ended_overlap(self):
        """Returns True when existing interval has no end date."""
        df = pd.DataFrame(
            {
                "start_date": [pd.Timestamp("2020-01-01")],
                "end_date": [pd.NaT],
            }
        )
        assert _overlaps(df, pd.Timestamp("2022-01-01"), None) is True


class TestRememberArtistVariant:
    """Tests the variant-decision persistence."""

    def test_writes_to_avc(self, tmp_pq_dir):
        """Writes a decision row to avc.parquet."""
        import helpers.io as io_mod

        _remember_artist_variant("Beatles{The Beatles", "The Beatles", True, "obvious")
        df = pd.read_parquet(io_mod.AVC_PQ)
        assert len(df) == 1
        assert df.iloc[0]["canonical_name"] == "The Beatles"
        assert df.iloc[0]["to_link"] == True  # noqa: E712  — numpy bool
