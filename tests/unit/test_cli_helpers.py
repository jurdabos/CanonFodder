"""
Unit tests for helpers.cli (pure utility functions) and additional CLI/query tests.
"""
import pandas as pd
import pytest
from click.testing import CliRunner
from helpers.cli import (
    _interval_ok,
    _parse_date,
    _split_variants,
    make_signature,
    make_signature_hash,
)
from helpers.query import avc_df
from main import cli


class TestSplitVariants:
    """Tests the variant-string splitter."""

    def test_single(self):
        """Returns one element for a string without separator."""
        assert _split_variants("Beatles") == ["Beatles"]

    def test_multiple(self):
        """Splits on the '{' separator."""
        result = _split_variants("Beatles{The Beatles")
        assert result == ["Beatles", "The Beatles"]

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace from each variant."""
        result = _split_variants(" A { B ")
        assert result == ["A", "B"]


class TestMakeSignature:
    """Tests canonical signature construction."""

    def test_sorts_and_joins(self):
        """Produces a sorted, '{'-joined string."""
        sig = make_signature(["Beatles", "The Beatles"])
        assert sig == "Beatles{The Beatles"

    def test_strips_whitespace(self):
        """Ignores empty strings and strips whitespace."""
        sig = make_signature(["  B  ", "", "A"])
        assert sig == "A{B"


class TestMakeSignatureHash:
    """Tests SHA-256 hash of signature."""

    def test_deterministic(self):
        """Produces the same hash for the same input."""
        assert make_signature_hash("abc") == make_signature_hash("abc")
        assert len(make_signature_hash("abc")) == 64


class TestParseDate:
    """Tests the date parser."""

    def test_valid_date(self):
        """Parses a YYYY-MM-DD string."""
        result = _parse_date("2024-06-15")
        assert result is not None
        assert result.day == 15

    def test_empty_returns_none(self):
        """Returns None for an empty string."""
        assert _parse_date("") is None
        assert _parse_date("   ") is None

    def test_invalid_raises(self):
        """Raises ValueError for unparseable dates."""
        with pytest.raises(ValueError, match="not a valid date"):
            _parse_date("not-a-date")


class TestIntervalOk:
    """Tests the interval validator."""

    def test_valid_interval(self):
        """Accepts a valid start < end pair."""
        _interval_ok(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))

    def test_none_start_raises(self):
        """Raises ValueError when start is None."""
        with pytest.raises(ValueError, match="Start date"):
            _interval_ok(None, pd.Timestamp("2024-12-31"))

    def test_end_before_start_raises(self):
        """Raises ValueError when end < start."""
        with pytest.raises(ValueError, match="End date"):
            _interval_ok(pd.Timestamp("2024-12-31"), pd.Timestamp("2024-01-01"))

    def test_none_end_ok(self):
        """Accepts None as end (open-ended interval)."""
        _interval_ok(pd.Timestamp("2024-01-01"), None)


class TestAvcDf:
    """Tests the query.avc_df function."""

    def test_returns_empty_when_missing(self, tmp_pq_dir):
        """Returns an empty DataFrame with expected columns when no file exists."""
        df = avc_df()
        assert df.empty
        assert "artist_variants_hash" in df.columns

    def test_reads_existing(self, tmp_pq_dir):
        """Reads an existing avc.parquet file."""
        import helpers.io as io_mod
        data = pd.DataFrame({
            "artist_variants_hash": ["h1"],
            "artist_variants_text": ["A{B"],
            "canonical_name": ["A"],
            "to_link": [True],
            "comment": [""],
            "stamp": pd.to_datetime(["2024-01-01"], utc=True),
        })
        data.to_parquet(io_mod.AVC_PQ, index=False)
        df = avc_df()
        assert len(df) == 1
