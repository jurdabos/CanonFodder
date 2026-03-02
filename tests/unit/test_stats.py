"""
Unit tests for helpers.stats (pure statistical/ML utility functions).
"""

import numpy as np
import pandas as pd
import pytest
from helpers.stats import (
    cramers_v,
    drop_high_corr_features,
    iterative_correlation_dropper,
    length_stats,
    missing_value_ratio,
    show_cm_and_report,
    variance_testing,
    winsorization_outliers,
)


class TestCramersV:
    """Tests Cramér's V computation."""

    def test_perfect_association(self):
        """Returns 1.0 for perfectly associated variables (3×3 avoids Yates' correction)."""
        x = pd.Series(["a", "a", "b", "b", "c", "c"])
        y = pd.Series(["x", "x", "y", "y", "z", "z"])
        assert cramers_v(x, y) == pytest.approx(1.0)

    def test_no_association(self):
        """Returns a low value for unrelated variables."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.choice(["a", "b"], size=200))
        y = pd.Series(rng.choice(["x", "y"], size=200))
        assert cramers_v(x, y) < 0.2


class TestDropHighCorrFeatures:
    """Tests drop_high_corr_features identification logic."""

    def test_identifies_correlated_pair(self):
        """Detects a pair above the threshold and picks the lower-variance one to drop."""
        cm = pd.DataFrame(
            [[1.0, 0.95], [0.95, 1.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        var_table = pd.DataFrame({"features": ["A", "B"], "variances": [10.0, 5.0]})
        pairs, to_drop = drop_high_corr_features(cm, threshold=0.9, var_table=var_table)
        assert ("B", "A") in pairs or ("A", "B") in pairs
        assert "B" in to_drop

    def test_no_pair_below_threshold(self):
        """Returns empty lists when all correlations are below threshold."""
        cm = pd.DataFrame(
            [[1.0, 0.1], [0.1, 1.0]],
            columns=["A", "B"],
            index=["A", "B"],
        )
        var_table = pd.DataFrame({"features": ["A", "B"], "variances": [1.0, 1.0]})
        pairs, to_drop = drop_high_corr_features(cm, threshold=0.5, var_table=var_table)
        assert pairs == []
        assert to_drop == []


class TestIterativeCorrelationDropper:
    """Tests the iterative column-dropping routine."""

    def test_drops_correlated_columns(self):
        """Drops columns exceeding the cutoff."""
        rng = np.random.default_rng(7)
        base = rng.normal(size=100)
        df = pd.DataFrame(
            {
                "a": base,
                "b": base + rng.normal(0, 0.01, 100),  # nearly perfect corr with a
                "c": rng.normal(size=100),
            }
        )
        varframe = pd.DataFrame(
            {
                "features": ["a", "b", "c"],
                "variances": [df[c].var() for c in df.columns],
            }
        )
        result = iterative_correlation_dropper(df, cutoff=0.8, varframe=varframe, min_features=2)
        assert len(result.columns) == 2

    def test_stops_when_no_pairs_above_cutoff(self):
        """Keeps all columns when none exceed the cutoff."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=50),
                "y": rng.normal(size=50),
                "z": rng.normal(size=50),
            }
        )
        varframe = pd.DataFrame(
            {
                "features": ["x", "y", "z"],
                "variances": [df[c].var() for c in df.columns],
            }
        )
        result = iterative_correlation_dropper(df, cutoff=0.99, varframe=varframe, min_features=1)
        assert len(result.columns) == 3


class TestLengthStats:
    """Tests length-based feature extraction."""

    def test_single_variant(self):
        """Computes correct stats for a single name."""
        s = length_stats("Beatles")
        assert s["n_variants"] == 1
        assert s["sig_len"] == 7
        assert s["avg_name_len"] == 7.0

    def test_multiple_variants(self):
        """Computes correct stats for '{'-delimited variants."""
        s = length_stats("Beatles{The Beatles")
        assert s["n_variants"] == 2
        assert s["sig_len"] == 7 + len("The Beatles")


class TestMissingValueRatio:
    """Tests missing_value_ratio."""

    def test_all_present(self):
        """Returns 0 when no values are missing."""
        assert missing_value_ratio(pd.Series([1, 2, 3])) == 0.0

    def test_half_missing(self):
        """Returns 50 when half are NaN."""
        assert missing_value_ratio(pd.Series([1, None, 3, None])) == pytest.approx(50.0)

    def test_all_missing(self):
        """Returns 100 when all are NaN."""
        assert missing_value_ratio(pd.Series([None, None])) == pytest.approx(100.0)


class TestShowCmAndReport:
    """Tests show_cm_and_report prints without errors."""

    def test_prints_report(self, capsys):
        """Produces output containing expected labels."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        show_cm_and_report(y_true, y_pred, title="Test")
        captured = capsys.readouterr().out
        assert "Test" in captured
        assert "no link" in captured
        assert "link" in captured

    def test_no_title(self, capsys):
        """Omits title line when empty."""
        show_cm_and_report([0, 1], [0, 1])
        captured = capsys.readouterr().out
        assert "Actual" in captured


class TestVarianceTesting:
    """Tests variance_testing wrapper."""

    def test_selects_high_variance_features(self):
        """Drops a zero-variance column."""
        df = pd.DataFrame(
            {
                "high_var": [1, 2, 3, 4, 5],
                "zero_var": [1, 1, 1, 1, 1],
            }
        )
        var_df, selected = variance_testing(df, varthresh=0.01)
        assert "high_var" in selected.tolist()
        assert "zero_var" not in selected.tolist()
        assert len(var_df) == 2


class TestWinsorisationOutliers:
    """Tests winsorization_outliers detection."""

    def test_finds_outliers(self):
        """Detects extreme values beyond 1st/99th percentile."""
        data = list(range(1, 101)) + [999]
        outliers = winsorization_outliers(data)
        assert 999 in outliers

    def test_no_outliers(self):
        """Returns empty list when values are tightly clustered."""
        data = [5] * 100
        outliers = winsorization_outliers(data)
        assert outliers == []
