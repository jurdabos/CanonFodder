"""
Unit tests for helpers.cluster (fuzzy scoring, clustering utilities).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from helpers.cluster import (
    anchors_ok,
    calculate_clustering_metrics,
    clf_proba,
    dbscan_with_anchors,
    expand_pairs,
    fuzzy_scores,
    tree_to_rule_list,
)


class TestFuzzyScores:
    """Tests the rapidfuzz similarity wrapper."""

    def test_identical_strings(self):
        """Returns all 1.0 for identical inputs."""
        scores = fuzzy_scores("Beatles", "Beatles")
        assert all(v == pytest.approx(1.0) for v in scores.values())

    def test_returns_six_keys(self):
        """Returns exactly the 6 expected scoring keys."""
        scores = fuzzy_scores("abc", "xyz")
        assert set(scores.keys()) == {
            "ratio",
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "WRatio",
            "QRatio",
        }

    def test_similar_strings_score_high(self):
        """Closely related strings score above 0.7."""
        scores = fuzzy_scores("Beatles", "The Beatles")
        assert scores["token_set_ratio"] > 0.7


class TestExpandPairs:
    """Tests pairwise expansion of variant rows."""

    def test_two_variants(self):
        """Expands two variants into one pair."""
        row = {"artist_variants": "A{B", "to_link": True}
        pairs = expand_pairs(row)
        assert len(pairs) == 1
        assert pairs[0][3] is True  # to_link flag preserved

    def test_three_variants(self):
        """Expands three variants into three pairs (C(3,2))."""
        row = {"artist_variants": "A{B{C", "to_link": False}
        pairs = expand_pairs(row)
        assert len(pairs) == 3

    def test_missing_key_raises(self):
        """Raises KeyError when the expected column is absent."""
        with pytest.raises(KeyError):
            expand_pairs({"wrong_col": "A{B", "to_link": True})


class TestAnchorsOk:
    """Tests anchor-set constraint checking."""

    def test_all_same_label(self):
        """Returns True when all anchor indices share the same non-noise label."""
        labels = np.array([0, 0, 1, 1])
        assert anchors_ok(labels, [[0, 1], [2, 3]]) is True

    def test_noise_label_fails(self):
        """Returns False when an anchor's first element is noise (-1)."""
        labels = np.array([-1, 0, 1, 1])
        assert anchors_ok(labels, [[0, 1]]) is False

    def test_mixed_labels_fail(self):
        """Returns False when anchors in the same set have different labels."""
        labels = np.array([0, 1, 1, 1])
        assert anchors_ok(labels, [[0, 1]]) is False


class TestCalculateClusteringMetrics:
    """Tests the clustering quality metrics calculator."""

    def test_basic_metrics(self):
        """Returns a dict with all expected keys."""
        data = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12], "y": [1, 2, 3, 10, 11, 12]})
        labels = np.array([0, 0, 0, 1, 1, 1])
        centers = np.array([[2, 2], [11, 11]])
        result = calculate_clustering_metrics("test", labels, data, cluster_centers=centers)
        assert result["Clustering Name"] == "test"
        assert result["Noise Percentage"] == 0.0
        assert not np.isnan(result["Silhouette Score"])
        assert not np.isnan(result["Weighted WSS"])

    def test_with_noise(self):
        """Reports correct noise percentage when noise labels are present."""
        data = pd.DataFrame({"x": [1, 2, 100, 10, 11], "y": [1, 2, 100, 10, 11]})
        labels = np.array([0, 0, -1, 1, 1])
        centers = np.array([[1.5, 1.5], [10.5, 10.5]])
        result = calculate_clustering_metrics("noisy", labels, data, cluster_centers=centers)
        assert result["Noise Percentage"] == pytest.approx(0.2)

    def test_no_model_bic_is_nan(self):
        """BIC is NaN when no model is provided."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        labels = np.array([0, 0, 0])
        result = calculate_clustering_metrics("no_bic", labels, data, cluster_centers=[[2, 2]])
        assert np.isnan(result["BIC"])


class TestDbscanWithAnchors:
    """Tests DBSCAN with anchor constraints."""

    def test_finds_eps(self):
        """Finds a valid eps that satisfies anchor constraints."""
        dist = np.array(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.9],
                [0.9, 0.9, 0.0],
            ]
        )
        eps, labels = dbscan_with_anchors(
            ["A", "B", "C"],
            dist,
            [[0, 1]],
            eps_range=np.arange(0.05, 1.0, 0.05),
        )
        assert labels[0] == labels[1]
        assert eps > 0

    def test_raises_when_impossible(self):
        """Raises RuntimeError when no eps satisfies all anchors."""
        dist = np.array(
            [
                [0.0, 0.9, 0.1],
                [0.9, 0.0, 0.9],
                [0.1, 0.9, 0.0],
            ]
        )
        with pytest.raises(RuntimeError, match="No ε"):
            dbscan_with_anchors(
                ["A", "B", "C"],
                dist,
                [[0, 1]],
                eps_range=np.arange(0.05, 0.5, 0.05),
            )


class TestClfProba:
    """Tests clf_proba with a simple mock classifier."""

    def test_returns_float(self):
        """Returns a probability float for a simple trained model."""
        from sklearn.linear_model import LogisticRegression

        X = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]] * 10)
        y = np.array([1, 0] * 10)
        clf = LogisticRegression().fit(X, y)
        prob = clf_proba("Beatles", "The Beatles", clf)
        assert 0.0 <= prob <= 1.0


class TestTreeToRuleList:
    """Tests decision tree rule extraction."""

    def test_extracts_rules(self):
        """Returns non-empty rule list from a fitted decision tree."""
        X = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]])
        y = np.array([0, 1, 0, 1])
        dt = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
        rules = tree_to_rule_list(dt, ["feat_a", "feat_b"], prob_threshold=0.5)
        assert isinstance(rules, list)
        # Should extract at least one class-1 rule from this simple tree
        assert len(rules) >= 1
        for condition, prob in rules:
            assert isinstance(condition, str)
            assert prob >= 0.5
