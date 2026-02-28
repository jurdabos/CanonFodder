"""
Tests for corefunc.canon.trainer — data preparation, feature engineering,
and feature pruning functions.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def tiny_avc_df():
    """Provides a minimal avc DataFrame with decided rows."""
    return pd.DataFrame({
        "artist_variants_hash": ["h1", "h2", "h3"],
        "artist_variants_text": ["Alpha{Alfa", "Beta{Betta", "Gamma{Delta"],
        "canonical_name": ["Alpha", "Beta", "Gamma"],
        "to_link": [True, True, False],
        "comment": ["", "", ""],
        "stamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
    })


@pytest.fixture()
def tiny_pairs_df():
    """Provides a small pairs DataFrame for feature computation."""
    return pd.DataFrame({
        "variant_a": ["Beatles", "Bohren & der Club of Gore", "Miles Davis"],
        "variant_b": ["The Beatles", "Bohren und der Club of Gore", "Mile Davis"],
        "to_link": [True, True, True],
    })


# ── build_training_data ───────────────────────────────────────────────────────
class TestBuildTrainingData:
    """Tests the pair-expansion and stratified split pipeline."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_avc_raises(self, mock_read):
        """Raises RuntimeError when avc.parquet is missing or empty."""
        mock_read.return_value = pd.DataFrame()
        from corefunc.canon.trainer import build_training_data
        with pytest.raises(RuntimeError, match="empty or missing"):
            build_training_data()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_splits_correctly(self, mock_read, tiny_avc_df):
        """Produces train and test splits with correct schema."""
        mock_read.return_value = tiny_avc_df
        from corefunc.canon.trainer import build_training_data
        train, test = build_training_data(test_size=0.50, random_state=42)
        assert "variant_a" in train.columns
        assert "variant_b" in train.columns
        assert "to_link" in train.columns
        assert len(train) + len(test) > 0


# ── Feature engineering ───────────────────────────────────────────────────────
class TestAddBaseFeatures:
    """Tests base + interaction feature computation."""

    def test_produces_expected_columns(self, tiny_pairs_df):
        """Adds ~53 feature columns (23 base + 30 interaction)."""
        from corefunc.canon.trainer import _add_base_features
        result = _add_base_features(tiny_pairs_df.copy())
        # Checking that core feature columns are present
        assert "ratio" in result.columns
        assert "partial_ratio" in result.columns
        assert "jaro_winkler" in result.columns
        assert "bigram_jaccard" in result.columns
        # Checking that interaction columns are added
        interaction_cols = [c for c in result.columns if "_minus_" in c or "_mul_" in c]
        assert len(interaction_cols) > 0

    def test_interaction_features_count(self, tiny_pairs_df):
        """Produces 30 interaction features from 6 similarity scores."""
        from corefunc.canon.trainer import _compute_interaction_features
        from helpers.features import compute_pair_features
        feats = [compute_pair_features("Beatles", "The Beatles")]
        feat_df = pd.DataFrame(feats)
        interactions = _compute_interaction_features(feat_df)
        assert len(interactions.columns) == 30


class TestPresenceFeatures:
    """Tests catalogue presence-and-quality features."""

    def test_both_empty(self):
        """Returns zeros when both album lists are empty."""
        from corefunc.canon.trainer import _presence_features
        feats = _presence_features([], [], "disco")
        assert feats["disco_best_match_score"] == 0.0
        assert feats["disco_any_exact_match"] == 0.0
        assert feats["disco_min_count"] == 0

    def test_exact_match(self):
        """Detects exact album matches."""
        from corefunc.canon.trainer import _presence_features
        feats = _presence_features(["Sunset Mission"], ["Sunset Mission"], "disco")
        assert feats["disco_any_exact_match"] == 1.0
        assert feats["disco_n_exact_matches"] == 1

    def test_fuzzy_match(self):
        """Detects fuzzy album matches above threshold."""
        from corefunc.canon.trainer import _presence_features
        feats = _presence_features(
            ["Sunset Mission"], ["Sunsett Mission"], "disco",
        )
        assert feats["disco_best_match_score"] > 80


class TestAddCatalogueFeatures:
    """Tests the full catalogue feature addition pipeline."""

    def test_adds_18_columns(self, tiny_pairs_df):
        """Adds 9 disco + 9 melo = 18 catalogue features."""
        from corefunc.canon.trainer import _add_catalogue_features
        albums = {"Beatles": ["Abbey Road"], "The Beatles": ["Abbey Road"]}
        tracks = {"Beatles": ["Come Together"], "The Beatles": ["Come Together"]}
        result = _add_catalogue_features(tiny_pairs_df.copy(), albums, tracks)
        disco_cols = [c for c in result.columns if c.startswith("disco_")]
        melo_cols = [c for c in result.columns if c.startswith("melo_")]
        assert len(disco_cols) == 9
        assert len(melo_cols) == 9


class TestComputeAllFeatures:
    """Tests the combined feature pipeline."""

    def test_without_catalogue(self, tiny_pairs_df):
        """Computes base + interaction when catalogue=False."""
        from corefunc.canon.trainer import compute_all_features
        result = compute_all_features(tiny_pairs_df.copy(), catalogue=False)
        assert "ratio" in result.columns
        assert all(c not in result.columns for c in ["disco_best_match_score"])

    def test_catalogue_requires_lookups(self, tiny_pairs_df):
        """Raises when catalogue=True but no lookups provided."""
        from corefunc.canon.trainer import compute_all_features
        with pytest.raises(ValueError, match="Catalogue lookups required"):
            compute_all_features(tiny_pairs_df.copy(), catalogue=True)


class TestPruneFeatureColumns:
    """Tests variance and correlation pruning."""

    def test_removes_zero_variance(self):
        """Drops columns with near-zero variance."""
        from corefunc.canon.trainer import prune_feature_columns
        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "good": rng.standard_normal(50),
            "const": np.ones(50),
            "also_good": rng.standard_normal(50),
        })
        surviving = prune_feature_columns(X)
        assert "const" not in surviving
        assert "good" in surviving


class TestCollectBestScores:
    """Tests the per-item fuzzy match score collector."""

    def test_empty_lists(self):
        """Returns empty list when either list is empty."""
        from corefunc.canon.trainer import _collect_best_scores
        assert _collect_best_scores([], ["a"]) == []
        assert _collect_best_scores(["a"], []) == []

    def test_perfect_match(self):
        """Returns 100 for identical items."""
        from corefunc.canon.trainer import _collect_best_scores
        scores = _collect_best_scores(["Abbey Road"], ["Abbey Road"])
        assert scores[0] == 100.0


# ── experiment_runner model catalogue ─────────────────────────────────────────
class TestBuildModelCatalogue:
    """Tests the model catalogue builder from experiment_runner."""

    def test_returns_all_models(self):
        """Returns 8 models: 5 base + 3 composites."""
        from corefunc.canon.experiment_runner import _build_model_catalogue
        catalogue = _build_model_catalogue(spw=1.0, device="cpu", random_state=42)
        assert "XGBoost" in catalogue
        assert "LightGBM" in catalogue
        assert "ExtraTrees" in catalogue
        assert "RandomForest" in catalogue
        assert "GradientBoosting" in catalogue
        assert "VotingEnsemble" in catalogue
        assert "StackingEnsemble" in catalogue
        assert "BaggingXGB" in catalogue
        assert len(catalogue) == 8


class TestSafeGetParams:
    """Tests _safe_get_params for extracting loggable params."""

    def test_filters_nested_objects(self):
        """Keeps only scalar params, drops nested objects."""
        from corefunc.canon.experiment_runner import _safe_get_params
        model = MagicMock()
        model.get_params.return_value = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "base_estimator": MagicMock(),
            "verbose": True,
            "name": "test",
        }
        safe = _safe_get_params(model)
        assert safe["n_estimators"] == 100
        assert safe["learning_rate"] == 0.05
        assert safe["verbose"] is True
        assert safe["name"] == "test"
        assert "base_estimator" not in safe

    def test_handles_exception(self):
        """Returns empty dict when get_params fails."""
        from corefunc.canon.experiment_runner import _safe_get_params
        model = MagicMock()
        model.get_params.side_effect = Exception("fail")
        assert _safe_get_params(model) == {}


class TestEvaluate:
    """Tests the _evaluate helper from experiment_runner."""

    def test_returns_metrics(self):
        """Returns precision, recall, f1, auc dict."""
        from corefunc.canon.experiment_runner import _evaluate
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 1, 0])
        model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2],
        ])
        y = np.array([0, 1, 1, 0])
        metrics = _evaluate(model, np.array([[1], [2], [3], [4]]), y)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert 0.0 <= metrics["auc"] <= 1.0
