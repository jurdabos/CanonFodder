"""
Tests for corefunc.canon.trainer — utility, data-building, feature dispatch,
evaluation helpers, CV loop, GPU fallback, and MLflow integration.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def tiny_pairs_df():
    """Provides a small pairs DataFrame for feature computation."""
    return pd.DataFrame(
        {
            "variant_a": ["Beatles", "Bohren & der Club of Gore", "Miles Davis"],
            "variant_b": ["The Beatles", "Bohren und der Club of Gore", "Mile Davis"],
            "to_link": [True, True, True],
        }
    )


@pytest.fixture()
def binary_labels():
    """Provides ground-truth and probability arrays for threshold tests."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4, 0.85, 0.15])
    return y_true, y_prob


@pytest.fixture()
def mock_avc_df():
    """Provides a minimal decided AVC DataFrame with enough rows for splitting."""
    rows = []
    for i in range(40):
        link = i < 20
        rows.append(
            {
                "artist_variants_hash": f"h{i}",
                "artist_variants_text": f"Artist{i}{{Artst{i}",
                "canonical_name": f"Artist{i}",
                "to_link": link,
                "comment": "",
                "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def mock_gs_mb_df():
    """Provides a mock gs_mb DataFrame with pos/neg pairs."""
    rows = []
    for i in range(30):
        rows.append(
            {
                "variant_a": f"ArtistA{i}",
                "variant_b": f"ArtistB{i}",
                "to_link": i < 15,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def mock_dbscan_df():
    """Provides a mock gs_mb_dbscan DataFrame."""
    rows = []
    for i in range(30):
        rows.append(
            {
                "variant_a": f"DbArtist{i}",
                "variant_b": f"DbArtst{i}",
                "to_link": i < 10,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def mock_scrobble_df():
    """Provides a tiny scrobble DataFrame for catalogue lookup tests."""
    return pd.DataFrame(
        {
            "artist_name": ["Beatles", "Beatles", "Radiohead", "Radiohead"],
            "album_title": ["Abbey Road", "Let It Be", "OK Computer", "Kid A"],
            "track_title": ["Come Together", "Let It Be", "Paranoid Android", "Everything"],
            "artist_mbid": [
                "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d",
                "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d",
                "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                "a74b1b7f-71a5-4011-9441-d0b5e4122711",
            ],
            "play_time": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════
class TestParseDelimited:
    """Tests the {-delimited string parser."""

    def test_normal_string(self):
        """Splits a {-delimited string into parts."""
        from corefunc.canon.trainer import _parse_delimited

        assert _parse_delimited("Abbey Road{Let It Be") == ["Abbey Road", "Let It Be"]

    def test_empty_string(self):
        """Returns empty list for empty/null input."""
        from corefunc.canon.trainer import _parse_delimited

        assert _parse_delimited("") == []
        assert _parse_delimited(None) == []

    def test_nan_input(self):
        """Returns empty list for NaN."""
        from corefunc.canon.trainer import _parse_delimited

        assert _parse_delimited(float("nan")) == []

    def test_single_item(self):
        """Handles a single-item string without separators."""
        from corefunc.canon.trainer import _parse_delimited

        assert _parse_delimited("Solo Album") == ["Solo Album"]


class TestJaccardSet:
    """Tests the Jaccard similarity for sets."""

    def test_identical_sets(self):
        """Returns 1.0 for identical sets."""
        from corefunc.canon.trainer import _jaccard_set

        assert _jaccard_set({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """Returns 0.0 for disjoint sets."""
        from corefunc.canon.trainer import _jaccard_set

        assert _jaccard_set({"a"}, {"b"}) == 0.0

    def test_both_empty(self):
        """Returns 0.0 for two empty sets."""
        from corefunc.canon.trainer import _jaccard_set

        assert _jaccard_set(set(), set()) == 0.0

    def test_partial_overlap(self):
        """Computes correct Jaccard for partial overlap."""
        from corefunc.canon.trainer import _jaccard_set

        assert _jaccard_set({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)


class TestFuzzyOverlap:
    """Tests fuzzy overlap counting."""

    def test_empty_lists(self):
        """Returns (0, 0.0) when either list is empty."""
        from corefunc.canon.trainer import _fuzzy_overlap

        assert _fuzzy_overlap([], ["x"]) == (0, 0.0)
        assert _fuzzy_overlap(["x"], []) == (0, 0.0)

    def test_exact_matches(self):
        """Counts exact matches that exceed the threshold."""
        from corefunc.canon.trainer import _fuzzy_overlap

        n, ratio = _fuzzy_overlap(["Abbey Road"], ["Abbey Road"], threshold=80)
        assert n == 1
        assert ratio > 0.0

    def test_no_match(self):
        """Returns 0 when items are completely different."""
        from corefunc.canon.trainer import _fuzzy_overlap

        n, _ = _fuzzy_overlap(["ZZZZZ"], ["AAAAA"], threshold=80)
        assert n == 0


class TestProportionalDiscoFeatures:
    """Tests proportional discography feature computation."""

    def test_returns_5_features(self):
        """Produces exactly 5 disco features."""
        from corefunc.canon.trainer import _proportional_disco_features

        feats = _proportional_disco_features(["Abbey Road"], ["Abbey Road"])
        assert len(feats) == 5
        assert "disco_fuzzy_album_ratio" in feats
        assert "disco_has_fuzzy_album_match" in feats
        assert "disco_exact_album_jaccard" in feats

    def test_both_empty(self):
        """Returns zeros for empty album lists."""
        from corefunc.canon.trainer import _proportional_disco_features

        feats = _proportional_disco_features([], [])
        assert feats["disco_fuzzy_album_ratio"] == 0.0
        assert feats["disco_min_album_count"] == 0


class TestProportionalMeloFeatures:
    """Tests proportional melography feature computation."""

    def test_returns_5_features(self):
        """Produces exactly 5 melo features."""
        from corefunc.canon.trainer import _proportional_melo_features

        feats = _proportional_melo_features(["Come Together"], ["Come Together"])
        assert len(feats) == 5
        assert "melo_fuzzy_track_ratio" in feats
        assert "melo_exact_track_jaccard" in feats

    def test_both_empty(self):
        """Returns zeros for empty track lists."""
        from corefunc.canon.trainer import _proportional_melo_features

        feats = _proportional_melo_features([], [])
        assert feats["melo_min_track_count"] == 0


class TestAddProportionalCatalogueFeatures:
    """Tests adding proportional catalogue features to a DataFrame."""

    def test_adds_10_columns(self, tiny_pairs_df):
        """Adds 5 disco + 5 melo = 10 proportional catalogue features."""
        from corefunc.canon.trainer import _add_proportional_catalogue_features

        albums = {"Beatles": ["Abbey Road"], "The Beatles": ["Abbey Road"]}
        tracks = {"Beatles": ["Come Together"], "The Beatles": ["Come Together"]}
        result = _add_proportional_catalogue_features(tiny_pairs_df.copy(), albums, tracks)
        disco_cols = [c for c in result.columns if c.startswith("disco_")]
        melo_cols = [c for c in result.columns if c.startswith("melo_")]
        assert len(disco_cols) == 5
        assert len(melo_cols) == 5


class TestComputeCrossTierInteractions:
    """Tests cross-tier interaction feature computation."""

    def test_produces_interactions(self):
        """Generates cross-tier and non-WS interaction features."""
        from corefunc.canon.trainer import _compute_cross_tier_interactions
        from helpers.features import compute_pair_features

        base_feats = compute_pair_features("Beatles", "The Beatles")
        interactions = _compute_cross_tier_interactions(base_feats)
        assert len(interactions) > 0
        # 6 WS × 13 non-WS = 78 cross-tier + C(13,2) = 78 non-WS = 156 total
        assert len(interactions) == 78 + 78


class TestAddBaseFeaturesOnly:
    """Tests base-only feature computation (no interactions)."""

    def test_adds_base_no_interactions(self, tiny_pairs_df):
        """Adds ~23 base features without interaction columns."""
        from corefunc.canon.trainer import _add_base_features_only

        result = _add_base_features_only(tiny_pairs_df.copy())
        assert "ratio" in result.columns
        assert "jaro_winkler" in result.columns
        # Verifying no interaction columns
        interaction_cols = [c for c in result.columns if "_minus_" in c or "_mul_" in c]
        assert len(interaction_cols) == 0


class TestAddSeparatedFeatures:
    """Tests separated feature computation (Exp 8 design)."""

    def test_adds_base_and_cross_tier(self, tiny_pairs_df):
        """Adds base features plus cross-tier interaction features."""
        from corefunc.canon.trainer import _add_separated_features

        result = _add_separated_features(tiny_pairs_df.copy())
        assert "ratio" in result.columns
        # Verifying cross-tier interactions are present
        assert len(result.columns) > 30


class TestComputeFeaturesForSplit:
    """Tests the feature dispatch helper."""

    def test_base_features(self, tiny_pairs_df):
        """Dispatches to base-only when features='base'."""
        from corefunc.canon.trainer import _compute_features_for_split

        result = _compute_features_for_split(
            tiny_pairs_df.copy(),
            features="base",
            feature_strategy="standard",
            catalogue=False,
            cat_design="proportional",
            name_to_albums=None,
            name_to_tracks=None,
            group_features=False,
        )
        assert "ratio" in result.columns

    def test_full_features_no_catalogue(self, tiny_pairs_df):
        """Dispatches to base + interaction when features='full', catalogue=False."""
        from corefunc.canon.trainer import _compute_features_for_split

        result = _compute_features_for_split(
            tiny_pairs_df.copy(),
            features="full",
            feature_strategy="standard",
            catalogue=False,
            cat_design="proportional",
            name_to_albums=None,
            name_to_tracks=None,
            group_features=False,
        )
        interaction_cols = [c for c in result.columns if "_minus_" in c or "_mul_" in c]
        assert len(interaction_cols) > 0

    def test_separated_strategy(self, tiny_pairs_df):
        """Dispatches to separated features when feature_strategy='separated'."""
        from corefunc.canon.trainer import _compute_features_for_split

        result = _compute_features_for_split(
            tiny_pairs_df.copy(),
            features="full",
            feature_strategy="separated",
            catalogue=False,
            cat_design="proportional",
            name_to_albums=None,
            name_to_tracks=None,
            group_features=False,
        )
        assert len(result.columns) > 20

    def test_with_catalogue_proportional(self, tiny_pairs_df):
        """Adds proportional catalogue features when requested."""
        from corefunc.canon.trainer import _compute_features_for_split

        albums = {"Beatles": ["Abbey Road"], "The Beatles": ["Abbey Road"]}
        tracks = {"Beatles": ["Come Together"], "The Beatles": ["Come Together"]}
        result = _compute_features_for_split(
            tiny_pairs_df.copy(),
            features="full",
            feature_strategy="standard",
            catalogue=True,
            cat_design="proportional",
            name_to_albums=albums,
            name_to_tracks=tracks,
            group_features=False,
        )
        assert any(c.startswith("disco_") for c in result.columns)

    def test_with_catalogue_presence(self, tiny_pairs_df):
        """Adds presence catalogue features when cat_design='presence'."""
        from corefunc.canon.trainer import _compute_features_for_split

        albums = {"Beatles": ["Abbey Road"], "The Beatles": ["Abbey Road"]}
        tracks = {"Beatles": ["Come Together"], "The Beatles": ["Come Together"]}
        result = _compute_features_for_split(
            tiny_pairs_df.copy(),
            features="full",
            feature_strategy="standard",
            catalogue=True,
            cat_design="presence",
            name_to_albums=albums,
            name_to_tracks=tracks,
            group_features=False,
        )
        assert any(c.startswith("disco_") for c in result.columns)

    def test_with_group_features(self, tiny_pairs_df):
        """Adds length_stats columns when group_features=True."""
        from corefunc.canon.trainer import _compute_features_for_split

        result = _compute_features_for_split(
            tiny_pairs_df.copy(),
            features="base",
            feature_strategy="standard",
            catalogue=False,
            cat_design="proportional",
            name_to_albums=None,
            name_to_tracks=None,
            group_features=True,
        )
        assert "sig_len" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════
class TestOptimalThreshold:
    """Tests the F1-optimal threshold finder."""

    def test_returns_valid_threshold(self, binary_labels):
        """Returns (threshold, best_f1) with valid values."""
        from corefunc.canon.trainer import _optimal_threshold

        y_true, y_prob = binary_labels
        thr, best_f1 = _optimal_threshold(y_true, y_prob)
        assert 0.0 <= thr <= 1.0
        assert 0.0 <= best_f1 <= 1.0

    def test_perfect_predictions(self):
        """Returns high F1 for perfect predictions."""
        from corefunc.canon.trainer import _optimal_threshold

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        _, best_f1 = _optimal_threshold(y_true, y_prob)
        assert best_f1 > 0.9


class TestEvalAt:
    """Tests metric computation at a given threshold."""

    def test_at_half(self, binary_labels):
        """Returns valid metrics at threshold 0.5."""
        from corefunc.canon.trainer import _eval_at

        y_true, y_prob = binary_labels
        m = _eval_at(y_true, y_prob, 0.5)
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "threshold" in m
        assert m["threshold"] == 0.5

    def test_at_zero(self, binary_labels):
        """Predicts all positive at threshold 0."""
        from corefunc.canon.trainer import _eval_at

        y_true, y_prob = binary_labels
        m = _eval_at(y_true, y_prob, 0.0)
        assert m["recall"] == 1.0


class TestHighPrecisionThreshold:
    """Tests the high-precision threshold finder."""

    def test_returns_dict(self, binary_labels):
        """Returns a dict with threshold, precision, recall, f1."""
        from corefunc.canon.trainer import _high_precision_threshold

        y_true, y_prob = binary_labels
        result = _high_precision_threshold(y_true, y_prob, min_precision=0.80)
        assert "threshold" in result
        assert "precision" in result
        assert "f1" in result

    def test_returns_default_when_impossible(self):
        """Returns default (thr=0.99) when no threshold meets precision floor."""
        from corefunc.canon.trainer import _high_precision_threshold

        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        result = _high_precision_threshold(y_true, y_prob, min_precision=0.99)
        assert result["threshold"] == 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# CV evaluate and GPU fallback
# ═══════════════════════════════════════════════════════════════════════════════
class TestCvEvaluate:
    """Tests the stratified k-fold CV loop."""

    @patch("corefunc.canon.trainer.experiment")
    def test_returns_mean_metrics(self, mock_exp):
        """Returns cv_mean_* and cv_std_* keys."""
        from corefunc.canon.trainer import _cv_evaluate
        from sklearn.ensemble import RandomForestClassifier

        mock_exp.log_cv_fold = MagicMock()
        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame(
            {
                "f1": rng.standard_normal(n),
                "f2": rng.standard_normal(n),
            }
        )
        y = np.array([0] * 50 + [1] * 50)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        metrics = _cv_evaluate(
            clf,
            X,
            y,
            ["f1", "f2"],
            n_folds=3,
            random_state=42,
            model_name="test",
        )
        assert "cv_mean_precision" in metrics
        assert "cv_mean_recall" in metrics
        assert "cv_mean_f1" in metrics
        assert "cv_mean_auc" in metrics
        assert "cv_std_f1" in metrics


class TestFitWithGpuFallback:
    """Tests the GPU fallback wrapper."""

    def test_success_on_cpu(self):
        """Fits normally and returns same device on success."""
        from corefunc.canon.trainer import _fit_with_gpu_fallback
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression()),
            ]
        )
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        result_pipe, device = _fit_with_gpu_fallback(pipe, X, y, "cpu")
        assert device == "cpu"

    def test_raises_non_cuda_error(self):
        """Re-raises when device is not 'cuda'."""
        from corefunc.canon.trainer import _fit_with_gpu_fallback

        pipe = MagicMock()
        pipe.fit.side_effect = RuntimeError("test error")
        with pytest.raises(RuntimeError, match="test error"):
            _fit_with_gpu_fallback(pipe, None, None, "cpu")

    def test_cuda_fallback_to_cpu(self):
        """Falls back to CPU when CUDA training fails."""
        from corefunc.canon.trainer import _fit_with_gpu_fallback

        mock_clf = MagicMock()
        mock_clf.device = "cuda"
        pipe = MagicMock()
        pipe.named_steps = {"clf": mock_clf}
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA OOM")

        pipe.fit.side_effect = side_effect
        result_pipe, device = _fit_with_gpu_fallback(pipe, None, None, "cuda")
        assert device == "cpu"
        mock_clf.set_params.assert_called_with(device="cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# MLflow helpers
# ═══════════════════════════════════════════════════════════════════════════════
class TestNextExperimentNumber:
    """Tests the MLflow experiment number lookup."""

    @patch("mlflow.tracking.MlflowClient")
    def test_no_experiment(self, mock_client_cls):
        """Returns 16 when no experiment exists."""
        from corefunc.canon.trainer import _next_experiment_number

        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = None
        assert _next_experiment_number() == 16

    @patch("mlflow.tracking.MlflowClient")
    def test_with_existing_runs(self, mock_client_cls):
        """Returns max(existing) + 1."""
        from corefunc.canon.trainer import _next_experiment_number

        mock_client = mock_client_cls.return_value
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_run1 = MagicMock()
        mock_run1.data.params = {"experiment": "17"}
        mock_run2 = MagicMock()
        mock_run2.data.params = {"experiment": "20"}
        mock_client.search_runs.return_value = [mock_run1, mock_run2]
        assert _next_experiment_number() == 21

    @patch("mlflow.tracking.MlflowClient")
    def test_exception_returns_16(self, mock_client_cls):
        """Returns 16 on any exception."""
        from corefunc.canon.trainer import _next_experiment_number

        mock_client_cls.side_effect = Exception("connection error")
        assert _next_experiment_number() == 16


class TestVerifyMlflow:
    """Tests the MLflow verification function."""

    @patch("corefunc.canon.trainer.experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_param")
    def test_success(self, mock_log, mock_run, mock_exp):
        """Passes when MLflow is reachable."""
        from corefunc.canon.trainer import verify_mlflow

        mock_run.return_value.__enter__ = MagicMock()
        mock_run.return_value.__exit__ = MagicMock(return_value=False)
        verify_mlflow()
        mock_exp.init_experiment.assert_called_once()

    @patch("corefunc.canon.trainer.experiment")
    @patch("mlflow.start_run")
    def test_failure_raises(self, mock_run, mock_exp):
        """Raises RuntimeError when MLflow is unreachable."""
        from corefunc.canon.trainer import verify_mlflow

        mock_run.side_effect = Exception("connection refused")
        with pytest.raises(RuntimeError, match="verification failed"):
            verify_mlflow()


# ═══════════════════════════════════════════════════════════════════════════════
# Data builders
# ═══════════════════════════════════════════════════════════════════════════════
class TestBuildAvcFullTest:
    """Tests the AVC-full test set builder."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_avc_raises(self, mock_read):
        """Raises when avc.parquet is empty."""
        from corefunc.canon.trainer import _build_avc_full_test

        mock_read.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="empty or missing"):
            _build_avc_full_test()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_pairs(self, mock_read, mock_avc_df):
        """Expands AVC groups to pairs and returns expected columns."""
        mock_read.return_value = mock_avc_df
        from corefunc.canon.trainer import _build_avc_full_test

        result = _build_avc_full_test()
        assert "variant_a" in result.columns
        assert "variant_b" in result.columns
        assert "to_link" in result.columns


class TestBuildMbdbTrainingData:
    """Tests the MBDB training data builder."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_no_data_raises(self, mock_read):
        """Raises when no gs_mb parquet files are found."""
        from corefunc.canon.trainer import _build_mbdb_training_data

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="No gs_mb parquet"):
            _build_mbdb_training_data()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_deduplicated_pairs(self, mock_read, mock_gs_mb_df):
        """Returns deduplicated pairs with WRatio filter."""
        mock_read.return_value = mock_gs_mb_df
        from corefunc.canon.trainer import _build_mbdb_training_data

        result = _build_mbdb_training_data()
        assert "variant_a" in result.columns
        assert "to_link" in result.columns


class TestBuildDbscanTrainingData:
    """Tests the DBSCAN training data builder."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_raises(self, mock_read):
        """Raises when gs_mb_dbscan.parquet is missing."""
        from corefunc.canon.trainer import _build_dbscan_training_data

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="gs_mb_dbscan"):
            _build_dbscan_training_data()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_pairs(self, mock_read, mock_dbscan_df):
        """Returns WRatio-filtered pairs."""
        mock_read.return_value = mock_dbscan_df
        from corefunc.canon.trainer import _build_dbscan_training_data

        result = _build_dbscan_training_data()
        assert "variant_a" in result.columns
        assert "to_link" in result.columns


class TestBuildMixedTrainingData:
    """Tests the mixed AVC + MBDB training data builder."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_avc_raises(self, mock_read):
        """Raises when avc.parquet is missing."""
        from corefunc.canon.trainer import _build_mixed_training_data

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="empty or missing"):
            _build_mixed_training_data()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_train_test_split(self, mock_read, mock_avc_df, mock_gs_mb_df):
        """Returns (train, test) DataFrames."""
        call_count = [0]

        def side_effect(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_avc_df
            return mock_gs_mb_df

        mock_read.side_effect = side_effect
        from corefunc.canon.trainer import _build_mixed_training_data

        train, test = _build_mixed_training_data(test_size=0.3)
        assert len(train) + len(test) > 0


class TestBuildAvcGroupSplit:
    """Tests the AVC group-level split builder."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_avc_raises(self, mock_read):
        """Raises when avc.parquet is missing."""
        from corefunc.canon.trainer import _build_avc_group_split

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="empty or missing"):
            _build_avc_group_split()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_pair_splits(self, mock_read, mock_avc_df):
        """Returns (train_pairs, test_pairs) at the pair level."""
        mock_read.return_value = mock_avc_df
        from corefunc.canon.trainer import _build_avc_group_split

        train, test = _build_avc_group_split(test_size=0.3)
        assert "variant_a" in train.columns
        assert "to_link" in train.columns
        assert len(train) + len(test) > 0


class TestBuildMbdbMaxTrainingData:
    """Tests the MBDB-max training data builder."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_raises(self, mock_read):
        """Raises when gs_mb_max.parquet is missing."""
        from corefunc.canon.trainer import _build_mbdb_max_training_data

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="gs_mb_max"):
            _build_mbdb_max_training_data()


class TestDispatchDataBuild:
    """Tests the data-build routing function."""

    @patch("corefunc.canon.trainer.build_training_data")
    def test_avc_pair(self, mock_build):
        """Routes avc + pair to build_training_data."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_build.return_value = (pd.DataFrame(), pd.DataFrame())
        _dispatch_data_build(
            data_source="avc",
            split_strategy="pair",
            test_source="holdout",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_build.assert_called_once()

    @patch("corefunc.canon.trainer._build_avc_group_split")
    def test_avc_group(self, mock_build):
        """Routes avc + group to _build_avc_group_split."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_build.return_value = (pd.DataFrame(), pd.DataFrame())
        _dispatch_data_build(
            data_source="avc",
            split_strategy="group",
            test_source="holdout",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_build.assert_called_once()

    @patch("corefunc.canon.trainer._build_mixed_training_data")
    def test_mixed(self, mock_build):
        """Routes mixed to _build_mixed_training_data."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_build.return_value = (pd.DataFrame(), pd.DataFrame())
        _dispatch_data_build(
            data_source="mixed",
            split_strategy="pair",
            test_source="holdout",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_build.assert_called_once()

    @patch("corefunc.canon.trainer._build_mbdb_training_data")
    @patch("corefunc.canon.trainer._build_avc_full_test")
    def test_mbdb_cross_domain(self, mock_test, mock_train):
        """Routes mbdb + avc-full to cross-domain builders."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_train.return_value = pd.DataFrame()
        mock_test.return_value = pd.DataFrame()
        _dispatch_data_build(
            data_source="mbdb",
            split_strategy="pair",
            test_source="avc-full",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_train.assert_called_once()
        mock_test.assert_called_once()

    @patch("corefunc.canon.trainer._build_dbscan_training_data")
    @patch("corefunc.canon.trainer._build_avc_full_test")
    def test_dbscan(self, mock_test, mock_train):
        """Routes dbscan to _build_dbscan_training_data."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_train.return_value = pd.DataFrame()
        mock_test.return_value = pd.DataFrame()
        _dispatch_data_build(
            data_source="dbscan",
            split_strategy="pair",
            test_source="avc-full",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_train.assert_called_once()

    @patch("corefunc.canon.trainer._build_feature_sep_training_data")
    @patch("corefunc.canon.trainer._build_avc_full_test")
    def test_dbscan_distribution(self, mock_test, mock_train):
        """Routes dbscan + distribution matching to _build_feature_sep_training_data."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_train.return_value = pd.DataFrame()
        mock_test.return_value = pd.DataFrame()
        _dispatch_data_build(
            data_source="dbscan",
            split_strategy="pair",
            test_source="avc-full",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
            neg_matching="distribution",
        )
        mock_train.assert_called_once()

    @patch("corefunc.canon.trainer._build_mbdb_max_training_data")
    @patch("corefunc.canon.trainer._build_avc_full_test")
    def test_mbdb_max(self, mock_test, mock_train):
        """Routes mbdb-max to _build_mbdb_max_training_data."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_train.return_value = pd.DataFrame()
        mock_test.return_value = pd.DataFrame()
        _dispatch_data_build(
            data_source="mbdb-max",
            split_strategy="pair",
            test_source="avc-full",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_train.assert_called_once()

    @patch("corefunc.canon.trainer._build_dbscan_capped_training_data")
    @patch("corefunc.canon.trainer._build_avc_full_test")
    def test_dbscan_capped(self, mock_test, mock_train):
        """Routes dbscan-capped to _build_dbscan_capped_training_data."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_train.return_value = pd.DataFrame()
        mock_test.return_value = pd.DataFrame()
        _dispatch_data_build(
            data_source="dbscan-capped",
            split_strategy="pair",
            test_source="avc-full",
            test_size=0.2,
            wratio_lower=60,
            wratio_upper=100,
        )
        mock_train.assert_called_once()

    @patch("corefunc.canon.trainer._build_avc_full_test")
    def test_unknown_data_source_raises(self, mock_test):
        """Raises for unknown data_source."""
        from corefunc.canon.trainer import _dispatch_data_build

        mock_test.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="Unknown data_source"):
            _dispatch_data_build(
                data_source="imaginary",
                split_strategy="pair",
                test_source="avc-full",
                test_size=0.2,
                wratio_lower=60,
                wratio_upper=100,
            )

    def test_cross_domain_requires_avc_full(self):
        """Raises when cross-domain but test_source != 'avc-full'."""
        from corefunc.canon.trainer import _dispatch_data_build

        with pytest.raises(RuntimeError, match="avc-full"):
            _dispatch_data_build(
                data_source="mbdb",
                split_strategy="pair",
                test_source="holdout",
                test_size=0.2,
                wratio_lower=60,
                wratio_upper=100,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Catalogue lookups
# ═══════════════════════════════════════════════════════════════════════════════
class TestLoadScrobbleOnlyLookups:
    """Tests scrobble-only catalogue lookup builder."""

    @patch("corefunc.canon.trainer.read_scrobble_df")
    def test_empty_scrobbles(self, mock_read):
        """Returns empty dicts when no scrobbles."""
        from corefunc.canon.trainer import _load_scrobble_only_lookups

        mock_read.return_value = None
        albums, tracks = _load_scrobble_only_lookups()
        assert albums == {}
        assert tracks == {}

    @patch("corefunc.canon.trainer.read_scrobble_df")
    def test_builds_lookups(self, mock_read, mock_scrobble_df):
        """Builds name→albums and name→tracks dicts."""
        from corefunc.canon.trainer import _load_scrobble_only_lookups

        mock_read.return_value = mock_scrobble_df
        albums, tracks = _load_scrobble_only_lookups()
        assert "Beatles" in albums
        assert "Radiohead" in albums
        assert "Beatles" in tracks


class TestLoadCatalogueLookups:
    """Tests unified catalogue lookup builder."""

    @patch("corefunc.canon.trainer.read_scrobble_df")
    @patch("corefunc.canon.trainer.read_parquet")
    def test_empty_scrobbles(self, mock_read_pq, mock_read_scrobble):
        """Returns empty dicts when no scrobble data."""
        from corefunc.canon.trainer import _load_catalogue_lookups

        mock_read_scrobble.return_value = None
        albums, tracks = _load_catalogue_lookups()
        assert albums == {}
        assert tracks == {}

    @patch("corefunc.canon.trainer.SOLO_DISCO_PQ")
    @patch("corefunc.canon.trainer.read_scrobble_df")
    @patch("corefunc.canon.trainer.read_parquet")
    def test_with_scrobble_fallback(self, mock_read_pq, mock_read_scrobble, mock_disco_path, mock_scrobble_df):
        """Falls back to scrobble data when solo disco is missing."""
        from corefunc.canon.trainer import _load_catalogue_lookups

        mock_disco_path.exists.return_value = False
        mock_read_scrobble.return_value = mock_scrobble_df
        albums, tracks = _load_catalogue_lookups()
        assert "Beatles" in albums
        assert "Radiohead" in tracks

    @patch("corefunc.canon.trainer.SOLO_DISCO_PQ")
    @patch("corefunc.canon.trainer.read_scrobble_df")
    @patch("corefunc.canon.trainer.read_parquet")
    def test_with_mbdb_disco(self, mock_read_pq, mock_read_scrobble, mock_disco_path, mock_scrobble_df):
        """Uses MBDB disco when available and MBID matches."""
        from corefunc.canon.trainer import _load_catalogue_lookups

        mock_disco_path.exists.return_value = True
        disco_df = pd.DataFrame(
            {
                "mbid": ["b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"],
                "albums_str": ["White Album{Revolver"],
                "tracks_str": ["Helter Skelter{Tomorrow Never Knows"],
            }
        )
        mock_read_pq.return_value = disco_df
        mock_read_scrobble.return_value = mock_scrobble_df
        albums, tracks = _load_catalogue_lookups()
        # Beatles should use MBDB disco
        assert "White Album" in albums.get("Beatles", [])


# ═══════════════════════════════════════════════════════════════════════════════
# compute_all_features with catalogue designs
# ═══════════════════════════════════════════════════════════════════════════════
class TestComputeAllFeaturesWithCatalogue:
    """Tests compute_all_features with both catalogue designs."""

    def test_proportional_catalogue(self, tiny_pairs_df):
        """Adds proportional catalogue features with cat_design='proportional'."""
        from corefunc.canon.trainer import compute_all_features

        albums = {"Beatles": ["Abbey Road"], "The Beatles": ["Abbey Road"]}
        tracks = {"Beatles": ["Come Together"], "The Beatles": ["Come Together"]}
        result = compute_all_features(
            tiny_pairs_df.copy(),
            catalogue=True,
            cat_design="proportional",
            name_to_albums=albums,
            name_to_tracks=tracks,
        )
        assert any(c.startswith("disco_") for c in result.columns)

    def test_presence_catalogue(self, tiny_pairs_df):
        """Adds presence catalogue features with cat_design='presence'."""
        from corefunc.canon.trainer import compute_all_features

        albums = {"Beatles": ["Abbey Road"], "The Beatles": ["Abbey Road"]}
        tracks = {"Beatles": ["Come Together"], "The Beatles": ["Come Together"]}
        result = compute_all_features(
            tiny_pairs_df.copy(),
            catalogue=True,
            cat_design="presence",
            name_to_albums=albums,
            name_to_tracks=tracks,
        )
        disco_cols = [c for c in result.columns if c.startswith("disco_")]
        assert len(disco_cols) == 9
