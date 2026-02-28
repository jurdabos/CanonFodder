"""
Tests for QA functions, experiment runner helpers, tuner search spaces,
and helpers/schema.py.
"""
import numpy as np
import json
import pandas as pd
from unittest.mock import patch, MagicMock

_SAMPLE_FEATS = json.dumps({"wratio": 0.85, "partial_ratio": 0.9, "token_sort": 0.8})


# ═══════════════════════════════════════════════════════════════════════════════
# corefunc/qa.py — qa_avc, qa_gs_mb, qa_uc, qa_predictions
# ═══════════════════════════════════════════════════════════════════════════════
class TestQaAvc:
    """Tests qa_avc function."""

    def test_qa_avc_missing(self, tmp_pq_dir):
        """Returns skipped when avc.parquet is missing."""
        from corefunc.qa import qa_avc
        report = qa_avc()
        assert report["status"] == "skipped"

    def test_qa_avc_valid(self, tmp_pq_dir):
        """Returns pass for valid avc data."""
        from helpers.io import dump_parquet, AVC_PQ
        df = pd.DataFrame({
            "artist_variants_hash": ["h1", "h2"],
            "artist_variants_text": ["Alpha{Alfa", "Beta{Beto"],
            "canonical_name": ["Alpha", "Beta"],
            "to_link": pd.array([True, False], dtype="boolean"),
            "comment": ["", ""],
            "stamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        })
        dump_parquet(df, AVC_PQ)
        from corefunc.qa import qa_avc
        report = qa_avc()
        assert report["passed"] is True
        assert report["row_count"] == 2
        assert report["duplicates"]["duplicate_count"] == 0


class TestQaGsMb:
    """Tests qa_gs_mb function."""

    def test_qa_gs_mb_missing(self, tmp_pq_dir):
        """Returns skipped when gs_mb.parquet is missing."""
        from corefunc.qa import qa_gs_mb
        report = qa_gs_mb()
        assert report["status"] == "skipped"

    def test_qa_gs_mb_valid(self, tmp_pq_dir):
        """Returns pass for valid gs_mb data."""
        from helpers.io import dump_parquet, GS_MB_PQ
        df = pd.DataFrame({
            "variant_a": ["Alpha", "Beta"],
            "variant_b": ["Alfa", "Beto"],
            "to_link": [True, False],
            "source": ["avc", "mbdb"],
        })
        dump_parquet(df, GS_MB_PQ)
        from corefunc.qa import qa_gs_mb
        report = qa_gs_mb()
        assert report["passed"] is True
        assert report["label_distribution"]["positive"] == 1
        assert report["source_breakdown"]["avc"] == 1


class TestQaUc:
    """Tests qa_uc function."""

    def test_qa_uc_missing(self, tmp_pq_dir):
        """Returns skipped when uc.parquet is missing."""
        from corefunc.qa import qa_uc
        report = qa_uc()
        assert report["status"] == "skipped"

    def test_qa_uc_valid(self, tmp_pq_dir):
        """Returns report for valid uc data."""
        from helpers.io import dump_parquet, UC_PQ
        df = pd.DataFrame({
            "country_code": ["DE", "HU", "DE"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"], utc=True),
        })
        dump_parquet(df, UC_PQ)
        from corefunc.qa import qa_uc
        report = qa_uc()
        assert report["row_count"] == 3
        assert report["unique_countries"] == 2
        assert report["passed"] is True


class TestQaPredictions:
    """Tests qa_predictions drift detection."""

    def test_missing_log(self, tmp_pq_dir):
        """Returns skipped when predictions_log is missing."""
        from corefunc.qa import qa_predictions
        report = qa_predictions()
        assert report["status"] == "skipped"

    def test_insufficient_data(self, tmp_pq_dir):
        """Returns insufficient_data when baseline or recent is empty."""
        from corefunc.qa import qa_predictions, PREDICTIONS_LOG_PQ
        from helpers.io import dump_parquet
        # All predictions are recent
        now = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame({
            "timestamp": [now - pd.Timedelta(hours=1)],
            "variant_a": ["Alpha"],
            "variant_b": ["Alfa"],
            "probability": [0.9],
            "features_json": [_SAMPLE_FEATS],
        })
        dump_parquet(df, PREDICTIONS_LOG_PQ)
        report = qa_predictions(baseline_days=30, recent_days=7)
        assert report["status"] == "insufficient_data"

    def test_no_drift(self, tmp_pq_dir):
        """Returns passed when no drift detected."""
        from corefunc.qa import qa_predictions, PREDICTIONS_LOG_PQ
        from helpers.io import dump_parquet
        now = pd.Timestamp.now(tz="UTC")
        # Creating baseline (15-20 days ago) and recent (1-3 days ago) data
        timestamps = (
            [now - pd.Timedelta(days=d) for d in range(15, 20)]
            + [now - pd.Timedelta(days=d) for d in range(1, 4)]
        )
        probs = [0.8, 0.85, 0.82, 0.79, 0.81, 0.80, 0.83, 0.81]
        feats = [_SAMPLE_FEATS] * 8
        pairs_a = ["Alpha"] * 5 + ["Beta"] * 3
        pairs_b = ["Alfa"] * 5 + ["Beto"] * 3
        df = pd.DataFrame({
            "timestamp": timestamps,
            "variant_a": pairs_a,
            "variant_b": pairs_b,
            "probability": probs,
            "features_json": feats,
        })
        dump_parquet(df, PREDICTIONS_LOG_PQ)
        report = qa_predictions(baseline_days=30, recent_days=7)
        assert report["passed"] is True
        assert "feature_quantiles" in report

    def test_drift_detected(self, tmp_pq_dir):
        """Returns warnings when mean probability shifts significantly."""
        from corefunc.qa import qa_predictions, PREDICTIONS_LOG_PQ
        from helpers.io import dump_parquet
        now = pd.Timestamp.now(tz="UTC")
        timestamps = (
            [now - pd.Timedelta(days=d) for d in range(15, 20)]
            + [now - pd.Timedelta(days=d) for d in range(1, 4)]
        )
        # Baseline ~0.9, recent ~0.5 => big shift
        probs = [0.9, 0.91, 0.88, 0.92, 0.89, 0.3, 0.35, 0.32]
        bl_feats = json.dumps({"wratio": 0.85, "partial_ratio": 0.9, "token_sort": 0.8})
        rc_feats = json.dumps({"wratio": 0.40, "partial_ratio": 0.35, "token_sort": 0.3})
        feats = [bl_feats] * 5 + [rc_feats] * 3
        pairs_a = ["Alpha"] * 5 + ["Gamma"] * 3
        pairs_b = ["Alfa"] * 5 + ["Gama"] * 3
        df = pd.DataFrame({
            "timestamp": timestamps,
            "variant_a": pairs_a,
            "variant_b": pairs_b,
            "probability": probs,
            "features_json": feats,
        })
        dump_parquet(df, PREDICTIONS_LOG_PQ)
        report = qa_predictions(baseline_days=30, recent_days=7)
        assert report["passed"] is False
        assert len(report["warnings"]) > 0
        assert "feature_quantiles" in report


# ═══════════════════════════════════════════════════════════════════════════════
# experiment_runner — helpers
# ═══════════════════════════════════════════════════════════════════════════════
class TestBuildModelCatalogue:
    """Tests _build_model_catalogue."""

    def test_returns_all_models(self):
        """Returns the expected 8 base + composite models."""
        from corefunc.canon.experiment_runner import _build_model_catalogue
        cat = _build_model_catalogue(spw=1.5, device="cpu", random_state=42)
        assert "XGBoost" in cat
        assert "LightGBM" in cat
        assert "RandomForest" in cat
        assert "ExtraTrees" in cat
        assert "GradientBoosting" in cat
        assert "VotingEnsemble" in cat
        assert "StackingEnsemble" in cat
        assert "BaggingXGB" in cat
        assert len(cat) == 8


class TestSafeGetParams:
    """Tests _safe_get_params."""

    def test_extracts_primitive_params(self):
        """Returns only str/int/float/bool params."""
        from corefunc.canon.experiment_runner import _safe_get_params
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=100, verbosity=-1)
        params = _safe_get_params(clf)
        assert "n_estimators" in params
        assert params["n_estimators"] == 100

    def test_handles_no_get_params(self):
        """Returns empty dict for objects without get_params."""
        from corefunc.canon.experiment_runner import _safe_get_params
        assert _safe_get_params(object()) == {}


class TestEvaluate:
    """Tests _evaluate."""

    def test_computes_metrics(self):
        """Returns precision, recall, f1, auc."""
        from corefunc.canon.experiment_runner import _evaluate
        from sklearn.linear_model import LogisticRegression
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
        y = np.array([1, 0, 1, 0, 1, 0])
        model = LogisticRegression(max_iter=200).fit(X, y)
        metrics = _evaluate(model, X, y)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert 0 <= metrics["auc"] <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# tuner — search spaces
# ═══════════════════════════════════════════════════════════════════════════════
class TestTunerSearchSpaces:
    """Tests search space constructors return proper classifiers."""

    def _mock_trial(self):
        """Creates a mock Optuna trial that returns valid values."""
        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, low, high, **kw: (low + high) // 2
        trial.suggest_float.side_effect = lambda name, low, high, **kw: (low + high) / 2
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        return trial

    def test_xgb_space(self):
        """Returns an XGBClassifier with sampled params."""
        from corefunc.canon.tuner import _xgb_search_space
        clf = _xgb_search_space(self._mock_trial(), spw=1.5, device="cpu")
        assert hasattr(clf, "fit")
        assert clf.device == "cpu"

    def test_lgbm_space(self):
        """Returns an LGBMClassifier with sampled params."""
        from corefunc.canon.tuner import _lgbm_search_space
        clf = _lgbm_search_space(self._mock_trial(), spw=1.5)
        assert hasattr(clf, "fit")

    def test_et_space(self):
        """Returns an ExtraTreesClassifier with sampled params."""
        from corefunc.canon.tuner import _et_search_space
        clf = _et_search_space(self._mock_trial())
        assert hasattr(clf, "fit")


# ═══════════════════════════════════════════════════════════════════════════════
# helpers/schema.py — uncovered validation/migration
# ═══════════════════════════════════════════════════════════════════════════════
class TestSchemaValidation:
    """Tests helpers/schema.py validation and migration paths."""

    def test_validate_schema_scrobble(self, populated_pq):
        """Validates a scrobble parquet file."""
        from helpers.schema import validate_schema
        from helpers.io import SCROBBLE_PQ
        info = validate_schema(SCROBBLE_PQ)
        assert "table" in info
        assert "file_version" in info
        assert "status" in info

    def test_validate_schema_artist_info(self, populated_pq):
        """Validates an artist_info parquet file."""
        from helpers.schema import validate_schema
        from helpers.io import ARTIST_INFO_PQ
        info = validate_schema(ARTIST_INFO_PQ)
        assert "table" in info

    def test_validate_nonexistent(self, tmp_pq_dir):
        """Handles nonexistent file gracefully."""
        from helpers.schema import validate_schema
        info = validate_schema(tmp_pq_dir / "nonexistent.parquet")
        assert info["status"] == "not_found" or "error" in info.get("status", "").lower() or info.get("table") is None

    def test_read_file_version(self, populated_pq):
        """Reads version from file metadata."""
        from helpers.schema import read_file_version
        from helpers.io import SCROBBLE_PQ
        result = read_file_version(SCROBBLE_PQ)
        # Returns a tuple (table_name_or_None, version_int)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_migrate_all(self, populated_pq):
        """Runs migration on a populated PQ dir."""
        from helpers.schema import migrate_all
        from helpers.io import PQ_DIR
        results = migrate_all(PQ_DIR)
        assert isinstance(results, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# trainer.py — additional data-prep coverage
# ═══════════════════════════════════════════════════════════════════════════════
class TestTrainerHelpers:
    """Tests additional trainer helper functions."""

    def test_eval_at(self):
        """Evaluates model at a given threshold."""
        from corefunc.canon.trainer import _eval_at
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.3, 0.8, 0.2])
        result = _eval_at(y_true, y_prob, 0.5)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_optimal_threshold(self):
        """Finds the F1-optimal threshold."""
        from corefunc.canon.trainer import _optimal_threshold
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        thr, f1 = _optimal_threshold(y_true, y_prob)
        assert 0.0 < thr < 1.0
        assert 0.0 <= f1 <= 1.0

    def test_high_precision_threshold(self):
        """Finds threshold dict for high-precision operation."""
        from corefunc.canon.trainer import _high_precision_threshold
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        result = _high_precision_threshold(y_true, y_prob, min_precision=0.5)
        assert isinstance(result, dict)
        assert "threshold" in result

    @patch("helpers.experiment.mlflow")
    def test_verify_mlflow(self, mock_mlflow):
        """Does not raise when MLflow is accessible."""
        mock_mlflow.get_tracking_uri.return_value = "sqlite:///test.db"
        from corefunc.canon.trainer import verify_mlflow
        # Should not raise
        verify_mlflow()

    def test_next_experiment_number(self):
        """Returns an integer >= 1."""
        from corefunc.canon.trainer import _next_experiment_number
        with patch("mlflow.search_runs", return_value=pd.DataFrame()):
            with patch("mlflow.get_experiment_by_name", return_value=None):
                with patch("mlflow.set_tracking_uri"):
                    n = _next_experiment_number()
                    assert isinstance(n, int)
                    assert n >= 1


class TestTrainerFitGpuFallback:
    """Tests _fit_with_gpu_fallback from both trainer and experiment_runner."""

    def test_cpu_success(self):
        """Succeeds on CPU without fallback."""
        from corefunc.canon.trainer import _fit_with_gpu_fallback
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        pipe = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression(max_iter=200))])
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y = np.array([1, 0, 1, 0])
        fitted_pipe, device = _fit_with_gpu_fallback(pipe, X, y, "cpu")
        assert device == "cpu"


class TestTrainerLoadCatalogueLookups:
    """Tests _load_catalogue_lookups with mocked data."""

    def test_returns_tuple_of_dicts(self, tmp_pq_dir, sample_artist_info_df):
        """Returns a tuple of lookup dicts from artist_info."""
        from helpers.io import dump_parquet, ARTIST_INFO_PQ
        dump_parquet(sample_artist_info_df, ARTIST_INFO_PQ)
        from corefunc.canon.trainer import _load_catalogue_lookups
        lookups = _load_catalogue_lookups()
        assert isinstance(lookups, tuple)
        assert all(isinstance(d, dict) for d in lookups)
