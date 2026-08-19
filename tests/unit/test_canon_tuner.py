"""Tests for corefunc/canon/tuner.py — objective branches, retrain, export, and orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _tiny_xy(n=30, n_test=12, seed=42):
    """Returns small numeric (X, y, X_test, y_test) frames with both classes."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = np.array([0, 1] * (n // 2))
    X_test = pd.DataFrame({"f1": rng.normal(size=n_test), "f2": rng.normal(size=n_test)})
    y_test = np.array([0, 1] * (n_test // 2))
    return X, y, X_test, y_test


def _fresh_trial():
    """Returns a real Optuna trial from a fresh maximise study."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    return study.ask()


class TestPrecisionBiasedObjectiveBranches:
    """Tests the model-branch and failure paths of _precision_biased_objective."""

    def test_xgboost_branch(self):
        """Scores a trial with the XGBoost search space."""
        from corefunc.canon.tuner import _precision_biased_objective

        X, y, _, _ = _tiny_xy()
        score = _precision_biased_objective(
            _fresh_trial(), "XGBoost", X, y, ["f1", "f2"], spw=1.0, device="cpu", n_folds=2
        )
        assert 0.0 <= score <= 1.0

    def test_extratrees_branch(self):
        """Scores a trial with the ExtraTrees search space."""
        from corefunc.canon.tuner import _precision_biased_objective

        X, y, _, _ = _tiny_xy()
        score = _precision_biased_objective(
            _fresh_trial(), "ExtraTrees", X, y, ["f1", "f2"], spw=1.0, device="cpu", n_folds=2
        )
        assert 0.0 <= score <= 1.0

    def test_fit_failure_returns_zero(self):
        """Returns 0.0 when a fold fit raises."""
        from sklearn.ensemble import ExtraTreesClassifier

        from corefunc.canon import tuner

        class FailingET(ExtraTreesClassifier):
            """ExtraTrees variant whose fit always fails."""

            def fit(self, *args, **kwargs):
                raise RuntimeError("synthetic fit failure")

        X, y, _, _ = _tiny_xy()
        with patch("corefunc.canon.tuner._et_search_space", return_value=FailingET(n_estimators=5)):
            score = tuner._precision_biased_objective(
                _fresh_trial(), "ExtraTrees", X, y, ["f1", "f2"], spw=1.0, device="cpu", n_folds=2
            )
        assert score == 0.0

    def test_unknown_model_raises(self):
        """Rejects unknown model names."""
        from corefunc.canon.tuner import _precision_biased_objective

        X, y, _, _ = _tiny_xy()
        with pytest.raises(ValueError, match="Unknown model"):
            _precision_biased_objective(_fresh_trial(), "Nope", X, y, ["f1", "f2"], spw=1.0, device="cpu")


@patch("corefunc.canon.tuner.experiment")
@patch("mlflow.log_artifact")
@patch("mlflow.set_tag")
class TestRetrainBest:
    """Tests _retrain_best model branches with real tiny training runs."""

    def test_lightgbm_full_pass(self, mock_tag, mock_art, mock_exp, tmp_path):
        """Trains LightGBM end-to-end and returns the metrics dict."""
        from corefunc.canon.tuner import _retrain_best

        X_train, y_train, X_test, y_test = _tiny_xy()
        with patch("corefunc.canon.tuner.ML_DIR", tmp_path):
            result = _retrain_best(
                "LightGBM",
                {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.1},
                X_train,
                y_train,
                X_test,
                y_test,
                ["f1", "f2"],
                spw=1.0,
                device="cpu",
                exp_num=1,
                n_folds=2,
            )
        assert {"auc", "opt_f1", "hiprec_prec", "cv_mean_f1"} <= result.keys()
        assert (tmp_path / "lightgbm_tuned.pkl").exists()
        mock_exp.log_model.assert_called_once()

    def test_xgboost_branch(self, mock_tag, mock_art, mock_exp, tmp_path):
        """Trains the XGBoost branch on CPU."""
        from corefunc.canon.tuner import _retrain_best

        X_train, y_train, X_test, y_test = _tiny_xy()
        with patch("corefunc.canon.tuner.ML_DIR", tmp_path):
            result = _retrain_best(
                "XGBoost",
                {"n_estimators": 5, "max_depth": 3},
                X_train,
                y_train,
                X_test,
                y_test,
                ["f1", "f2"],
                spw=1.0,
                device="cpu",
                exp_num=1,
                n_folds=2,
            )
        assert "auc" in result
        assert (tmp_path / "xgboost_tuned.pkl").exists()

    def test_extratrees_branch(self, mock_tag, mock_art, mock_exp, tmp_path):
        """Trains the ExtraTrees branch."""
        from corefunc.canon.tuner import _retrain_best

        X_train, y_train, X_test, y_test = _tiny_xy()
        with patch("corefunc.canon.tuner.ML_DIR", tmp_path):
            result = _retrain_best(
                "ExtraTrees",
                {"n_estimators": 5, "max_depth": 3},
                X_train,
                y_train,
                X_test,
                y_test,
                ["f1", "f2"],
                spw=1.0,
                device="cpu",
                exp_num=1,
                n_folds=2,
            )
        assert "auc" in result

    def test_unknown_model_raises(self, mock_tag, mock_art, mock_exp):
        """Rejects unknown model names before any training."""
        from corefunc.canon.tuner import _retrain_best

        X, y, X_test, y_test = _tiny_xy()
        with pytest.raises(ValueError, match="Unknown model"):
            _retrain_best("Nope", {}, X, y, X_test, y_test, ["f1"], 1.0, "cpu", 1)

    def test_device_fallback_logged(self, mock_tag, mock_art, mock_exp, tmp_path):
        """Logs device_fallback when the GPU fit falls back to CPU."""
        from lightgbm import LGBMClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        from corefunc.canon.tuner import _retrain_best

        X_train, y_train, X_test, y_test = _tiny_xy()
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), ["f1", "f2"])], verbose_feature_names_out=False
        )
        pre.set_output(transform="pandas")
        fitted = Pipeline([("prep", pre), ("clf", LGBMClassifier(n_estimators=5, verbosity=-1))])
        fitted.fit(X_train, y_train)
        with (
            patch("corefunc.canon.tuner.ML_DIR", tmp_path),
            patch("corefunc.canon.tuner._fit_with_gpu_fallback", return_value=(fitted, "cpu")),
        ):
            _retrain_best(
                "LightGBM",
                {"n_estimators": 5},
                X_train,
                y_train,
                X_test,
                y_test,
                ["f1", "f2"],
                1.0,
                "cuda",
                1,
                n_folds=2,
            )
        mock_exp.log_params.assert_any_call({"device_fallback": "cpu"})


class TestSaveBestHistoricalModels:
    """Tests save_best_historical_models selection and export paths."""

    @staticmethod
    def _run(run_id, model_type, opt_p, hip_p, hip_f1, auc):
        """Builds a fake MLflow run with 3-operating-point metrics."""
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id),
            data=SimpleNamespace(
                metrics={"opt_precision": opt_p, "hiprec_precision": hip_p, "hiprec_f1": hip_f1, "auc": auc},
                tags={"model_type": model_type},
            ),
        )

    @staticmethod
    def _fitted_model():
        """Returns a real fitted, picklable classifier."""
        from lightgbm import LGBMClassifier

        X, y, _, _ = _tiny_xy()
        return LGBMClassifier(n_estimators=2, verbosity=-1).fit(X, y)

    @patch("corefunc.canon.tuner.experiment")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_no_experiment_returns_empty(self, mock_uri, mock_client_cls, mock_exp):
        """Returns [] when the MLflow experiment does not exist."""
        from corefunc.canon.tuner import save_best_historical_models

        mock_client_cls.return_value.get_experiment_by_name.return_value = None
        assert save_best_historical_models() == []

    @patch("corefunc.canon.tuner.experiment")
    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_picks_highest_c9r_and_exports(self, mock_uri, mock_client_cls, mock_load, mock_exp, tmp_path):
        """Picks the best run per model type, skips ineligible runs, exports pickle."""
        from corefunc.canon.tuner import save_best_historical_models

        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
        client.search_runs.return_value = [
            self._run("run-worse", "LightGBM", 0.9, 0.90, 0.70, 0.80),  # c9r = 0.81
            self._run("run-better", "LightGBM", 0.9, 0.95, 0.75, 0.85),  # c9r = 0.86
            self._run("run-tuned", "LightGBM_tuned", 0.9, 0.99, 0.99, 0.99),  # skipped: tuned tag
            self._run("run-noopt", "XGBoost", 0.0, 0.99, 0.99, 0.99),  # skipped: opt_precision 0
        ]
        mock_load.return_value = self._fitted_model()
        with patch("corefunc.canon.tuner.ML_DIR", tmp_path):
            saved = save_best_historical_models()
        assert saved == [str(tmp_path / "lightgbm_best.pkl")]
        assert (tmp_path / "lightgbm_best.pkl").exists()
        mock_load.assert_called_once_with("runs:/run-better/model")

    @patch("corefunc.canon.tuner.experiment")
    @patch("mlflow.sklearn.load_model", side_effect=RuntimeError("broken artefact"))
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_tracking_uri")
    def test_export_failure_continues(self, mock_uri, mock_client_cls, mock_load, mock_exp, tmp_path):
        """Logs and skips a run whose model fails to load."""
        from corefunc.canon.tuner import save_best_historical_models

        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
        client.search_runs.return_value = [self._run("run-broken", "LightGBM", 0.9, 0.9, 0.7, 0.8)]
        with patch("corefunc.canon.tuner.ML_DIR", tmp_path):
            assert save_best_historical_models() == []


class TestRunTuning:
    """Tests run_tuning orchestration with all heavy dependencies mocked."""

    @staticmethod
    def _metrics():
        """Returns a plausible _retrain_best metrics dict."""
        return {
            "auc": 0.90,
            "default_f1": 0.80,
            "default_prec": 0.85,
            "default_rec": 0.75,
            "opt_thr": 0.50,
            "opt_f1": 0.82,
            "opt_prec": 0.88,
            "opt_rec": 0.77,
            "hiprec_thr": 0.90,
            "hiprec_f1": 0.60,
            "hiprec_prec": 0.95,
            "hiprec_rec": 0.50,
        }

    @staticmethod
    def _fake_study():
        """Returns a fake Optuna study with a completed best trial."""
        trial = SimpleNamespace(
            number=3,
            value=0.92,
            user_attrs={"cv_mean_precision": 0.92, "cv_worst_precision": 0.90, "cv_mean_f1": 0.85},
            params={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "is_unbalance": True, "verbosity": -1},
        )
        return SimpleNamespace(
            best_trial=trial,
            trials=[
                SimpleNamespace(number=0, value=0.9, params={}, user_attrs={}, state=SimpleNamespace(name="COMPLETE"))
            ],
            optimize=lambda *a, **k: None,
        )

    def _run(self, models=None):
        """Executes run_tuning under the standard mock stack."""
        from corefunc.canon.tuner import run_tuning

        train_df = pd.DataFrame(
            {
                "f1": np.arange(10, dtype="float64"),
                "f2": np.arange(10, dtype="float64") * 2,
                "to_link": [0, 1] * 5,
                "variant_a": [f"a{i}" for i in range(10)],
                "variant_b": [f"b{i}" for i in range(10)],
                "source": ["mb"] * 10,
                "_key": [f"k{i}" for i in range(10)],
            }
        )
        # Deliberately missing f2 to exercise the fill-0 warning branch
        test_df = pd.DataFrame({"f1": np.arange(4, dtype="float64"), "to_link": [0, 1, 0, 1]})
        with (
            patch("corefunc.canon.tuner.verify_mlflow"),
            patch("corefunc.canon.tuner._next_experiment_number", return_value=7),
            patch("corefunc.canon.tuner.save_best_historical_models", return_value=["ML/lightgbm_best.pkl"]),
            patch("corefunc.canon.tuner.build_training_data", return_value=(["p"] * 10, ["q"] * 4)),
            patch("corefunc.canon.tuner._load_catalogue_lookups", return_value=({"a": []}, {"a": []})),
            patch("corefunc.canon.tuner.compute_all_features", side_effect=[train_df, test_df]),
            patch("corefunc.canon.tuner.prune_feature_columns", return_value=["f1", "f2"]),
            patch("corefunc.canon.tuner.get_device", return_value="cpu"),
            patch("corefunc.canon.tuner.experiment") as mock_exp,
            patch("corefunc.canon.tuner.optuna.create_study", return_value=self._fake_study()),
            patch("corefunc.canon.tuner._retrain_best", return_value=self._metrics()) as mock_retrain,
        ):
            result = run_tuning(models=models, n_trials=2)
        return result, mock_retrain, mock_exp

    def test_full_orchestration(self):
        """Runs the full tuning pass for the default LightGBM model."""
        result, mock_retrain, mock_exp = self._run()
        assert set(result) == {"LightGBM"}
        assert result["LightGBM"]["auc"] == 0.90
        # Non-model params filtered before rebuilding the classifier
        best_params = mock_retrain.call_args[0][1]
        assert "is_unbalance" not in best_params
        assert "verbosity" not in best_params
        assert best_params["n_estimators"] == 300
        mock_exp.start_run.assert_called_once()

    def test_no_tunable_models_raises(self):
        """Raises RuntimeError when no known model is selected."""
        with pytest.raises(RuntimeError, match="No tunable models"):
            self._run(models=["Nope"])
