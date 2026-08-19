"""Tests for corefunc/canon/experiment_runner.py — experiment entry points."""

from unittest.mock import patch

import numpy as np
import pandas as pd


def _tiny_df(n=40, seed=42):
    """Returns a small feature DataFrame with a balanced to_link target."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "variant_a": [f"a{i}" for i in range(n)],
            "variant_b": [f"b{i}" for i in range(n)],
            "to_link": [0, 1] * (n // 2),
        }
    )


@patch("mlflow.set_tag")
@patch("corefunc.canon.experiment_runner.experiment")
@patch("corefunc.canon.experiment_runner.get_device", return_value="cpu")
class TestRunExperiment:
    """Tests run_experiment with mocked MLflow and a tiny real dataset."""

    def test_trains_filtered_models(self, mock_device, mock_exp, mock_tag):
        """Trains only the selected models and returns held-out metrics."""
        from corefunc.canon.experiment_runner import run_experiment

        with patch("corefunc.canon.model._build_gold_standard", return_value=_tiny_df()):
            results = run_experiment(models=["ExtraTrees"], n_folds=2, test_size=0.25)
        assert set(results) == {"ExtraTrees"}
        assert {"precision", "recall", "f1", "auc"} <= results["ExtraTrees"].keys()
        mock_exp.init_experiment.assert_called_once()
        mock_exp.log_model.assert_called_once()

    def test_default_runs_full_catalogue(self, mock_device, mock_exp, mock_tag):
        """Uses the whole catalogue when no model filter is given."""
        from corefunc.canon.experiment_runner import _build_model_catalogue, run_experiment

        n_catalogue = len(_build_model_catalogue(spw=1.0, device="cpu", random_state=47))
        with (
            patch("corefunc.canon.model._build_gold_standard", return_value=_tiny_df()),
            patch("corefunc.canon.experiment_runner._cv_evaluate", return_value={"cv_mean_f1": 0.5}),
            patch("corefunc.canon.experiment_runner._fit_with_gpu_fallback") as mock_fit,
            patch(
                "corefunc.canon.experiment_runner._evaluate",
                return_value={"precision": 0.5, "recall": 0.5, "f1": 0.5, "auc": 0.5},
            ),
        ):
            pipeline = MagicMockPipeline()
            mock_fit.return_value = (pipeline, "cpu")
            results = run_experiment(n_folds=2, test_size=0.25)
        assert len(results) == n_catalogue


class MagicMockPipeline:
    """Minimal pipeline stand-in for fully mocked experiment passes."""

    def __init__(self):
        from unittest.mock import MagicMock

        self._steps = MagicMock()
        self.named_steps = {"prep": MagicMock()}

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@patch("mlflow.set_tag")
@patch("corefunc.canon.experiment_runner.experiment")
@patch("corefunc.canon.experiment_runner.get_device", return_value="cpu")
class TestRunHoldoutExperiment:
    """Tests run_holdout_experiment with explicit train/test frames."""

    def test_trains_and_evaluates_on_holdout(self, mock_device, mock_exp, mock_tag):
        """Trains on train_df and evaluates on the fixed test_df."""
        from corefunc.canon.experiment_runner import run_holdout_experiment

        results = run_holdout_experiment(
            train_df=_tiny_df(40), test_df=_tiny_df(12, seed=7), models=["ExtraTrees"], n_folds=2
        )
        assert set(results) == {"ExtraTrees"}
        assert 0.0 <= results["ExtraTrees"]["auc"] <= 1.0
        mock_exp.log_model.assert_called_once()
