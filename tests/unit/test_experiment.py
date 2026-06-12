"""
Unit tests for helpers.experiment — MLflow tracking wrapper.
"""

from unittest.mock import MagicMock, patch

from helpers.experiment import DEFAULT_EXPERIMENT, TRACKING_URI, init_experiment, log_cv_fold, log_metrics, log_params


class TestInitExperiment:
    """Tests for init_experiment."""

    @patch("helpers.experiment.mlflow")
    def test_sets_tracking_uri_and_experiment(self, mock_mlflow):
        """Configures URI and experiment, returning experiment ID."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "42"
        mock_mlflow.set_experiment.return_value = mock_exp
        exp_id = init_experiment("test-exp")
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        assert exp_id == "42"

    def test_tracking_uri_is_sqlite(self):
        """Uses a SQLite URI, not a filesystem path."""
        assert TRACKING_URI.startswith("sqlite:///")
        assert TRACKING_URI.endswith("mlruns.db")

    @patch("helpers.experiment.mlflow")
    def test_uses_default_experiment_name(self, mock_mlflow):
        """Falls back to DEFAULT_EXPERIMENT when no name is given."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_mlflow.set_experiment.return_value = mock_exp
        init_experiment()
        mock_mlflow.set_experiment.assert_called_once_with(DEFAULT_EXPERIMENT)


class TestLogParams:
    """Tests for log_params."""

    @patch("helpers.experiment.mlflow")
    def test_delegates_to_mlflow(self, mock_mlflow):
        """Forwards param dict to mlflow.log_params."""
        params = {"lr": 0.05, "depth": 4}
        log_params(params)
        mock_mlflow.log_params.assert_called_once_with(params)


class TestLogMetrics:
    """Tests for log_metrics."""

    @patch("helpers.experiment.mlflow")
    def test_delegates_to_mlflow(self, mock_mlflow):
        """Forwards metrics dict to mlflow.log_metrics."""
        metrics = {"auc": 0.95, "f1": 0.88}
        log_metrics(metrics)
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)

    @patch("helpers.experiment.mlflow")
    def test_passes_step(self, mock_mlflow):
        """Forwards step parameter when provided."""
        log_metrics({"auc": 0.9}, step=3)
        mock_mlflow.log_metrics.assert_called_once_with({"auc": 0.9}, step=3)


class TestLogCvFold:
    """Tests for log_cv_fold."""

    @patch("helpers.experiment.mlflow")
    def test_creates_nested_run(self, mock_mlflow):
        """Opens a nested run and logs fold index + metrics."""
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        log_cv_fold(2, {"f1": 0.85, "auc": 0.91})
        mock_mlflow.start_run.assert_called_once_with(run_name="fold_2", nested=True)
        mock_mlflow.log_param.assert_called_once_with("fold", 2)
        mock_mlflow.log_metrics.assert_called_once_with({"f1": 0.85, "auc": 0.91})
