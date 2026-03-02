"""
Tests for helpers.experiment — MLflow tracking wrapper.
"""

import numpy as np
from unittest.mock import patch, MagicMock


class TestInitAndBasicOps:
    """Tests init_experiment and basic logging functions."""

    @patch("helpers.experiment.mlflow")
    def test_init_experiment(self, mock_mlflow):
        """Configures tracking URI and sets experiment."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "42"
        mock_mlflow.set_experiment.return_value = mock_exp
        from helpers.experiment import init_experiment

        eid = init_experiment("test-exp")
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        assert eid == "42"

    @patch("helpers.experiment.mlflow")
    def test_start_run(self, mock_mlflow):
        """Delegates to mlflow.start_run."""
        from helpers.experiment import start_run

        start_run(run_name="test", nested=True)
        mock_mlflow.start_run.assert_called_once_with(run_name="test", nested=True)

    @patch("helpers.experiment.mlflow")
    def test_log_params(self, mock_mlflow):
        """Delegates to mlflow.log_params."""
        from helpers.experiment import log_params

        log_params({"a": 1, "b": "x"})
        mock_mlflow.log_params.assert_called_once_with({"a": 1, "b": "x"})

    @patch("helpers.experiment.mlflow")
    def test_log_metrics(self, mock_mlflow):
        """Delegates to mlflow.log_metrics."""
        from helpers.experiment import log_metrics

        log_metrics({"f1": 0.85}, step=3)
        mock_mlflow.log_metrics.assert_called_once_with({"f1": 0.85}, step=3)

    @patch("helpers.experiment.mlflow")
    def test_log_artifact(self, mock_mlflow):
        """Delegates to mlflow.log_artifact."""
        from helpers.experiment import log_artifact

        log_artifact("/tmp/file.txt")
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/file.txt")

    @patch("helpers.experiment.mlflow")
    def test_log_model(self, mock_mlflow):
        """Delegates to mlflow.sklearn.log_model."""
        from helpers.experiment import log_model

        model = MagicMock()
        log_model(model, artifact_path="my_model")
        mock_mlflow.sklearn.log_model.assert_called_once_with(model, name="my_model")

    @patch("helpers.experiment.mlflow")
    def test_log_cv_fold(self, mock_mlflow):
        """Creates a nested run for a CV fold."""
        from helpers.experiment import log_cv_fold

        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        log_cv_fold(2, {"f1": 0.9}, run_name_prefix="xgb_fold")
        mock_mlflow.start_run.assert_called_once_with(run_name="xgb_fold_2", nested=True)


class TestConfusionMatrix:
    """Tests log_confusion_matrix with mocked matplotlib and mlflow."""

    @patch("helpers.experiment.mlflow")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.close")
    def test_saves_and_logs(self, mock_close, mock_subplots, mock_mlflow, tmp_path):
        """Generates confusion matrix plot and logs to MLflow."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        with patch("helpers.experiment.PROJECT_ROOT", tmp_path):
            from helpers.experiment import log_confusion_matrix

            (tmp_path / "ML").mkdir()
            log_confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
        mock_fig.savefig.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()


class TestFeatureImportance:
    """Tests log_feature_importance."""

    @patch("helpers.experiment.mlflow")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.close")
    def test_direct_importances(self, mock_close, mock_subplots, mock_mlflow, tmp_path):
        """Logs feature importances from a model with feature_importances_."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        model = MagicMock()
        model.feature_importances_ = np.array([0.3, 0.7])
        with patch("helpers.experiment.PROJECT_ROOT", tmp_path):
            (tmp_path / "ML").mkdir()
            from helpers.experiment import log_feature_importance

            log_feature_importance(model, ["feat_a", "feat_b"], top_n=2)
        mock_fig.savefig.assert_called_once()

    @patch("helpers.experiment.mlflow")
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.close")
    def test_pipeline_importances(self, mock_close, mock_subplots, mock_mlflow, tmp_path):
        """Extracts importances from a Pipeline's last step."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        inner = MagicMock()
        inner.feature_importances_ = np.array([0.5, 0.5])
        model = MagicMock()
        model.named_steps = {"scaler": MagicMock(), "clf": inner}
        del model.feature_importances_
        with patch("helpers.experiment.PROJECT_ROOT", tmp_path):
            (tmp_path / "ML").mkdir()
            from helpers.experiment import log_feature_importance

            log_feature_importance(model, ["a", "b"])
        mock_fig.savefig.assert_called_once()

    @patch("helpers.experiment.mlflow")
    def test_no_importances_skips(self, mock_mlflow):
        """Skips when model has no feature_importances_."""
        model = MagicMock(spec=[])
        from helpers.experiment import log_feature_importance

        log_feature_importance(model, ["a"])
        mock_mlflow.log_artifact.assert_not_called()


class TestGetShapEstimator:
    """Tests _get_shap_estimator for ensemble and CUDA handling."""

    def test_voting_extracts_sub(self):
        """Extracts tree-based sub-estimator from VotingClassifier."""
        from sklearn.ensemble import VotingClassifier
        from helpers.experiment import _get_shap_estimator

        sub = MagicMock()
        sub.feature_importances_ = np.array([0.5])
        vc = VotingClassifier(estimators=[("a", sub)])
        vc.estimators_ = [sub]
        result = _get_shap_estimator(vc)
        assert result is sub

    def test_stacking_extracts_sub(self):
        """Extracts tree-based sub-estimator from StackingClassifier."""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from helpers.experiment import _get_shap_estimator

        sub = MagicMock()
        sub.feature_importances_ = np.array([0.5])
        sc = StackingClassifier(
            estimators=[("a", sub)],
            final_estimator=LogisticRegression(),
        )
        sc.estimators_ = [sub]
        result = _get_shap_estimator(sc)
        assert result is sub

    def test_bagging_extracts_first(self):
        """Extracts first estimator from BaggingClassifier."""
        from sklearn.ensemble import BaggingClassifier
        from helpers.experiment import _get_shap_estimator

        sub = MagicMock()
        bc = BaggingClassifier()
        bc.estimators_ = [sub]
        result = _get_shap_estimator(bc)
        assert result is sub

    def test_bagging_empty_returns_none(self):
        """Returns None when BaggingClassifier has no fitted estimators."""
        from sklearn.ensemble import BaggingClassifier
        from helpers.experiment import _get_shap_estimator

        bc = BaggingClassifier()
        bc.estimators_ = []
        assert _get_shap_estimator(bc) is None

    def test_voting_no_tree_returns_none(self):
        """Returns None when no sub-estimator has feature_importances_."""
        from sklearn.ensemble import VotingClassifier
        from helpers.experiment import _get_shap_estimator

        sub = MagicMock(spec=[])  # no feature_importances_
        vc = VotingClassifier(estimators=[("a", sub)])
        vc.estimators_ = [sub]
        assert _get_shap_estimator(vc) is None

    def test_cuda_deep_copies_to_cpu(self):
        """Deep-copies CUDA XGBoost to CPU."""
        from helpers.experiment import _get_shap_estimator

        model = MagicMock()
        model.device = "cuda:0"
        model.feature_importances_ = np.array([0.5])
        result = _get_shap_estimator(model)
        assert result is not None


class TestLogShapSummary:
    """Tests log_shap_summary with mocked shap."""

    @patch("helpers.experiment.mlflow")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.tight_layout", create=True)
    def test_shap_summary_pipeline(self, _mock_tight, mock_figure, mock_mlflow, tmp_path):
        """Runs SHAP on a Pipeline model with mocked shap module."""
        import pandas as pd

        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[0.1, 0.2]])
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        inner = MagicMock()
        inner.feature_importances_ = np.array([0.5, 0.5])
        model = MagicMock()
        model.named_steps = {"clf": inner}
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with patch.dict("sys.modules", {"shap": mock_shap}):
            with patch("helpers.experiment.PROJECT_ROOT", tmp_path):
                (tmp_path / "ML").mkdir()
                from helpers.experiment import log_shap_summary

                log_shap_summary(model, X, ["a", "b"])

    @patch("helpers.experiment.mlflow")
    def test_shap_import_error(self, mock_mlflow):
        """Skips gracefully when shap is not installed."""
        with patch.dict("sys.modules", {"shap": None}):
            from helpers.experiment import log_shap_summary

            # This should not raise
            log_shap_summary(MagicMock(), np.array([[1]]), ["a"])
