"""Tests for corefunc/canon/trainer.py — run_training orchestration and data builders."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _tiny_split_df(n, seed=42):
    """Returns a small feature DataFrame with the trainer's expected columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "to_link": [0, 1] * (n // 2),
            "variant_a": [f"a{i}" for i in range(n)],
            "variant_b": [f"b{i}" for i in range(n)],
            "source": ["avc"] * n,
            "_key": [f"k{i}" for i in range(n)],
        }
    )


class TestRunTrainingOrchestration:
    """Tests run_training with the data/feature pipeline mocked and real tiny training."""

    @staticmethod
    def _mocks(models=None):
        """Returns the context-manager stack around a run_training call."""
        from lightgbm import LGBMClassifier

        train_df = _tiny_split_df(30)
        test_df = _tiny_split_df(10, seed=7)
        return (
            patch("corefunc.canon.trainer.verify_mlflow"),
            patch("corefunc.canon.trainer._dispatch_data_build", return_value=(["p"] * 30, ["q"] * 10)),
            patch("corefunc.canon.trainer._compute_features_for_split", side_effect=[train_df, test_df]),
            patch("corefunc.canon.trainer.prune_feature_columns", return_value=["f1", "f2"]),
            patch("corefunc.canon.trainer.get_device", return_value="cpu"),
            patch(
                "corefunc.canon.trainer._build_model_catalogue",
                return_value={"LightGBM": LGBMClassifier(n_estimators=5, verbosity=-1)},
            ),
            patch("corefunc.canon.trainer.experiment"),
            patch("mlflow.set_tag"),
        )

    def test_full_pass(self):
        """Runs the unified pipeline end-to-end for one model."""
        from corefunc.canon.trainer import run_training

        with contextlib_exit(self._mocks()) as mocks:
            mock_exp = mocks[6]
            results = run_training(
                run_name="test_run", n_folds=2, models=["LightGBM"], catalogue=False, experiment_num=42
            )
        assert set(results) == {"LightGBM"}
        assert {"auc", "opt_f1", "hiprec_prec", "cv_mean_f1"} <= results["LightGBM"].keys()
        mock_exp.init_experiment.assert_called_once()
        mock_exp.log_model.assert_called_once()

    def test_no_models_raises(self):
        """Raises RuntimeError when the model filter selects nothing."""
        from corefunc.canon.trainer import run_training

        with contextlib_exit(self._mocks()):
            with pytest.raises(RuntimeError, match="No models selected"):
                run_training(n_folds=2, models=["Nope"], catalogue=False, experiment_num=42)


def contextlib_exit(mocks):
    """Enters a tuple of patchers as a single context manager; returns the started mocks."""

    class _Stack:
        def __enter__(self):
            return [m.__enter__() for m in mocks]

        def __exit__(self, *exc):
            for m in reversed(mocks):
                m.__exit__(*exc)
            return False

    return _Stack()


class TestNextExperimentNumber:
    """Tests _next_experiment_number fallbacks and param parsing."""

    @patch("corefunc.canon.trainer.experiment")
    @patch("mlflow.tracking.MlflowClient")
    def test_no_experiment_returns_baseline(self, mock_client_cls, mock_exp):
        """Returns the baseline 16 when the experiment does not exist."""
        from corefunc.canon.trainer import _next_experiment_number

        mock_client_cls.return_value.get_experiment_by_name.return_value = None
        assert _next_experiment_number() == 16

    @patch("corefunc.canon.trainer.experiment")
    @patch("mlflow.tracking.MlflowClient")
    def test_max_plus_one_skips_bad_params(self, mock_client_cls, mock_exp):
        """Returns max(experiment)+1, ignoring non-integer params."""
        from corefunc.canon.trainer import _next_experiment_number

        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="1")
        client.search_runs.return_value = [
            SimpleNamespace(data=SimpleNamespace(params={"experiment": "20"})),
            SimpleNamespace(data=SimpleNamespace(params={"experiment": "abc"})),
            SimpleNamespace(data=SimpleNamespace(params={})),
        ]
        assert _next_experiment_number() == 21

    @patch("corefunc.canon.trainer.experiment")
    @patch("mlflow.tracking.MlflowClient", side_effect=RuntimeError("store down"))
    def test_store_failure_returns_baseline(self, mock_client_cls, mock_exp):
        """Returns the baseline 16 when the tracking store is unreachable."""
        from corefunc.canon.trainer import _next_experiment_number

        assert _next_experiment_number() == 16


class TestDbscanCappedCacheHit:
    """Tests the cached-parquet fast path of _build_dbscan_capped_training_data."""

    def test_returns_cached_parquet(self, tmp_path, monkeypatch):
        """Returns the cached capped pairs without touching the DBSCAN pipeline."""
        import corefunc.canon.trainer as trainer

        monkeypatch.setattr(trainer, "PQ_DIR", tmp_path)
        cached = pd.DataFrame({"variant_a": ["A"], "variant_b": ["B"], "to_link": [True]})
        cached.to_parquet(tmp_path / "gs_mb_dbscan_capped.parquet", index=False)
        result = trainer._build_dbscan_capped_training_data()
        assert len(result) == 1


class TestFeatureSepTrainingData:
    """Tests _build_feature_sep_training_data with tmp parquet inputs."""

    @staticmethod
    def _write_inputs(tmp_path):
        """Writes gs_mb.parquet positives and gs_mb_dbscan.parquet negatives."""
        positives = pd.DataFrame(
            {
                "variant_a": ["Metallica", "Nirvana", "Kraftwerk", "Portishead", "Radiohead", "Massive Attack"],
                "variant_b": ["Metalica", "Nirvanа", "Kraftwerkk", "Portished", "Radiohed", "Massive Atack"],
                "to_link": [True] * 6,
            }
        )
        positives.to_parquet(tmp_path / "gs_mb.parquet", index=False)
        negatives = pd.DataFrame(
            {
                "variant_a": ["Metallica", "Nirvana", "Kraftwerk", "Portishead", "Gamma", "Delta"],
                "variant_b": ["Metallixa", "Nirvanx", "Kraftwerx", "Portishex", "Gondola", "Telta"],
                "to_link": [False] * 6,
            }
        )
        negatives.to_parquet(tmp_path / "gs_mb_dbscan.parquet", index=False)

    def test_distribution_matched_happy_path(self, tmp_path, monkeypatch):
        """Combines sampled positives with distribution-matched negatives."""
        import corefunc.canon.trainer as trainer

        self._write_inputs(tmp_path)
        monkeypatch.setattr(trainer, "PQ_DIR", tmp_path)
        monkeypatch.setattr(trainer, "GS_MB_PQ", tmp_path / "gs_mb.parquet")
        train = trainer._build_feature_sep_training_data(neg_count=4)
        assert train["to_link"].sum() == 4
        assert (~train["to_link"]).sum() == len(train) - 4
        assert len(train) > 4

    def test_missing_gs_mb_raises(self, tmp_path, monkeypatch):
        """Raises when gs_mb.parquet is absent."""
        import corefunc.canon.trainer as trainer

        monkeypatch.setattr(trainer, "PQ_DIR", tmp_path)
        monkeypatch.setattr(trainer, "GS_MB_PQ", tmp_path / "gs_mb.parquet")
        with pytest.raises(RuntimeError, match="gs_mb.parquet"):
            trainer._build_feature_sep_training_data()

    def test_missing_dbscan_raises(self, tmp_path, monkeypatch):
        """Raises when gs_mb_dbscan.parquet is absent."""
        import corefunc.canon.trainer as trainer

        pd.DataFrame({"variant_a": ["A"], "variant_b": ["B"], "to_link": [True]}).to_parquet(
            tmp_path / "gs_mb.parquet", index=False
        )
        monkeypatch.setattr(trainer, "PQ_DIR", tmp_path)
        monkeypatch.setattr(trainer, "GS_MB_PQ", tmp_path / "gs_mb.parquet")
        with pytest.raises(RuntimeError, match="gs_mb_dbscan.parquet"):
            trainer._build_feature_sep_training_data()
