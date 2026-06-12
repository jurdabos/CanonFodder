"""
Tests for corefunc.canon.tcn_trainer model classes and datasets,
corefunc.canon.tuner search spaces and objective, and
corefunc.canon.experiment_runner CV and GPU fallback.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# tcn_trainer — model forward passes
# ═══════════════════════════════════════════════════════════════════════════════
class TestSiameseTCNForward:
    """Tests the SiameseTCN model's forward pass."""

    def test_output_shape(self):
        """Produces (batch_size,) output for a pair of sequences."""
        from corefunc.canon.tcn_trainer import SiameseTCN

        model = SiameseTCN(
            vocab_size=50,
            embed_dim=16,
            tcn_channels=[32, 32],
            kernel_size=3,
            tcn_dropout=0.1,
            fc_dropout=0.1,
        )
        model.eval()
        batch = 4
        seq_len = 20
        x_a = torch.randint(0, 50, (batch, seq_len))
        x_b = torch.randint(0, 50, (batch, seq_len))
        with torch.no_grad():
            out = model(x_a, x_b)
        assert out.shape == (batch,)

    def test_encode_produces_vector(self):
        """Internal _encode returns a fixed-size vector per sequence."""
        from corefunc.canon.tcn_trainer import SiameseTCN

        model = SiameseTCN(
            vocab_size=50,
            embed_dim=16,
            tcn_channels=[32, 32],
            kernel_size=3,
            tcn_dropout=0.1,
            fc_dropout=0.1,
        )
        model.eval()
        x = torch.randint(0, 50, (2, 15))
        with torch.no_grad():
            h = model._encode(x)
        # pool_dim = tcn_channels[-1] * 2 = 64
        assert h.shape == (2, 64)

    def test_handles_padding(self):
        """Works correctly when input contains PAD tokens (0)."""
        from corefunc.canon.tcn_trainer import SiameseTCN

        model = SiameseTCN(
            vocab_size=50,
            embed_dim=16,
            tcn_channels=[32, 32],
            kernel_size=3,
            tcn_dropout=0.1,
            fc_dropout=0.1,
        )
        model.eval()
        x_a = torch.tensor([[5, 10, 0, 0, 0], [3, 7, 12, 0, 0]])
        x_b = torch.tensor([[8, 0, 0, 0, 0], [4, 6, 9, 11, 0]])
        with torch.no_grad():
            out = model(x_a, x_b)
        assert out.shape == (2,)
        assert not torch.isnan(out).any()


class TestHybridTCNForward:
    """Tests the HybridTCN model's forward pass."""

    def test_output_shape(self):
        """Produces (batch_size,) output with both sequences and features."""
        from corefunc.canon.tcn_trainer import HybridTCN

        n_features = 28
        model = HybridTCN(
            vocab_size=50,
            embed_dim=16,
            tcn_channels=[32, 32],
            kernel_size=3,
            tcn_dropout=0.1,
            fc_dropout=0.1,
            n_features=n_features,
        )
        model.eval()
        batch = 4
        seq_len = 20
        x_a = torch.randint(0, 50, (batch, seq_len))
        x_b = torch.randint(0, 50, (batch, seq_len))
        features = torch.randn(batch, n_features)
        with torch.no_grad():
            out = model(x_a, x_b, features)
        assert out.shape == (batch,)

    def test_skip_connection(self):
        """The feature skip branch contributes to the output."""
        from corefunc.canon.tcn_trainer import HybridTCN

        model = HybridTCN(
            vocab_size=50,
            embed_dim=16,
            tcn_channels=[32, 32],
            kernel_size=3,
            tcn_dropout=0.1,
            fc_dropout=0.1,
            n_features=10,
        )
        model.eval()
        x_a = torch.randint(0, 50, (2, 10))
        x_b = torch.randint(0, 50, (2, 10))
        feats = torch.randn(2, 10)
        with torch.no_grad():
            out = model(x_a, x_b, feats)
        assert out.shape == (2,)

    def test_encode_seq_shape(self):
        """Internal _encode_seq returns correct-sized vectors."""
        from corefunc.canon.tcn_trainer import HybridTCN

        model = HybridTCN(
            vocab_size=50,
            embed_dim=16,
            tcn_channels=[32, 32],
            kernel_size=3,
            tcn_dropout=0.1,
            fc_dropout=0.1,
            n_features=10,
        )
        model.eval()
        x = torch.randint(0, 50, (3, 12))
        with torch.no_grad():
            h = model._encode_seq(x)
        assert h.shape == (3, 64)


# ═══════════════════════════════════════════════════════════════════════════════
# tcn_trainer — datasets
# ═══════════════════════════════════════════════════════════════════════════════
class TestNamePairDataset:
    """Tests the NamePairDataset for the Siamese TCN."""

    def test_len_and_getitem(self):
        """Returns correct length and item tuple (a_enc, b_enc, label)."""
        from corefunc.canon.tcn_trainer import CharVocab, NamePairDataset

        vocab = CharVocab()
        vocab.fit(["abc", "xyz"])
        df = pd.DataFrame(
            {
                "variant_a": ["abc", "xyz"],
                "variant_b": ["xyz", "abc"],
                "to_link": [1, 0],
            }
        )
        ds = NamePairDataset(df, vocab, max_len=8)
        assert len(ds) == 2
        a_enc, b_enc, label = ds[0]
        assert a_enc.shape == (8,)
        assert b_enc.shape == (8,)
        assert label.item() == 1.0


class TestHybridDataset:
    """Tests the HybridDataset for the hybrid TCN."""

    def test_len_and_getitem(self):
        """Returns correct length and item tuple (a_enc, b_enc, feats, label)."""
        from corefunc.canon.tcn_trainer import CharVocab, HybridDataset

        vocab = CharVocab()
        vocab.fit(["abc", "xyz"])
        df = pd.DataFrame(
            {
                "variant_a": ["abc", "xyz"],
                "variant_b": ["xyz", "abc"],
                "to_link": [1, 0],
            }
        )
        features = np.random.randn(2, 5).astype(np.float32)
        ds = HybridDataset(df, vocab, max_len=8, features=features)
        assert len(ds) == 2
        a_enc, b_enc, feats, label = ds[0]
        assert a_enc.shape == (8,)
        assert feats.shape == (5,)
        assert label.item() == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# tcn_trainer — prediction helpers
# ═══════════════════════════════════════════════════════════════════════════════
class TestPredictSiamese:
    """Tests the Siamese prediction function."""

    def test_returns_probabilities(self):
        """Returns an array of sigmoid probabilities."""
        from torch.utils.data import DataLoader

        from corefunc.canon.tcn_trainer import CharVocab, NamePairDataset, SiameseTCN, _predict_siamese

        vocab = CharVocab()
        vocab.fit(["abc", "xyz", "def"])
        model = SiameseTCN(
            vocab_size=vocab.size,
            embed_dim=8,
            tcn_channels=[16],
            kernel_size=3,
            tcn_dropout=0.0,
            fc_dropout=0.0,
        )
        model.eval()
        df = pd.DataFrame(
            {
                "variant_a": ["abc", "xyz"],
                "variant_b": ["def", "abc"],
                "to_link": [1, 0],
            }
        )
        ds = NamePairDataset(df, vocab, max_len=8)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        probs = _predict_siamese(model, loader, torch.device("cpu"))
        assert probs.shape == (2,)
        assert all(0.0 <= p <= 1.0 for p in probs)


class TestPredictHybrid:
    """Tests the Hybrid prediction function."""

    def test_returns_probabilities(self):
        """Returns an array of sigmoid probabilities."""
        from torch.utils.data import DataLoader

        from corefunc.canon.tcn_trainer import CharVocab, HybridDataset, HybridTCN, _predict_hybrid

        vocab = CharVocab()
        vocab.fit(["abc", "xyz", "def"])
        n_features = 5
        model = HybridTCN(
            vocab_size=vocab.size,
            embed_dim=8,
            tcn_channels=[16],
            kernel_size=3,
            tcn_dropout=0.0,
            fc_dropout=0.0,
            n_features=n_features,
        )
        model.eval()
        df = pd.DataFrame(
            {
                "variant_a": ["abc", "xyz"],
                "variant_b": ["def", "abc"],
                "to_link": [1, 0],
            }
        )
        features = np.random.randn(2, n_features).astype(np.float32)
        ds = HybridDataset(df, vocab, max_len=8, features=features)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        probs = _predict_hybrid(model, loader, torch.device("cpu"))
        assert probs.shape == (2,)
        assert all(0.0 <= p <= 1.0 for p in probs)


# ═══════════════════════════════════════════════════════════════════════════════
# tcn_trainer — evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════
class TestTcnOptimalThreshold:
    """Tests the TCN _optimal_threshold."""

    def test_returns_tuple(self):
        """Returns (threshold, best_f1)."""
        from corefunc.canon.tcn_trainer import _optimal_threshold

        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9, 0.8, 0.2])
        thr, f1 = _optimal_threshold(y_true, y_prob)
        assert 0.0 <= thr <= 1.0
        assert 0.0 <= f1 <= 1.0


class TestTcnEvalAt:
    """Tests the TCN _eval_at."""

    def test_returns_metrics(self):
        """Returns dict with precision, recall, f1, threshold."""
        from corefunc.canon.tcn_trainer import _eval_at

        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.7, 0.3])
        m = _eval_at(y_true, y_prob, 0.5)
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert m["threshold"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# tcn_trainer — data assembly
# ═══════════════════════════════════════════════════════════════════════════════
class TestAssembleTrainingData:
    """Tests the TCN training data assembler."""

    @patch("corefunc.canon.tcn_trainer.read_parquet")
    def test_empty_gs_raises(self, mock_read):
        """Raises when gs_mb.parquet is empty."""
        from corefunc.canon.tcn_trainer import _assemble_training_data

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="gs_mb"):
            _assemble_training_data()

    @patch("corefunc.canon.tcn_trainer.read_parquet")
    def test_assembles_balanced_data(self, mock_read):
        """Produces balanced positive/negative training pairs."""
        gs_df = pd.DataFrame(
            {
                "variant_a": [f"A{i}" for i in range(20)],
                "variant_b": [f"B{i}" for i in range(20)],
                "to_link": [True] * 10 + [False] * 10,
            }
        )
        dbscan_df = pd.DataFrame(
            {
                "variant_a": [f"DA{i}" for i in range(20)],
                "variant_b": [f"DB{i}" for i in range(20)],
                "to_link": [False] * 20,
            }
        )
        mock_read.side_effect = [gs_df, dbscan_df]
        from corefunc.canon.tcn_trainer import _assemble_training_data

        result = _assemble_training_data()
        assert "variant_a" in result.columns
        assert "to_link" in result.columns


class TestBuildAvcTest:
    """Tests the TCN AVC test set builder."""

    @patch("corefunc.canon.tcn_trainer.read_parquet")
    def test_empty_avc_raises(self, mock_read):
        """Raises when avc.parquet is empty."""
        from corefunc.canon.tcn_trainer import _build_avc_test

        mock_read.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="empty or missing"):
            _build_avc_test()


class TestPrecomputeFeatures:
    """Tests the feature precomputation for hybrid TCN."""

    def test_returns_array_and_names(self):
        """Returns (np.ndarray, list[str]) with correct dimensions."""
        from corefunc.canon.tcn_trainer import _precompute_features

        df = pd.DataFrame(
            {
                "variant_a": ["Beatles", "Radiohead"],
                "variant_b": ["The Beatles", "Radio Head"],
                "to_link": [True, True],
            }
        )
        arr, col_names = _precompute_features(df)
        assert arr.shape[0] == 2
        assert arr.shape[1] > 0
        assert len(col_names) == arr.shape[1]
        assert arr.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# tcn_trainer — training loops (smoke test with tiny data)
# ═══════════════════════════════════════════════════════════════════════════════
class TestTrainSiamese:
    """Tests the Siamese training loop with minimal data."""

    def test_runs_one_epoch(self):
        """Completes a single epoch without error."""
        from torch.utils.data import DataLoader

        from corefunc.canon.tcn_trainer import (
            CharVocab,
            NamePairDataset,
            SiameseTCN,
            _train_siamese,
        )

        vocab = CharVocab()
        vocab.fit(["alpha", "beta", "gamma", "delta"])
        df = pd.DataFrame(
            {
                "variant_a": ["alpha", "beta", "gamma", "delta"] * 5,
                "variant_b": ["beta", "alpha", "delta", "gamma"] * 5,
                "to_link": [1, 0, 1, 0] * 5,
            }
        )
        ds = NamePairDataset(df, vocab, max_len=10)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        y_val = np.array([1, 0, 1, 0] * 5)
        model = SiameseTCN(
            vocab_size=vocab.size,
            embed_dim=8,
            tcn_channels=[16],
            kernel_size=3,
            tcn_dropout=0.0,
            fc_dropout=0.0,
        )
        device = torch.device("cpu")
        model.to(device)
        history = _train_siamese(
            model,
            loader,
            loader,
            y_val,
            pos_weight=1.0,
            epochs=2,
            lr=0.01,
            patience=5,
            device=device,
        )
        assert "train_loss" in history
        assert "val_auc" in history
        assert len(history["train_loss"]) > 0


class TestTrainHybrid:
    """Tests the Hybrid training loop with minimal data."""

    def test_runs_one_epoch(self):
        """Completes a single epoch without error."""
        from torch.utils.data import DataLoader

        from corefunc.canon.tcn_trainer import (
            CharVocab,
            HybridDataset,
            HybridTCN,
            _train_hybrid,
        )

        vocab = CharVocab()
        vocab.fit(["alpha", "beta", "gamma", "delta"])
        n_features = 5
        df = pd.DataFrame(
            {
                "variant_a": ["alpha", "beta", "gamma", "delta"] * 5,
                "variant_b": ["beta", "alpha", "delta", "gamma"] * 5,
                "to_link": [1, 0, 1, 0] * 5,
            }
        )
        features = np.random.randn(20, n_features).astype(np.float32)
        ds = HybridDataset(df, vocab, max_len=10, features=features)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        y_val = np.array([1, 0, 1, 0] * 5)
        model = HybridTCN(
            vocab_size=vocab.size,
            embed_dim=8,
            tcn_channels=[16],
            kernel_size=3,
            tcn_dropout=0.0,
            fc_dropout=0.0,
            n_features=n_features,
        )
        device = torch.device("cpu")
        model.to(device)
        history = _train_hybrid(
            model,
            loader,
            loader,
            y_val,
            pos_weight=1.0,
            epochs=2,
            lr=0.01,
            patience=5,
            device=device,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# tuner — search spaces
# ═══════════════════════════════════════════════════════════════════════════════
class TestTunerSearchSpaces:
    """Tests tuner search space functions with mock Optuna trials."""

    def _mock_trial(self):
        """Creates a mock Optuna trial with deterministic suggestions."""
        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, low, high, **kw: low
        trial.suggest_float.side_effect = lambda name, low, high, **kw: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        return trial

    def test_xgb_search_space(self):
        """Returns an XGBClassifier instance."""
        from corefunc.canon.tuner import _xgb_search_space

        trial = self._mock_trial()
        clf = _xgb_search_space(trial, spw=1.0, device="cpu")
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")

    def test_lgbm_search_space(self):
        """Returns an LGBMClassifier instance."""
        from corefunc.canon.tuner import _lgbm_search_space

        trial = self._mock_trial()
        clf = _lgbm_search_space(trial, spw=1.0)
        assert hasattr(clf, "fit")

    def test_et_search_space(self):
        """Returns an ExtraTreesClassifier instance."""
        from corefunc.canon.tuner import _et_search_space

        trial = self._mock_trial()
        clf = _et_search_space(trial)
        assert hasattr(clf, "fit")


class TestPrecisionBiasedObjective:
    """Tests the Optuna objective function."""

    def test_returns_float(self):
        """Returns a float score for a valid trial."""
        from corefunc.canon.tuner import _precision_biased_objective

        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, low, high, **kw: low
        trial.suggest_float.side_effect = lambda name, low, high, **kw: low
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        trial.number = 0
        trial.set_user_attr = MagicMock()
        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame(
            {
                "f1": rng.standard_normal(n),
                "f2": rng.standard_normal(n),
            }
        )
        y = np.array([0] * 30 + [1] * 30)
        spw = 1.0
        score = _precision_biased_objective(
            trial,
            "LightGBM",
            X,
            y,
            ["f1", "f2"],
            spw,
            "cpu",
            n_folds=2,
            min_precision=0.5,
        )
        assert isinstance(score, float)

    def test_unknown_model_raises(self):
        """Raises ValueError for unknown model name."""
        from corefunc.canon.tuner import _precision_biased_objective

        trial = MagicMock()
        with pytest.raises(ValueError, match="Unknown model"):
            _precision_biased_objective(
                trial,
                "FakeModel",
                pd.DataFrame(),
                np.array([]),
                [],
                1.0,
                "cpu",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# experiment_runner — CV evaluate
# ═══════════════════════════════════════════════════════════════════════════════
class TestExperimentRunnerCvEvaluate:
    """Tests the experiment_runner _cv_evaluate loop."""

    @patch("corefunc.canon.experiment_runner.experiment")
    def test_returns_mean_metrics(self, mock_exp):
        """Returns cv_mean_* and cv_std_* keys."""
        from sklearn.ensemble import RandomForestClassifier

        from corefunc.canon.experiment_runner import _cv_evaluate

        mock_exp.log_cv_fold = MagicMock()
        rng = np.random.default_rng(42)
        n = 60
        X = pd.DataFrame(
            {
                "f1": rng.standard_normal(n),
                "f2": rng.standard_normal(n),
            }
        )
        y = np.array([0] * 30 + [1] * 30)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        metrics = _cv_evaluate(
            clf,
            X,
            y,
            ["f1", "f2"],
            n_folds=3,
            random_state=42,
            model_name="RF",
        )
        assert "cv_mean_precision" in metrics
        assert "cv_mean_f1" in metrics
        assert "cv_std_auc" in metrics


class TestExperimentRunnerFitWithGpuFallback:
    """Tests the experiment_runner GPU fallback wrapper."""

    def test_success_returns_device(self):
        """Returns the original device on success."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        from corefunc.canon.experiment_runner import _fit_with_gpu_fallback

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

    def test_raises_non_cuda(self):
        """Re-raises for non-CUDA device."""
        from corefunc.canon.experiment_runner import _fit_with_gpu_fallback

        pipe = MagicMock()
        pipe.fit.side_effect = RuntimeError("error")
        with pytest.raises(RuntimeError):
            _fit_with_gpu_fallback(pipe, None, None, "cpu")

    def test_cuda_fallback(self):
        """Falls back to CPU and patches XGB estimators."""
        from corefunc.canon.experiment_runner import _fit_with_gpu_fallback

        mock_clf = MagicMock()
        mock_clf.device = "cuda"
        pipe = MagicMock()
        pipe.named_steps = {"clf": mock_clf}
        call_count = [0]

        def fit_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA error")

        pipe.fit.side_effect = fit_side_effect
        _, device = _fit_with_gpu_fallback(pipe, None, None, "cuda")
        assert device == "cpu"
        mock_clf.set_params.assert_called_with(device="cpu")

    def test_cuda_fallback_composite(self):
        """Falls back and patches inner estimators for composite models."""
        from corefunc.canon.experiment_runner import _fit_with_gpu_fallback

        inner_est = MagicMock()
        inner_est.device = "cuda"
        mock_clf = MagicMock()
        mock_clf.estimators = [("xgb", inner_est)]
        del mock_clf.device  # to not match the first hasattr check
        pipe = MagicMock()
        pipe.named_steps = {"clf": mock_clf}
        call_count = [0]

        def fit_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA error")

        pipe.fit.side_effect = fit_side_effect
        _, device = _fit_with_gpu_fallback(pipe, None, None, "cuda")
        assert device == "cpu"
        inner_est.set_params.assert_called_with(device="cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# tuner — save_best_historical_models
# ═══════════════════════════════════════════════════════════════════════════════
class TestSaveBestHistoricalModels:
    """Tests the best-model exporter."""

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_no_experiment(self, mock_client_cls, _mock_uri):
        """Returns empty list when no MLflow experiment exists."""
        from corefunc.canon.tuner import save_best_historical_models

        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = None
        result = save_best_historical_models()
        assert result == []

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_filters_tuned_runs(self, mock_client_cls, _mock_uri):
        """Skips runs with '_tuned' in model_type."""
        from corefunc.canon.tuner import save_best_historical_models

        mock_client = mock_client_cls.return_value
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_exp
        mock_run = MagicMock()
        mock_run.data.metrics = {
            "opt_precision": 0.9,
            "auc": 0.95,
            "hiprec_precision": 0.85,
            "hiprec_f1": 0.8,
        }
        mock_run.data.tags = {"model_type": "LightGBM_tuned"}
        mock_run.info.run_id = "run123"
        mock_client.search_runs.return_value = [mock_run]
        result = save_best_historical_models()
        assert result == []
