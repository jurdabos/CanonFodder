"""
Unit tests for corefunc.canon — gold standard builder, XGBoost pipeline.
"""
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from corefunc.canon import _build_gold_standard, evaluate, train_model


def _make_avc(n: int = 40) -> pd.DataFrame:
    """Builds a minimal avc-style DataFrame with n rows of variant pairs."""
    rows = []
    for i in range(n):
        a = f"Artist {i}"
        b = f"Artis {i}" if i % 2 == 0 else f"Unrelated {i}"
        rows.append({
            "artist_variants": f"{a}{{{b}",
            "canonical_name": a,
            "to_link": i % 2 == 0,
            "comment": "",
        })
    return pd.DataFrame(rows)


class TestBuildGoldStandard:
    """Tests the gold-standard builder."""

    @patch("corefunc.canon.read_parquet")
    def test_builds_feature_matrix(self, mock_read):
        """Returns a DataFrame with fuzzy-score and length features."""
        mock_read.return_value = _make_avc(40)
        gs = _build_gold_standard()
        assert "to_link" in gs.columns
        assert "ratio" in gs.columns
        assert "avg_name_len" in gs.columns
        assert len(gs) > 0

    @patch("corefunc.canon.read_parquet", return_value=None)
    def test_raises_on_empty_avc(self, mock_read):
        """Raises FileNotFoundError when avc.parquet is empty."""
        with pytest.raises(FileNotFoundError, match="avc.parquet"):
            _build_gold_standard()

    @patch("corefunc.canon.read_parquet")
    def test_raises_on_empty_df(self, mock_read):
        """Raises FileNotFoundError when the DataFrame is empty."""
        mock_read.return_value = pd.DataFrame()
        with pytest.raises(FileNotFoundError, match="avc.parquet"):
            _build_gold_standard()


class TestTrainModel:
    """Tests the full train_model pipeline."""

    @patch("corefunc.canon.read_parquet")
    def test_trains_and_returns_pipeline(self, mock_read, monkeypatch, tmp_path):
        """Trains a model and returns a Pipeline object."""
        mock_read.return_value = _make_avc(60)
        ml_dir = tmp_path / "ML"
        ml_dir.mkdir()
        monkeypatch.setattr("corefunc.canon.MODEL_DIR", ml_dir)
        monkeypatch.setattr("corefunc.canon.MODEL_PATH", ml_dir / "xgb.json")
        monkeypatch.setattr("corefunc.canon.COLUMNS_PATH", ml_dir / "xgb_columns.json")
        from sklearn.pipeline import Pipeline
        model = train_model(test_size=0.3, random_state=42)
        assert isinstance(model, Pipeline)
        assert "xgb" in model.named_steps
        assert (ml_dir / "xgb.json").exists()


class TestEvaluate:
    """Tests the evaluate reporting function."""

    def test_prints_report(self, capsys):
        """Prints classification report and AUC score."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        from xgboost import XGBClassifier
        from sklearn.compose import ColumnTransformer
        # Building a tiny trained model
        rng = np.random.default_rng(99)
        X = pd.DataFrame({"f1": rng.random(20), "f2": rng.random(20)})
        y = pd.Series([0] * 10 + [1] * 10)
        pre = ColumnTransformer([("num", RobustScaler(), ["f1", "f2"])], remainder="drop")
        xgb = XGBClassifier(n_estimators=5, random_state=42, eval_metric="logloss")
        model = Pipeline([("prep", pre), ("xgb", xgb)])
        model.fit(X, y)
        evaluate(model, X, y)
        captured = capsys.readouterr()
        assert "XGBoost report" in captured.out
        assert "AUC" in captured.out
