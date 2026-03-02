"""
Unit tests for corefunc.model_server (FastAPI model serving layer).

Tests all functions directly without going through the HTTP transport,
so the httpx optional dependency is not required.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi import HTTPException
import corefunc.model_server as srv
from corefunc.model_server import (
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
    _preprocess,
    health,
    predict,
    predict_batch,
)


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Ensures the module-level model state is clean for every test."""
    srv._model = None
    srv._columns = None
    yield
    srv._model = None
    srv._columns = None


def _make_mock_model(pred: int = 1, prob: float = 0.8):
    """Returns a MagicMock that behaves like a fitted sklearn Pipeline."""
    model = MagicMock()
    model.predict.return_value = np.array([pred])
    model.predict_proba.return_value = np.array([[1.0 - prob, prob]])
    model.feature_names_in_ = ["ratio", "partial_ratio", "WRatio"]
    return model


# ── _startup ──────────────────────────────────────────────────────────────────
class TestStartup:
    """Tests for the on-startup model loading hook."""

    def test_loads_model_successfully(self, monkeypatch):
        """Sets _model and _columns when load_model succeeds."""
        fake_model = _make_mock_model()
        monkeypatch.setattr(srv, "load_model", lambda: fake_model)
        srv._startup()
        assert srv._model is fake_model
        assert srv._columns == ["ratio", "partial_ratio", "WRatio"]

    def test_handles_missing_model_gracefully(self, monkeypatch):
        """Leaves _model as None when the pickle file is absent."""
        monkeypatch.setattr(
            srv,
            "load_model",
            MagicMock(side_effect=FileNotFoundError("not found")),
        )
        srv._startup()
        assert srv._model is None
        assert srv._columns is None


# ── _preprocess ───────────────────────────────────────────────────────────────
class TestPreprocess:
    """Tests for the single-row DataFrame builder."""

    def test_returns_dataframe_with_correct_columns(self):
        """Returns a DataFrame matching _columns order."""
        srv._columns = ["ratio", "partial_ratio", "WRatio"]
        df = _preprocess({"ratio": 0.9, "partial_ratio": 0.85, "WRatio": 0.7})
        assert list(df.columns) == ["ratio", "partial_ratio", "WRatio"]
        assert df["ratio"].iloc[0] == pytest.approx(0.9)

    def test_fills_missing_columns_with_zero(self):
        """Fills columns absent from the input with 0.0."""
        srv._columns = ["ratio", "partial_ratio", "WRatio"]
        df = _preprocess({"ratio": 0.9})
        assert df["partial_ratio"].iloc[0] == 0.0
        assert df["WRatio"].iloc[0] == 0.0

    def test_drops_extra_columns(self):
        """Returns only the columns in _columns, ignoring extras."""
        srv._columns = ["ratio"]
        df = _preprocess({"ratio": 0.9, "extra_col": 42.0})
        assert list(df.columns) == ["ratio"]
        assert "extra_col" not in df.columns

    def test_none_columns_returns_as_is(self):
        """Returns the raw DataFrame when _columns is None."""
        srv._columns = None
        df = _preprocess({"some_feat": 1.5})
        assert "some_feat" in df.columns
        assert len(df) == 1

    def test_empty_data(self):
        """Handles empty data dict without raising."""
        srv._columns = ["ratio"]
        df = _preprocess({})
        assert df["ratio"].iloc[0] == 0.0

    def test_single_row_output(self):
        """Always returns exactly one row."""
        srv._columns = ["a", "b"]
        df = _preprocess({"a": 1, "b": 2})
        assert len(df) == 1


# ── health ────────────────────────────────────────────────────────────────────
class TestHealth:
    """Tests for the /health endpoint function."""

    def test_model_not_loaded(self):
        """Returns model_loaded=False when no model is set."""
        result = health()
        assert result["status"] == "ok"
        assert result["model_loaded"] is False

    def test_model_loaded(self):
        """Returns model_loaded=True when a model is set."""
        srv._model = _make_mock_model()
        result = health()
        assert result["status"] == "ok"
        assert result["model_loaded"] is True

    def test_includes_model_path(self):
        """Returns the MODEL_PATH string in the response."""
        result = health()
        assert "model_path" in result
        assert isinstance(result["model_path"], str)


# ── predict ───────────────────────────────────────────────────────────────────
class TestPredict:
    """Tests for the /predict endpoint function."""

    def test_raises_503_without_model(self):
        """Raises HTTPException 503 when _model is None."""
        req = PredictRequest(data={"ratio": 0.9})
        with pytest.raises(HTTPException) as exc_info:
            predict(req)
        assert exc_info.value.status_code == 503
        assert "Model not loaded" in exc_info.value.detail

    def test_returns_predict_response(self):
        """Returns a PredictResponse with the expected fields."""
        srv._model = _make_mock_model(pred=1, prob=0.8)
        srv._columns = ["ratio", "partial_ratio", "WRatio"]
        req = PredictRequest(data={"ratio": 0.9, "partial_ratio": 0.85, "WRatio": 0.7})
        result = predict(req)
        assert isinstance(result, PredictResponse)
        assert result.prediction == 1
        assert result.probability == pytest.approx(0.8)
        assert result.should_link is True

    def test_prediction_zero_means_no_link(self):
        """Returns should_link=False when prediction is 0."""
        srv._model = _make_mock_model(pred=0, prob=0.3)
        srv._columns = ["ratio"]
        req = PredictRequest(data={"ratio": 0.5})
        result = predict(req)
        assert result.prediction == 0
        assert result.should_link is False

    def test_fills_missing_features(self):
        """Preprocesses correctly when request omits some features."""
        srv._model = _make_mock_model(pred=1, prob=0.9)
        srv._columns = ["ratio", "partial_ratio", "WRatio"]
        req = PredictRequest(data={"ratio": 0.9})
        result = predict(req)
        assert isinstance(result, PredictResponse)
        # Verifying the model was called with all columns filled
        call_df = srv._model.predict.call_args[0][0]
        assert list(call_df.columns) == ["ratio", "partial_ratio", "WRatio"]

    def test_probability_in_valid_range(self):
        """Returns a probability between 0.0 and 1.0."""
        srv._model = _make_mock_model(pred=1, prob=0.65)
        srv._columns = ["ratio"]
        req = PredictRequest(data={"ratio": 0.7})
        result = predict(req)
        assert 0.0 <= result.probability <= 1.0


# ── predict_batch ─────────────────────────────────────────────────────────────
class TestPredictBatch:
    """Tests for the /predict_batch endpoint function."""

    def test_raises_503_without_model(self):
        """Raises HTTPException 503 when _model is None."""
        req = PredictBatchRequest(data=[{"ratio": 0.9}])
        with pytest.raises(HTTPException) as exc_info:
            predict_batch(req)
        assert exc_info.value.status_code == 503

    def test_returns_correct_count(self):
        """Returns one result per input item."""
        srv._model = _make_mock_model(pred=0, prob=0.3)
        srv._columns = ["ratio"]
        req = PredictBatchRequest(data=[{"ratio": 0.5}, {"ratio": 0.9}, {"ratio": 0.1}])
        result = predict_batch(req)
        assert len(result["results"]) == 3

    def test_empty_batch(self):
        """Returns empty results list for an empty batch."""
        srv._model = _make_mock_model()
        srv._columns = ["ratio"]
        req = PredictBatchRequest(data=[])
        result = predict_batch(req)
        assert result["results"] == []

    def test_result_item_shape(self):
        """Each result item has prediction, probability, and should_link keys."""
        srv._model = _make_mock_model(pred=1, prob=0.75)
        srv._columns = ["ratio"]
        req = PredictBatchRequest(data=[{"ratio": 0.8}])
        result = predict_batch(req)
        item = result["results"][0]
        assert "prediction" in item
        assert "probability" in item
        assert "should_link" in item
        assert item["prediction"] == 1
        assert item["should_link"] is True

    def test_mixed_predictions(self):
        """Handles varying predictions across batch items."""
        model = MagicMock()
        model.predict.side_effect = [np.array([0]), np.array([1])]
        model.predict_proba.side_effect = [
            np.array([[0.8, 0.2]]),
            np.array([[0.1, 0.9]]),
        ]
        model.feature_names_in_ = ["ratio"]
        srv._model = model
        srv._columns = ["ratio"]
        req = PredictBatchRequest(data=[{"ratio": 0.3}, {"ratio": 0.95}])
        result = predict_batch(req)
        assert result["results"][0]["should_link"] is False
        assert result["results"][1]["should_link"] is True


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class TestPydanticSchemas:
    """Tests for the Pydantic request/response models."""

    def test_predict_request_accepts_valid_data(self):
        """PredictRequest accepts a dict of floats/ints."""
        req = PredictRequest(data={"ratio": 0.9, "count": 5})
        assert req.data["ratio"] == 0.9

    def test_predict_batch_request_accepts_list(self):
        """PredictBatchRequest accepts a list of dicts."""
        req = PredictBatchRequest(data=[{"a": 1.0}, {"a": 2.0}])
        assert len(req.data) == 2

    def test_predict_response_fields(self):
        """PredictResponse stores prediction, probability, and should_link."""
        resp = PredictResponse(prediction=1, probability=0.85, should_link=True)
        assert resp.prediction == 1
        assert resp.probability == 0.85
        assert resp.should_link is True

    def test_predict_response_no_link(self):
        """PredictResponse correctly represents a no-link prediction."""
        resp = PredictResponse(prediction=0, probability=0.15, should_link=False)
        assert resp.prediction == 0
        assert resp.should_link is False
