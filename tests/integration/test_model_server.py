"""
Integration tests for corefunc.model_server (FastAPI).
"""
import pytest
from unittest.mock import MagicMock
import numpy as np
from fastapi.testclient import TestClient
from corefunc.model_server import app


@pytest.fixture()
def client():
    """Provides a FastAPI TestClient."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Returns status ok regardless of model state."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestPredictEndpoint:
    """Tests the /predict endpoint."""

    def test_predict_without_model(self, client):
        """Returns 503 when model is not loaded."""
        import corefunc.model_server as srv
        srv._model = None
        resp = client.post("/predict", json={"data": {"ratio": 90.0}})
        assert resp.status_code == 503

    def test_predict_with_model(self, client):
        """Returns prediction when model is available."""
        import corefunc.model_server as srv
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        srv._model = mock_model
        srv._columns = ["ratio", "partial_ratio"]
        resp = client.post("/predict", json={"data": {"ratio": 90.0, "partial_ratio": 85.0}})
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 1
        assert body["should_link"] is True
        assert 0.0 <= body["probability"] <= 1.0
        # Cleaning up
        srv._model = None
        srv._columns = None


class TestPredictBatchEndpoint:
    """Tests the /predict_batch endpoint."""

    def test_batch_prediction(self, client):
        """Returns batch results when model is available."""
        import corefunc.model_server as srv
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        srv._model = mock_model
        srv._columns = ["ratio"]
        payload = {"data": [{"ratio": 50.0}, {"ratio": 95.0}]}
        resp = client.post("/predict_batch", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 2
        # Cleaning up
        srv._model = None
        srv._columns = None
