"""
FastAPI server for serving the artist-name canonisation model.

Endpoints
---------
GET  /health          – liveness / readiness probe
POST /predict         – single-pair prediction
POST /predict_batch   – batch prediction
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from importlib.metadata import version as pkg_version
from helpers.inference import load_model, MODEL_PATH

logger = logging.getLogger(__name__)
app = FastAPI(title="c9r model server", version=pkg_version("c9r"))
_model = None
_columns: Optional[List[str]] = None


@app.on_event("startup")
def _startup() -> None:
    """Loads the LightGBM pipeline when the server starts."""
    global _model, _columns
    try:
        _model = load_model()
        _columns = list(_model.feature_names_in_)
        logger.info("Model loaded: %d features.", len(_columns))
    except FileNotFoundError as exc:
        logger.warning("Model not available at startup: %s", exc)


def _preprocess(data: Dict[str, Union[float, int]]) -> pd.DataFrame:
    """Builds a single-row DataFrame with the expected columns."""
    df = pd.DataFrame([data])
    for col in _columns or []:
        if col not in df.columns:
            df[col] = 0.0
    return df[_columns] if _columns else df


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Represents a single prediction request."""
    data: Dict[str, Union[float, int]]


class PredictBatchRequest(BaseModel):
    """Represents a batch prediction request."""
    data: List[Dict[str, Union[float, int]]]


class PredictResponse(BaseModel):
    """Represents a single prediction response."""
    prediction: int
    probability: float
    should_link: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    """Returns server health status."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Returns a single link/no-link prediction."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    df = _preprocess(req.data)
    pred = int(_model.predict(df)[0])
    prob = float(_model.predict_proba(df)[0, 1])
    return PredictResponse(prediction=pred, probability=prob, should_link=bool(pred))


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest) -> dict:
    """Returns batch link/no-link predictions."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = []
    for item in req.data:
        df = _preprocess(item)
        pred = int(_model.predict(df)[0])
        prob = float(_model.predict_proba(df)[0, 1])
        results.append({"prediction": pred, "probability": prob, "should_link": bool(pred)})
    return {"results": results}
