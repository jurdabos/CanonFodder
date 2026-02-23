"""
FastAPI server for serving the XGBoost artist-name canonisation model.

Endpoints
---------
GET  /health          – liveness / readiness probe
POST /predict         – single-pair prediction
POST /predict_batch   – batch prediction
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ML" / "xgb.json"
COLUMNS_PATH = PROJECT_ROOT / "ML" / "xgb_columns.json"
app = FastAPI(title="c9r model server", version="0.6.0")
_model: Optional[XGBClassifier] = None
_columns: Optional[List[str]] = None


def _load_model() -> tuple[XGBClassifier, List[str]]:
    """Loads the XGBoost model and column list from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Columns not found: {COLUMNS_PATH}")
    xgb = XGBClassifier()
    xgb.load_model(MODEL_PATH)
    cols = json.loads(COLUMNS_PATH.read_text())
    logger.info("Model loaded from %s", MODEL_PATH)
    return xgb, cols


@app.on_event("startup")
def _startup() -> None:
    """Loads the model when the server starts."""
    global _model, _columns
    try:
        _model, _columns = _load_model()
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
    return {"status": "ok", "model_loaded": _model is not None}


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
