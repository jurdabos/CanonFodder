"""
Provides the XGBoost-based canonisation model pipeline.

Functions
---------
train_model()   – loads avc.parquet, computes fuzzy features, trains, saves.
evaluate()      – prints classification report + AUC on a held-out set.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import process
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from helpers.io import AVC_PQ, read_parquet
from helpers import cluster, stats

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ML"
MODEL_PATH = MODEL_DIR / "xgb.json"
COLUMNS_PATH = MODEL_DIR / "xgb_columns.json"


def _build_gold_standard() -> pd.DataFrame:
    """Loads avc.parquet, expands pairs, and computes fuzzy-score features."""
    gs = read_parquet(AVC_PQ)
    if gs is None or gs.empty:
        raise FileNotFoundError("avc.parquet not found or empty — run 'c9r review' first.")
    # Expanding into pairwise rows
    if {"variant_a", "variant_b"}.issubset(gs.columns) is False:
        rows = []
        for _, row in gs.iterrows():
            rows.extend(cluster.expand_pairs(row))
        gs = pd.DataFrame(rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    # Computing fuzzy-score features
    score_cols = {"ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio", "QRatio"}
    if score_cols.difference(gs.columns):
        scores = gs.apply(
            lambda r: pd.Series(cluster.fuzzy_scores(r["variant_a"], r["variant_b"])),
            axis=1,
        )
        for col in score_cols:
            if col not in gs.columns:
                gs[col] = scores[col]
    # Adding engineered length features
    gs = pd.concat([gs, gs["variants"].apply(stats.length_stats)], axis=1)
    return gs


def train_model(*, test_size: float = 0.25, random_state: int = 47) -> Pipeline:
    """
    Trains an XGBoost classifier on the gold-standard pairs in avc.parquet.

    Saves the model to ML/xgb.json and columns to ML/xgb_columns.json.
    Returns the fitted sklearn Pipeline.
    """
    gs = _build_gold_standard()
    target = "to_link"
    num_cols = [c for c in gs.columns if c not in ["variants", target, "variant_a", "variant_b"]]
    X = gs[num_cols]
    y = gs[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    # Building pipeline: RobustScaler -> XGBoost
    pre = ColumnTransformer([("num", Pipeline([("scaler", RobustScaler())]), num_cols)], remainder="drop")
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.75,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
        eval_metric="logloss",
        random_state=49,
        n_jobs=-1,
    )
    model = Pipeline([("prep", pre), ("xgb", xgb)])
    model.fit(X_train, y_train)
    # Evaluating on the held-out set
    evaluate(model, X_test, y_test)
    # Saving
    MODEL_DIR.mkdir(exist_ok=True)
    model.named_steps["xgb"].save_model(MODEL_PATH)
    COLUMNS_PATH.write_text(json.dumps(num_cols, indent=2))
    log.info("Model saved to %s", MODEL_PATH)
    return model


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Prints classification report and ROC-AUC for the held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n=== XGBoost report (held-out) ===")
    print(classification_report(y_test, y_pred, target_names=["no link", "link"]))
    print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
