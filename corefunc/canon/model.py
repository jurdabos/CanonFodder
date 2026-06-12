"""
Provides the LightGBM-based canonisation model pipeline.

Functions
---------
train_model()   – loads avc.parquet, computes fuzzy features, trains, saves.
evaluate()      – computes classification report + AUC on a held-out set.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from helpers import cluster, experiment, stats
from helpers.features import compute_pair_features
from helpers.io import AVC_PQ, GS_MB_PQ, read_parquet

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "ML"
MODEL_PATH = MODEL_DIR / "lgbm_legacy.pkl"
COLUMNS_PATH = MODEL_DIR / "lgbm_columns.json"


def _build_gold_standard(augment: bool = False) -> pd.DataFrame:
    """
    Loads avc.parquet, expands pairs, and computes fuzzy-score features.

    When augment is True and gs_mb.parquet exists, merges MBDB-sourced pairs
    into the training set before computing features.
    """
    gs = read_parquet(AVC_PQ)
    if gs is None or gs.empty:
        raise FileNotFoundError("avc.parquet not found or empty — run 'c9r canon avc seed' first.")
    # Expanding into pairwise rows
    if {"variant_a", "variant_b"}.issubset(gs.columns) is False:
        rows = []
        for _, row in gs.iterrows():
            rows.extend(cluster.expand_pairs(row))
        gs = pd.DataFrame(rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    # Merging MBDB augmentation pairs if requested
    n_mb_pairs = 0
    if augment:
        mb = read_parquet(GS_MB_PQ)
        if mb is not None and not mb.empty:
            mb = mb[["variant_a", "variant_b", "to_link"]].copy()
            mb = mb.dropna(subset=["variant_a", "variant_b"])
            # Synthesising a variants column for length_stats compatibility
            mb["variants"] = mb["variant_a"] + "{" + mb["variant_b"]
            n_mb_pairs = len(mb)
            gs = pd.concat([gs, mb], ignore_index=True)
            log.info("Merged %d MBDB pairs into gold standard (total: %d).", n_mb_pairs, len(gs))
        else:
            log.warning("augment=True but gs_mb.parquet not found — run 'c9r canon avc augment'.")
    # Computing three-tier pairwise features
    feature_cols = set(compute_pair_features("a", "b").keys())
    if feature_cols.difference(gs.columns):
        feat_df = gs.apply(
            lambda r: pd.Series(compute_pair_features(r["variant_a"], r["variant_b"])),
            axis=1,
        )
        for col in feat_df.columns:
            if col not in gs.columns:
                gs[col] = feat_df[col]
    # Adding engineered length features
    gs = pd.concat([gs, gs["variants"].apply(stats.length_stats)], axis=1)
    return gs


def train_model(
    *,
    test_size: float = 0.25,
    random_state: int = 47,
    run_name: str | None = None,
    augment: bool = False,
) -> Pipeline:
    """
    Trains a LightGBM classifier on the gold-standard pairs in avc.parquet.

    When augment is True, also includes pairs from gs_mb.parquet.
    Saves the model to ML/lgbm_legacy.pkl and columns to ML/lgbm_columns.json.
    Logs parameters, metrics, and artefacts to MLflow.
    Returns the fitted sklearn Pipeline.
    """
    experiment.init_experiment()
    gs = _build_gold_standard(augment=augment)
    target = "to_link"
    num_cols = [c for c in gs.columns if c not in ["variants", target, "variant_a", "variant_b"]]
    X = gs[num_cols]
    y = gs[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    # Defining LightGBM hyperparameters
    lgbm_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "is_unbalance": True,
        "random_state": 49,
        "n_jobs": -1,
        "verbosity": -1,
    }
    # Building pipeline: RobustScaler -> LightGBM
    pre = ColumnTransformer([("num", Pipeline([("scaler", RobustScaler())]), num_cols)], remainder="drop")
    clf = LGBMClassifier(**lgbm_params)
    model = Pipeline([("prep", pre), ("clf", clf)])
    with experiment.start_run(run_name=run_name):
        # Logging training parameters
        experiment.log_params(
            {
                "test_size": test_size,
                "random_state": random_state,
                "augment": augment,
                "n_features": len(num_cols),
                "n_total_pairs": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "pos_train": int(y_train.sum()),
                "neg_train": int((y_train == 0).sum()),
                **{k: v for k, v in lgbm_params.items() if k != "n_jobs"},
            }
        )
        model.fit(X_train, y_train)
        # Evaluating on the held-out set
        metrics = evaluate(model, X_test, y_test)
        experiment.log_metrics(metrics)
        # Saving model artefacts to disk
        MODEL_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        COLUMNS_PATH.write_text(json.dumps(num_cols, indent=2))
        # Logging artefacts and model to MLflow
        experiment.log_artifact(MODEL_PATH)
        experiment.log_artifact(COLUMNS_PATH)
        experiment.log_model(model)
        log.info("Model saved to %s and logged to MLflow.", MODEL_PATH)
    return model


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Computes classification metrics for the held-out test set.

    Returns a dict with precision, recall, f1, and auc suitable for MLflow logging.
    Also prints the full classification report to stdout.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": auc,
    }
    print("\n=== LightGBM report (held-out) ===")
    print(classification_report(y_test, y_pred, target_names=["no link", "link"]))
    print(f"AUC: {auc:.3f}")
    return metrics
