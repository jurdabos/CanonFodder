"""
Orchestrates multi-model experiments for artist name variant classification.

Trains and evaluates several base and composite models inside a single
MLflow parent run.  Each candidate gets its own nested child run with
stratified k-fold cross-validation, per-fold metrics, and optional
Optuna-based hyperparameter tuning.
"""
from __future__ import annotations
import logging
import warnings
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from helpers import experiment
from helpers.device import get_device

log = logging.getLogger(__name__)


# ── Model catalogue ───────────────────────────────────────────────────────────
def _build_model_catalogue(
    spw: float,
    device: str,
    random_state: int,
) -> dict[str, Any]:
    """Returns a dict mapping model name → unfitted classifier instance.

    All classifiers are bare (no Pipeline wrapper yet) — the caller wraps
    them in a RobustScaler Pipeline.
    """
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.75,
        scale_pos_weight=spw,
        eval_metric="logloss",
        device=device,
        random_state=random_state,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        is_unbalance=True,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state,
    )
    # CPU variant for composite models (avoids CUDA↔CPU device mismatch
    # during sklearn's internal cross-validation and meta-learner ops)
    xgb_cpu = clone(xgb).set_params(device="cpu")
    # Composite models
    voting = VotingClassifier(
        estimators=[("xgb", xgb_cpu), ("rf", rf), ("lgbm", lgbm)],
        voting="soft",
        n_jobs=-1,
    )
    stacking = StackingClassifier(
        estimators=[("xgb", xgb_cpu), ("rf", rf), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1,
    )
    bagging = BaggingClassifier(
        estimator=XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=spw,
            eval_metric="logloss",
            device="cpu",
            random_state=random_state,
            n_jobs=1,
        ),
        n_estimators=10,
        random_state=random_state,
        n_jobs=-1,
    )
    return {
        "XGBoost": xgb,
        "RandomForest": rf,
        "ExtraTrees": et,
        "LightGBM": lgbm,
        "GradientBoosting": gbm,
        "VotingEnsemble": voting,
        "StackingEnsemble": stacking,
        "BaggingXGB": bagging,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def _evaluate(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Computes classification metrics on a given split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
    auc = roc_auc_score(y, y_prob)
    return {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": auc,
    }


def _safe_get_params(clf) -> dict[str, Any]:
    """Extracts loggable params from a classifier, skipping nested objects."""
    try:
        raw = clf.get_params(deep=False)
    except Exception:
        return {}
    safe: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
    return safe


# ── Cross-validation loop ────────────────────────────────────────────────────
def _cv_evaluate(
    clf,
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    *,
    n_folds: int = 5,
    random_state: int = 47,
    model_name: str = "",
) -> dict[str, float]:
    """Runs stratified k-fold CV and logs per-fold metrics to MLflow.

    Returns the mean metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        # Building a fresh pipeline per fold
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        fold_pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_pipeline.fit(X_tr, y_tr)
        metrics = _evaluate(fold_pipeline, X_val, y_val)
        fold_metrics.append(metrics)
        experiment.log_cv_fold(fold_idx, metrics, run_name_prefix=f"{model_name}_fold")
    # Aggregating
    mean_metrics: dict[str, float] = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        mean_metrics[f"cv_mean_{key}"] = float(np.mean(vals))
        mean_metrics[f"cv_std_{key}"] = float(np.std(vals))
    return mean_metrics


# ── GPU-safe training wrapper ─────────────────────────────────────────────────
def _fit_with_gpu_fallback(pipeline: Pipeline, X, y, device: str) -> tuple[Pipeline, str]:
    """Fits the pipeline; retries on CPU if CUDA fails at runtime."""
    try:
        pipeline.fit(X, y)
        return pipeline, device
    except Exception as exc:
        if device != "cuda":
            raise
        log.warning("CUDA training failed (%s) — retrying on CPU.", exc)
        # Patching XGBoost estimators inside the pipeline to CPU
        clf = pipeline.named_steps.get("clf")
        if clf is not None and hasattr(clf, "device"):
            clf.set_params(device="cpu")
        elif clf is not None and hasattr(clf, "estimators"):
            # Composite — patch any inner XGB estimators
            for name, est in getattr(clf, "estimators", []):
                if hasattr(est, "device"):
                    est.set_params(device="cpu")
        pipeline.fit(X, y)
        return pipeline, "cpu"


# ── Public entry point ────────────────────────────────────────────────────────
def run_experiment(
    *,
    augment: bool = True,
    test_size: float = 0.25,
    n_folds: int = 5,
    random_state: int = 47,
    run_name: str | None = None,
    models: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Trains and evaluates all candidate models, logging to MLflow.

    Parameters
    ----------
    augment : whether to include gs_mb.parquet pairs.
    test_size : fraction held out for final evaluation.
    n_folds : number of CV folds.
    random_state : seed for reproducibility.
    run_name : optional MLflow parent run name.
    models : optional list of model names to run (default: all).

    Returns a dict mapping model_name → held-out test metrics.
    """
    from corefunc.canon.model import _build_gold_standard
    experiment.init_experiment()
    device = get_device()
    # Preparing data
    gs = _build_gold_standard(augment=augment)
    target = "to_link"
    num_cols = [c for c in gs.columns if c not in ["variants", target, "variant_a", "variant_b"]]
    X = gs[num_cols]
    y = gs[target].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info(
        "Experiment data: %d total (%d train, %d test), %d features, spw=%.2f, device=%s",
        len(X), len(X_train), len(X_test), len(num_cols), spw, device,
    )
    catalogue = _build_model_catalogue(spw, device, random_state)
    if models:
        catalogue = {k: v for k, v in catalogue.items() if k in models}
    results: dict[str, dict[str, float]] = {}
    parent_run_name = run_name or "experiment_run"
    with experiment.start_run(run_name=parent_run_name):
        experiment.log_params({
            "augment": augment,
            "test_size": test_size,
            "n_folds": n_folds,
            "random_state": random_state,
            "n_features": len(num_cols),
            "n_total_pairs": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "device_probed": device,
            "model_count": len(catalogue),
        })
        for model_name, clf in catalogue.items():
            log.info("─── Training %s ───", model_name)
            with experiment.start_run(run_name=model_name, nested=True):
                import mlflow
                mlflow.set_tag("model_type", model_name)
                # Logging model-specific params
                safe_params = _safe_get_params(clf)
                safe_params["device_used"] = device
                experiment.log_params(safe_params)
                # Cross-validation
                cv_metrics = _cv_evaluate(
                    clf, X_train, y_train, num_cols,
                    n_folds=n_folds,
                    random_state=random_state,
                    model_name=model_name,
                )
                experiment.log_metrics(cv_metrics)
                # Training final model on full training set
                pre = ColumnTransformer(
                    [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
                    remainder="drop",
                    verbose_feature_names_out=False,
                )
                pre.set_output(transform="pandas")
                final_pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
                final_pipeline, actual_device = _fit_with_gpu_fallback(
                    final_pipeline, X_train, y_train, device,
                )
                if actual_device != device:
                    experiment.log_params({"device_fallback": actual_device})
                # Evaluating on held-out test set
                test_metrics = _evaluate(final_pipeline, X_test, y_test)
                experiment.log_metrics(test_metrics)
                results[model_name] = test_metrics
                # Printing classification report
                y_pred = final_pipeline.predict(X_test)
                print(f"\n=== {model_name} (held-out) ===")
                print(classification_report(y_test, y_pred, target_names=["no link", "link"]))
                print(f"AUC: {test_metrics['auc']:.4f}")
                # Logging artefacts
                experiment.log_confusion_matrix(y_test, y_pred)
                experiment.log_feature_importance(final_pipeline, num_cols)
                # SHAP for tree-based single models (not composites)
                if model_name not in ("VotingEnsemble", "StackingEnsemble", "BaggingXGB"):
                    # Passing pre-processed data to SHAP
                    X_test_transformed = final_pipeline.named_steps["prep"].transform(X_test)
                    experiment.log_shap_summary(final_pipeline, X_test_transformed, num_cols)
                # Logging model
                experiment.log_model(final_pipeline)
                log.info("%s → F1=%.4f, AUC=%.4f", model_name, test_metrics["f1"], test_metrics["auc"])
    # Printing summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{name:<22} {m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")
    print("=" * 70)
    return results


# ── Holdout experiment (explicit train/test DataFrames) ───────────────────────
def run_holdout_experiment(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int = 10,
    random_state: int = 47,
    run_name: str | None = None,
    models: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Trains on train_df and evaluates on a fixed test_df (no random split).

    Designed for experiments where the test set is a hand-curated gold
    standard (e.g. avc.parquet) independent of the training data.
    Both DataFrames must already contain feature columns and 'to_link'.
    """
    experiment.init_experiment()
    device = get_device()
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    num_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")]
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info(
        "Holdout experiment: %d train, %d test, %d features, spw=%.2f, device=%s",
        len(X_train), len(X_test), len(num_cols), spw, device,
    )
    catalogue = _build_model_catalogue(spw, device, random_state)
    if models:
        catalogue = {k: v for k, v in catalogue.items() if k in models}
    results: dict[str, dict[str, float]] = {}
    parent_run_name = run_name or "holdout_experiment"
    with experiment.start_run(run_name=parent_run_name):
        experiment.log_params({
            "experiment_type": "holdout",
            "n_folds": n_folds,
            "random_state": random_state,
            "n_features": len(num_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_source": "avc_hand_curated",
            "device_probed": device,
            "model_count": len(catalogue),
        })
        for model_name, clf in catalogue.items():
            log.info("─── Training %s ───", model_name)
            with experiment.start_run(run_name=model_name, nested=True):
                import mlflow
                mlflow.set_tag("model_type", model_name)
                safe_params = _safe_get_params(clf)
                safe_params["device_used"] = device
                experiment.log_params(safe_params)
                # Cross-validation on the training set
                cv_metrics = _cv_evaluate(
                    clf, X_train, y_train, num_cols,
                    n_folds=n_folds,
                    random_state=random_state,
                    model_name=model_name,
                )
                experiment.log_metrics(cv_metrics)
                # Training final model on full training set
                pre = ColumnTransformer(
                    [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
                    remainder="drop",
                    verbose_feature_names_out=False,
                )
                pre.set_output(transform="pandas")
                final_pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
                final_pipeline, actual_device = _fit_with_gpu_fallback(
                    final_pipeline, X_train, y_train, device,
                )
                if actual_device != device:
                    experiment.log_params({"device_fallback": actual_device})
                # Evaluating on the hand-curated holdout test set
                test_metrics = _evaluate(final_pipeline, X_test, y_test)
                experiment.log_metrics(test_metrics)
                results[model_name] = test_metrics
                y_pred = final_pipeline.predict(X_test)
                print(f"\n=== {model_name} (avc holdout) ===")
                print(classification_report(y_test, y_pred, target_names=["no link", "link"]))
                print(f"AUC: {test_metrics['auc']:.4f}")
                experiment.log_confusion_matrix(y_test, y_pred)
                experiment.log_feature_importance(final_pipeline, num_cols)
                if model_name not in ("VotingEnsemble", "StackingEnsemble", "BaggingXGB"):
                    X_test_transformed = final_pipeline.named_steps["prep"].transform(X_test)
                    experiment.log_shap_summary(final_pipeline, X_test_transformed, num_cols)
                experiment.log_model(final_pipeline)
                log.info("%s → F1=%.4f, AUC=%.4f", model_name, test_metrics["f1"], test_metrics["auc"])
    # Printing summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{name:<22} {m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")
    print("=" * 70)
    return results
