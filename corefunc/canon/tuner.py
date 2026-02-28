"""
Optuna hyperparameter tuning for canonisation models.

Searches over LightGBM, XGBoost, and ExtraTrees with a precision-biased
objective that penalises any CV fold dropping below a configurable
precision floor.  After tuning, retrains the best configuration with
full k-fold CV and logs to MLflow in the same format as ``c9r train``.
"""
from __future__ import annotations
import json
import logging
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from corefunc.canon.experiment_runner import _safe_get_params
from corefunc.canon.trainer import (
    RANDOM_STATE,
    _eval_at,
    _fit_with_gpu_fallback,
    _high_precision_threshold,
    _next_experiment_number,
    _optimal_threshold,
    build_training_data,
    compute_all_features,
    prune_feature_columns,
    verify_mlflow,
    _load_catalogue_lookups,
)
from helpers import experiment
from helpers.device import get_device

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ML"
ML_DIR.mkdir(exist_ok=True)
_TUNABLE_MODELS = {"LightGBM", "XGBoost", "ExtraTrees"}
_DEFAULT_TUNE_MODELS = ["LightGBM"]


# ═════════════════════════════════════════════════════════════════════════════
# Search spaces
# ═════════════════════════════════════════════════════════════════════════════
def _xgb_search_space(
    trial: optuna.Trial, spw: float, device: str,
) -> XGBClassifier:
    """Samples XGBoost hyperparameters from the search space."""
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        scale_pos_weight=trial.suggest_float(
            "scale_pos_weight", max(spw * 0.5, 0.5), spw * 2.0,
        ),
        eval_metric="logloss",
        device=device,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _lgbm_search_space(trial: optuna.Trial, spw: float) -> LGBMClassifier:
    """Samples LightGBM hyperparameters from the search space."""
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 3.0, log=True),
        is_unbalance=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def _et_search_space(trial: optuna.Trial) -> ExtraTreesClassifier:
    """Samples ExtraTrees hyperparameters from the search space."""
    return ExtraTreesClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
        max_depth=trial.suggest_int("max_depth", 4, 16),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 0.8),
        class_weight=trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"],
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Objective function
# ═════════════════════════════════════════════════════════════════════════════
def _precision_biased_objective(
    trial: optuna.Trial,
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    spw: float,
    device: str,
    *,
    n_folds: int = 3,
    min_precision: float = 0.90,
) -> float:
    """Evaluates a trial via k-fold CV with a precision-biased objective.

    Returns ``mean_precision - 0.5 * max(0, min_precision - worst_fold_precision)``.
    This rewards models with consistently high precision across folds.
    """
    # Sampling hyperparameters
    if model_name == "XGBoost":
        clf = _xgb_search_space(trial, spw, device)
    elif model_name == "LightGBM":
        clf = _lgbm_search_space(trial, spw)
    elif model_name == "ExtraTrees":
        clf = _et_search_space(trial)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_precisions: list[float] = []
    fold_f1s: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        pipe = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pipe.fit(X_tr, y_tr)
            except Exception as exc:
                log.warning("Trial %d fold %d failed: %s", trial.number, fold_idx, exc)
                return 0.0
        y_prob = pipe.predict_proba(X_val)[:, 1]
        # Computing precision at the F1-optimal threshold
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_val, y_prob)
        f1s = 2 * (prec_arr[:-1] * rec_arr[:-1]) / (prec_arr[:-1] + rec_arr[:-1] + 1e-12)
        best_idx = np.argmax(f1s)
        opt_thr = float(thr_arr[best_idx])
        y_pred = (y_prob >= opt_thr).astype(int)
        fold_p = precision_score(y_val, y_pred, zero_division=0)
        fold_f = f1_score(y_val, y_pred, zero_division=0)
        fold_precisions.append(fold_p)
        fold_f1s.append(fold_f)
    mean_prec = float(np.mean(fold_precisions))
    worst_prec = float(np.min(fold_precisions))
    # Applying penalty for folds below the precision floor
    penalty = 0.5 * max(0.0, min_precision - worst_prec)
    score = mean_prec - penalty
    # Reporting intermediate metrics for monitoring
    trial.set_user_attr("cv_mean_precision", mean_prec)
    trial.set_user_attr("cv_worst_precision", worst_prec)
    trial.set_user_attr("cv_mean_f1", float(np.mean(fold_f1s)))
    trial.set_user_attr("penalty", penalty)
    return score


# ═════════════════════════════════════════════════════════════════════════════
# Post-tuning: retrain best and log to MLflow
# ═════════════════════════════════════════════════════════════════════════════
def _retrain_best(
    model_name: str,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    num_cols: list[str],
    spw: float,
    device: str,
    exp_num: int,
    *,
    n_folds: int = 5,
) -> dict[str, float]:
    """Retrains the best configuration with full CV and MLflow logging.

    Returns the held-out test metrics dict.
    """
    import mlflow
    # Reconstructing the classifier from best params
    if model_name == "XGBoost":
        clf = XGBClassifier(
            **best_params, eval_metric="logloss", device=device,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    elif model_name == "LightGBM":
        clf = LGBMClassifier(
            **best_params, is_unbalance=True, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=-1,
        )
    elif model_name == "ExtraTrees":
        clf = ExtraTreesClassifier(
            **best_params, random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    with experiment.start_run(run_name=f"{model_name}_tuned", nested=True):
        mlflow.set_tag("model_type", f"{model_name}_tuned")
        mlflow.set_tag("tuning_method", "optuna")
        safe_params = _safe_get_params(clf)
        safe_params["device_used"] = device
        experiment.log_params(safe_params)
        # Running full k-fold CV with per-fold tracking
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            pre = ColumnTransformer(
                [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
                remainder="drop",
                verbose_feature_names_out=False,
            )
            pre.set_output(transform="pandas")
            fold_pipe = Pipeline([("prep", pre), ("clf", clone(clf))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_pipe.fit(X_tr, y_tr)
            y_prob = fold_pipe.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            y_pred = fold_pipe.predict(X_val)
            metrics = {
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "auc": auc,
            }
            fold_metrics.append(metrics)
            experiment.log_cv_fold(
                fold_idx, metrics, run_name_prefix=f"{model_name}_tuned_fold",
            )
        # Logging CV aggregates
        cv_agg: dict[str, float] = {}
        for key in fold_metrics[0]:
            vals = [m[key] for m in fold_metrics]
            cv_agg[f"cv_mean_{key}"] = float(np.mean(vals))
            cv_agg[f"cv_std_{key}"] = float(np.std(vals))
        experiment.log_metrics(cv_agg)
        # Training final model on full training set
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        final_pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipeline, actual_device = _fit_with_gpu_fallback(
                final_pipeline, X_train, y_train, device,
            )
        if actual_device != device:
            experiment.log_params({"device_fallback": actual_device})
        # Evaluating at 3 operating points on held-out test
        y_prob = final_pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        default_m = _eval_at(y_test, y_prob, 0.5)
        opt_thr, _ = _optimal_threshold(y_test, y_prob)
        optimal_m = _eval_at(y_test, y_prob, opt_thr)
        hiprec_m = _high_precision_threshold(y_test, y_prob)
        experiment.log_metrics({
            "auc": auc,
            "default_f1": default_m["f1"],
            "default_precision": default_m["precision"],
            "default_recall": default_m["recall"],
            "opt_threshold": optimal_m["threshold"],
            "opt_f1": optimal_m["f1"],
            "opt_precision": optimal_m["precision"],
            "opt_recall": optimal_m["recall"],
            "hiprec_threshold": hiprec_m["threshold"],
            "hiprec_f1": hiprec_m["f1"],
            "hiprec_precision": hiprec_m["precision"],
            "hiprec_recall": hiprec_m["recall"],
        })
        result = {
            "auc": auc,
            "default_f1": default_m["f1"],
            "default_prec": default_m["precision"],
            "default_rec": default_m["recall"],
            "opt_thr": optimal_m["threshold"],
            "opt_f1": optimal_m["f1"],
            "opt_prec": optimal_m["precision"],
            "opt_rec": optimal_m["recall"],
            "hiprec_thr": hiprec_m["threshold"],
            "hiprec_f1": hiprec_m["f1"],
            "hiprec_prec": hiprec_m["precision"],
            "hiprec_rec": hiprec_m["recall"],
            **cv_agg,
        }
        log.info(
            "%s_tuned → AUC=%.4f | opt P=%.4f F1=%.4f (thr=%.3f)"
            " | hiP P=%.4f F1=%.4f (thr=%.3f)",
            model_name, auc, optimal_m["precision"], optimal_m["f1"],
            opt_thr, hiprec_m["precision"], hiprec_m["f1"],
            hiprec_m["threshold"],
        )
        # Printing classification report
        y_pred_opt = (y_prob >= opt_thr).astype(int)
        print(f"\n=== {model_name}_tuned (optimal thr={opt_thr:.3f}) ===")
        print(classification_report(
            y_test, y_pred_opt, target_names=["no link", "link"],
        ))
        # Logging artefacts
        experiment.log_confusion_matrix(y_test, y_pred_opt)
        experiment.log_feature_importance(final_pipeline, num_cols)
        X_test_transformed = final_pipeline.named_steps["prep"].transform(X_test)
        experiment.log_shap_summary(
            final_pipeline, X_test_transformed, num_cols,
        )
        experiment.log_model(final_pipeline)
        # Saving pickle to ML/
        pkl_path = ML_DIR / f"{model_name.lower()}_tuned.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(final_pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved tuned model to %s", pkl_path)
        mlflow.log_artifact(str(pkl_path))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Saving best historical models
# ═════════════════════════════════════════════════════════════════════════════
def save_best_historical_models() -> list[str]:
    """Exports the best model per type from MLflow as pickles in ``ML/``.

    Scans all runs with 3-operating-point metrics (Exp 14+), picks the
    highest-AUC run per model type, and saves the sklearn pipeline.
    Returns the list of saved file paths.
    """
    import mlflow
    mlflow.set_tracking_uri(experiment.TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment.DEFAULT_EXPERIMENT)
    if exp is None:
        log.warning("No MLflow experiment found — nothing to export.")
        return []
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        max_results=500,
    )
    # Collecting runs with 3-operating-point metrics (from Exp 14+)
    candidates: dict[str, dict[str, Any]] = {}
    for r in runs:
        m = r.data.metrics
        model_type = r.data.tags.get("model_type", "")
        opt_p = m.get("opt_precision", 0)
        auc = m.get("auc", 0)
        hip_p = m.get("hiprec_precision", 0)
        hip_f1 = m.get("hiprec_f1", 0)
        c9r_score = 0.4 * hip_p + 0.3 * hip_f1 + 0.3 * auc
        if not model_type or opt_p == 0 or "_tuned" in model_type:
            continue
        if model_type not in candidates or c9r_score > candidates[model_type]["c9r_score"]:
            candidates[model_type] = {
                "run_id": r.info.run_id,
                "c9r_score": c9r_score,
                "auc": auc,
                "hip_p": hip_p,
                "hip_f1": hip_f1,
            }
    saved: list[str] = []
    for model_type, info in sorted(candidates.items()):
        run_id = info["run_id"]
        try:
            model_uri = f"runs:/{run_id}/model"
            pipeline = mlflow.sklearn.load_model(model_uri)
            pkl_name = f"{model_type.lower()}_best.pkl"
            pkl_path = ML_DIR / pkl_name
            with open(pkl_path, "wb") as f:
                pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved.append(str(pkl_path))
            log.info(
                "Saved %s → %s (c9r=%.4f, AUC=%.4f, HiP_P=%.4f)",
                model_type, pkl_path, info["c9r_score"], info["auc"], info["hip_p"],
            )
        except Exception as exc:
            log.warning("Failed to export %s (run %s): %s", model_type, run_id[:8], exc)
    return saved


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════
def run_tuning(
    *,
    run_name: str | None = None,
    models: list[str] | None = None,
    n_trials: int = 60,
    n_folds: int = 3,
    test_size: float = 0.20,
    min_precision: float = 0.90,
    catalogue: bool = True,
) -> dict[str, dict[str, float]]:
    """Runs Optuna hyperparameter tuning with a precision-biased objective.

    Parameters
    ----------
    run_name : optional MLflow parent run name (auto-generated if None).
    models : model names to tune (default: LightGBM, XGBoost, ExtraTrees).
    n_trials : Optuna trials per model.
    n_folds : CV folds for the tuning inner loop (default 3 for speed).
    test_size : held-out test fraction.
    min_precision : precision floor for the objective penalty.
    catalogue : whether to include catalogue features.

    Returns a dict mapping model_name → held-out test metrics of the tuned model.
    """
    # ── Step 0: Pre-verifying MLflow ───────────────────────────────────────
    verify_mlflow()
    exp_num = _next_experiment_number()
    log.info("Tuning as Experiment %d.", exp_num)
    # ── Step 0b: Saving best historical models ─────────────────────────────
    log.info("Exporting best historical models as pickles...")
    saved_models = save_best_historical_models()
    if saved_models:
        for p in saved_models:
            log.info("  → %s", p)
    else:
        log.info("  No historical models to export.")
    # ── Step 1: Building training data ─────────────────────────────────────
    train_pairs, test_pairs = build_training_data(
        test_size=test_size, random_state=RANDOM_STATE,
    )
    # ── Step 2: Loading catalogue lookups ──────────────────────────────────
    name_to_albums: dict[str, list[str]] | None = None
    name_to_tracks: dict[str, list[str]] | None = None
    if catalogue:
        name_to_albums, name_to_tracks = _load_catalogue_lookups()
    # ── Step 3: Computing features (once, shared across all trials) ────────
    log.info("Computing features for training set...")
    train_df = compute_all_features(
        train_pairs, catalogue=catalogue, cat_design="proportional",
        name_to_albums=name_to_albums, name_to_tracks=name_to_tracks,
    )
    log.info("Computing features for test set...")
    test_df = compute_all_features(
        test_pairs, catalogue=catalogue, cat_design="proportional",
        name_to_albums=name_to_albums, name_to_tracks=name_to_tracks,
    )
    # ── Step 4: Pruning ────────────────────────────────────────────────────
    target = "to_link"
    exclude = {target, "variant_a", "variant_b", "source", "_key"}
    all_num = [
        c for c in train_df.columns
        if c not in exclude
        and train_df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_feature_columns(train_df[all_num])
    missing = [c for c in num_cols if c not in test_df.columns]
    if missing:
        log.warning("Columns missing in test_df (filling with 0): %s", missing)
        for c in missing:
            test_df[c] = 0.0
    log.info("Post-pruning: %d features.", len(num_cols))
    # ── Step 5: Preparing arrays ───────────────────────────────────────────
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info(
        "Train: %d | Test: %d | Features: %d | spw: %.2f | device: %s",
        len(X_train), len(X_test), len(num_cols), spw, device,
    )
    # ── Step 6: Selecting models to tune ───────────────────────────────────
    model_names = models or _DEFAULT_TUNE_MODELS
    model_names = [m for m in model_names if m in _TUNABLE_MODELS]
    if not model_names:
        raise RuntimeError(
            f"No tunable models selected. Choose from: {sorted(_TUNABLE_MODELS)}"
        )
    # ── Step 7: Running Optuna studies ─────────────────────────────────────
    parent_name = run_name or f"exp{exp_num}_optuna_tune"
    experiment.init_experiment()
    results: dict[str, dict[str, float]] = {}
    with experiment.start_run(run_name=parent_name):
        experiment.log_params({
            "experiment": exp_num,
            "experiment_type": "optuna_tuning",
            "random_state": RANDOM_STATE,
            "test_size": test_size,
            "tune_folds": n_folds,
            "n_trials": n_trials,
            "min_precision": min_precision,
            "catalogue_features": catalogue,
            "n_features": len(num_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_pos": int(y_train.sum()),
            "train_neg": int((y_train == 0).sum()),
            "spw": round(spw, 2),
            "device_probed": device,
            "models_tuned": ",".join(model_names),
        })
        for model_name in model_names:
            log.info("═══ Tuning %s (%d trials) ═══", model_name, n_trials)
            # Suppressing Optuna's default logging (we log results ourselves)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                study_name=f"{model_name}_exp{exp_num}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            study.optimize(
                lambda trial: _precision_biased_objective(
                    trial, model_name, X_train, y_train, num_cols,
                    spw, device, n_folds=n_folds, min_precision=min_precision,
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            best = study.best_trial
            log.info(
                "%s best trial #%d: score=%.4f (mean_P=%.4f, worst_P=%.4f, mean_F1=%.4f)",
                model_name, best.number, best.value,
                best.user_attrs.get("cv_mean_precision", 0),
                best.user_attrs.get("cv_worst_precision", 0),
                best.user_attrs.get("cv_mean_f1", 0),
            )
            # Filtering out non-model params before rebuilding the classifier
            skip_keys = {"is_unbalance", "eval_metric", "device",
                         "random_state", "n_jobs", "verbosity"}
            best_params = {
                k: v for k, v in best.params.items() if k not in skip_keys
            }
            log.info("  Best params: %s", best_params)
            # Logging study artefacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Saving trial history as JSON
                history_path = Path(tmpdir) / f"{model_name}_trials.json"
                trials_data = []
                for t in study.trials:
                    trials_data.append({
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "user_attrs": t.user_attrs,
                        "state": t.state.name,
                    })
                history_path.write_text(
                    json.dumps(trials_data, indent=2, default=str),
                )
                experiment.log_artifact(history_path)
                # Saving best params as JSON
                params_path = Path(tmpdir) / f"{model_name}_best_params.json"
                params_path.write_text(
                    json.dumps(best_params, indent=2, default=str),
                )
                experiment.log_artifact(params_path)
            # Retraining with best hyperparams (full 5-fold CV + held-out eval)
            log.info("Retraining %s with tuned hyperparameters...", model_name)
            result = _retrain_best(
                model_name, best_params,
                X_train, y_train, X_test, y_test,
                num_cols, spw, device, exp_num,
                n_folds=5,
            )
            results[model_name] = result
    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print(
        f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
        f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
        f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}"
    )
    print("-" * 130)
    for name in sorted(results, key=lambda k: results[k]["opt_f1"], reverse=True):
        r = results[name]
        print(
            f"{name + '_tuned':<22} {r['auc']:>6.4f} | "
            f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} "
            f"{r['default_f1']:>6.4f} | "
            f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} "
            f"{r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
            f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} "
            f"{r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}"
        )
    print("=" * 130)
    # Selecting best model by c9r composite score (0.4×HiP_P + 0.3×HiP_F1 + 0.3×AUC)
    best_model = max(results, key=lambda k: 0.4 * results[k]["hiprec_prec"] + 0.3 * results[k]["hiprec_f1"] + 0.3 * results[k]["auc"])
    score = 0.4 * results[best_model]["hiprec_prec"] + 0.3 * results[best_model]["hiprec_f1"] + 0.3 * results[best_model]["auc"]
    print(f"\nBest tuned model by c9r score: {best_model}_tuned (score={score:.4f})")
    return results
