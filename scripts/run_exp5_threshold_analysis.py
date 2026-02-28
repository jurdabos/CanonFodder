"""
Runs Experiment 5b: threshold-tuned holdout experiment.

Reuses the band-filtered data from Exp 5, but instead of the default 0.5
threshold, sweeps thresholds to find the optimal F1 for each model on the
AVC holdout set.  Reports metrics at both default and optimal thresholds.
"""
from __future__ import annotations
import logging
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp5_filtered import build_filtered_test, build_filtered_train, _add_features
from helpers.device import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Finds the threshold that maximises F1 on the given labels/probabilities."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns n+1 precisions/recalls; thresholds has n
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Computes classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def main():
    """Runs threshold-tuned evaluation on the band-filtered holdout data."""
    log.info("=== Experiment 5b: threshold-tuned holdout ===")
    # Building filtered datasets
    train_raw = build_filtered_train()
    test_raw = build_filtered_test()
    # Computing features
    log.info("Computing features for %d training pairs...", len(train_raw))
    train_df = _add_features(train_raw)
    log.info("Computing features for %d test pairs...", len(test_raw))
    test_df = _add_features(test_raw)
    # Preparing data
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    num_cols = [
        c for c in train_df.columns
        if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info("Train: %d | Test: %d | Features: %d | spw: %.2f", len(X_train), len(X_test), len(num_cols), spw)
    log.info("Test distribution: pos=%d, neg=%d (%.1f%% positive)", y_test.sum(), (y_test == 0).sum(), 100 * y_test.mean())
    # Building model catalogue
    from corefunc.canon.experiment_runner import _build_model_catalogue
    catalogue = _build_model_catalogue(spw, device, random_state=47)
    # Training and evaluating each model
    results = []
    for model_name, clf in catalogue.items():
        log.info("─── Training %s ───", model_name)
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
        # Getting probabilities
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        # Evaluating at default threshold
        default_metrics = _evaluate_at_threshold(y_test, y_prob, 0.5)
        # Finding optimal threshold
        opt_thr, opt_f1 = _optimal_threshold(y_test, y_prob)
        optimal_metrics = _evaluate_at_threshold(y_test, y_prob, opt_thr)
        # Finding conservative (high precision) threshold: precision >= 0.80
        best_hi_prec = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for t in np.arange(0.50, 0.99, 0.01):
            m = _evaluate_at_threshold(y_test, y_prob, t)
            if m["precision"] >= 0.80 and m["f1"] > best_hi_prec["f1"]:
                best_hi_prec = m
        results.append({
            "model": model_name,
            "auc": auc,
            "default_f1": default_metrics["f1"],
            "default_prec": default_metrics["precision"],
            "default_rec": default_metrics["recall"],
            "opt_thr": optimal_metrics["threshold"],
            "opt_f1": optimal_metrics["f1"],
            "opt_prec": optimal_metrics["precision"],
            "opt_rec": optimal_metrics["recall"],
            "hiprec_thr": best_hi_prec["threshold"],
            "hiprec_f1": best_hi_prec["f1"],
            "hiprec_prec": best_hi_prec["precision"],
            "hiprec_rec": best_hi_prec["recall"],
        })
        log.info(
            "%s → AUC=%.4f | default F1=%.4f | optimal F1=%.4f (thr=%.3f) | hi-prec F1=%.4f (thr=%.3f, P=%.3f)",
            model_name, auc,
            default_metrics["f1"],
            optimal_metrics["f1"], opt_thr,
            best_hi_prec["f1"], best_hi_prec["threshold"], best_hi_prec["precision"],
        )
        # Printing classification report at optimal threshold
        y_pred_opt = (y_prob >= opt_thr).astype(int)
        print(f"\n=== {model_name} (optimal thr={opt_thr:.3f}) ===")
        print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
    # Printing summary table
    print("\n" + "=" * 120)
    print(f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | {'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | {'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x["opt_f1"], reverse=True):
        print(
            f"{r['model']:<22} {r['auc']:>6.4f} | "
            f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} {r['default_f1']:>6.4f} | "
            f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} {r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
            f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} {r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}"
        )
    print("=" * 120)
    return results


if __name__ == "__main__":
    main()
