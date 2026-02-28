"""
Runs Experiment 7b: Negative subsampling from Exp 6 training data.

Loads gs_mb_dbscan.parquet (388K pairs, 180 pos) from Exp 6, subsamples
negatives to various ratios, and re-runs the holdout experiment.
Avoids re-running the expensive DBSCAN + MBDB pipeline.
"""
from __future__ import annotations
import logging
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.io import AVC_PQ, read_parquet # noqa: E402
from helpers import cluster # noqa: E402
from helpers.device import get_device # noqa: E402
from scripts.run_exp6_dbscan import ( # noqa: E402
    add_all_features, prune_features, # noqa: E402
    _optimal_threshold, _evaluate_at_threshold, # noqa: E402
    GS_DBSCAN_PQ, WRATIO_LOWER, WRATIO_UPPER, # noqa: E402
) # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
# Trying multiple neg:pos ratios to find the sweet spot
NEG_RATIOS = [5, 10, 20, 50]


def subsample_negatives(df: pd.DataFrame, ratio: int) -> pd.DataFrame:
    """Subsamples negatives with stratified WRatio band allocation."""
    pos = df[df["to_link"].eq(True)].copy()
    neg = df[df["to_link"].eq(False)].copy()
    n_pos = len(pos)
    target_neg = min(n_pos * ratio, len(neg))
    if len(neg) <= target_neg:
        return df
    # Computing WRatio for stratification
    neg["_wr"] = neg.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    neg["_band"] = pd.cut(neg["_wr"], bins=[60, 70, 80, 90, 95, 100], right=False)
    sampled_parts = []
    band_counts = neg["_band"].value_counts()
    total = band_counts.sum()
    for band, count in band_counts.items():
        band_df = neg[neg["_band"] == band]
        n_sample = max(1, int(round(target_neg * count / total)))
        n_sample = min(n_sample, len(band_df))
        sampled_parts.append(band_df.sample(n=n_sample, random_state=RANDOM_STATE))
    neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_wr", "_band"])
    combined = pd.concat([pos, neg_sampled], ignore_index=True)
    return combined.reset_index(drop=True)


def build_avc_test() -> pd.DataFrame:
    """Builds the AVC test set filtered to WRatio [60, 100)."""
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)].drop(columns=["_wr"])
    return test_raw.reset_index(drop=True)


def run_single_ratio(train_df, test_df, num_cols, ratio_label):
    """Trains all 8 models for a single ratio configuration."""
    from corefunc.canon.experiment_runner import _build_model_catalogue
    target = "to_link"
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info("[%s] Train: %d | Test: %d | spw: %.2f", ratio_label, len(X_train), len(X_test), spw)
    catalogue = _build_model_catalogue(spw, device, random_state=RANDOM_STATE)
    results = []
    for model_name, clf in catalogue.items():
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop", verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        default_m = _evaluate_at_threshold(y_test, y_prob, 0.5)
        opt_thr, _ = _optimal_threshold(y_test, y_prob)
        optimal_m = _evaluate_at_threshold(y_test, y_prob, opt_thr)
        best_hi = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for t in np.arange(0.50, 0.99, 0.01):
            m = _evaluate_at_threshold(y_test, y_prob, t)
            if m["precision"] >= 0.80 and m["f1"] > best_hi["f1"]:
                best_hi = m
        results.append({
            "ratio": ratio_label, "model": model_name, "auc": auc,
            "default_f1": default_m["f1"],
            "opt_thr": optimal_m["threshold"], "opt_f1": optimal_m["f1"],
            "opt_prec": optimal_m["precision"], "opt_rec": optimal_m["recall"],
            "hiprec_thr": best_hi["threshold"], "hiprec_f1": best_hi["f1"],
            "hiprec_prec": best_hi["precision"],
        })
    return results


def main():
    """Runs Experiment 7b across multiple neg:pos ratios."""
    log.info("=== Experiment 7b: Negative subsampling sweep ===")
    # Loading Exp 6 data
    full_train = read_parquet(GS_DBSCAN_PQ)
    n_pos = full_train["to_link"].sum()
    n_neg = len(full_train) - n_pos
    log.info("Exp 6 data: %d pairs (pos=%d, neg=%d, ratio %.0f:1).", len(full_train), n_pos, n_neg, n_neg / max(n_pos, 1))
    # Building AVC test
    test_raw = build_avc_test()
    log.info("AVC test: %d pairs (pos=%d, neg=%d).",
             len(test_raw), test_raw["to_link"].sum(), (test_raw["to_link"].eq(False)).sum())
    # Pre-computing test features once
    log.info("Computing AVC test features...")
    test_df = add_all_features(test_raw)
    # Also including the full (unsubsampled) Exp 6 ratio for comparison
    all_ratios = NEG_RATIOS + [0]  # 0 means "no subsampling" (full Exp 6)
    all_results = []
    for ratio in all_ratios:
        if ratio == 0:
            label = f"full ({n_neg // max(n_pos, 1)}:1)"
            train_sub = full_train.copy()
        else:
            label = f"{ratio}:1"
            train_sub = subsample_negatives(full_train, ratio)
        pos_count = train_sub["to_link"].sum()
        neg_count = len(train_sub) - pos_count
        log.info("── Ratio %s: %d pairs (pos=%d, neg=%d) ──", label, len(train_sub), pos_count, neg_count)
        # Computing features
        train_df = add_all_features(train_sub)
        # Pruning
        target = "to_link"
        exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
        all_num = [c for c in train_df.columns
                   if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")]
        num_cols = prune_features(train_df[all_num])
        missing = [c for c in num_cols if c not in test_df.columns]
        for c in missing:
            test_df[c] = 0.0
        # Running experiment
        results = run_single_ratio(train_df, test_df, num_cols, label)
        all_results.extend(results)
    # Printing cross-ratio comparison (best model per ratio)
    print("\n" + "=" * 100)
    print("CROSS-RATIO COMPARISON (best model per ratio)")
    print("=" * 100)
    print(f"{'Ratio':<15} {'Model':<22} {'AUC':>6} {'Def F1':>7} {'Opt thr':>8} {'Opt F1':>7} {'Opt P':>6} {'Opt R':>6} {'HiP F1':>7} {'HiP P':>6}")
    print("-" * 100)
    ratios_seen = []
    for r in all_results:
        if r["ratio"] not in ratios_seen:
            ratios_seen.append(r["ratio"])
    for ratio in ratios_seen:
        ratio_results = [r for r in all_results if r["ratio"] == ratio]
        best = max(ratio_results, key=lambda x: 0.4 * x["hiprec_prec"] + 0.3 * x["hiprec_f1"] + 0.3 * x["auc"])
        print(f"{best['ratio']:<15} {best['model']:<22} {best['auc']:>6.4f} {best['default_f1']:>7.4f} "
              f"{best['opt_thr']:>8.3f} {best['opt_f1']:>7.4f} {best['opt_prec']:>6.3f} {best['opt_rec']:>6.3f} "
              f"{best['hiprec_f1']:>7.4f} {best['hiprec_prec']:>6.3f}")
    print("=" * 100)
    # Printing full table for each ratio
    for ratio in ratios_seen:
        ratio_results = sorted([r for r in all_results if r["ratio"] == ratio], key=lambda x: x["opt_f1"], reverse=True)
        print(f"\n── {ratio} ──")
        for r in ratio_results:
            print(f"  {r['model']:<22} AUC={r['auc']:.4f}  defF1={r['default_f1']:.4f}  "
                  f"optF1={r['opt_f1']:.4f} (thr={r['opt_thr']:.3f})  hiPF1={r['hiprec_f1']:.4f}")
    return all_results


if __name__ == "__main__":
    main()
