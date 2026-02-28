"""
Runs Experiment 11: full model catalogue on gs_mb_max-expanded training data.

Retrains ExtraTrees, RF, XGBoost, LightGBM, GradientBoosting, and composite
models on the ~917K training set assembled for Exp 10 (529K MBDB-max positives
+ 388K distribution-matched DBSCAN negatives), then evaluates on the same AVC
holdout test set used across all experiments.

Compares directly against Exp 6 (ExtraTrees AUC=0.892, F1=0.705 on ~388K
DBSCAN training at 2158:1 class ratio).
"""
from __future__ import annotations
import itertools
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from corefunc.canon.experiment_runner import _build_model_catalogue # noqa: E402
from helpers.io import AVC_PQ, PQ_DIR, read_parquet, sanitize # noqa: E402
from helpers.features import compute_pair_features # noqa: E402
from helpers import cluster, stats # noqa: E402
from helpers.device import get_device # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_MB_MAX_PQ = PQ_DIR / "gs_mb_max.parquet"
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
MAX_NAMES_PER_ARTIST = 30

# Similarity scores used for interaction features (same as Exp 6)
_SIM_SCORES = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio", "QRatio"]


# ═════════════════════════════════════════════════════════════════════════════
# Data assembly (reuses Exp 10 logic)
# ═════════════════════════════════════════════════════════════════════════════
def _compute_wratio_bulk(df: pd.DataFrame, label: str = "") -> pd.Series:
    """Computes WRatio for every row with progress logging."""
    n = len(df)
    log.info("Computing WRatio for %d %s pairs...", n, label)
    wr = df.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    log.info("  Done. WRatio range: [%.0f, %.0f]", wr.min(), wr.max())
    return wr


def assemble_training() -> pd.DataFrame:
    """Assembles the gs_mb_max-based training set with distribution-matched negatives."""
    # Loading gs_mb_max positives
    positives_all = read_parquet(GS_MB_MAX_PQ)
    log.info("gs_mb_max.parquet: %d pairs.", len(positives_all))
    # Filtering positives to WRatio [60, 100)
    pos_wr = _compute_wratio_bulk(positives_all, "positive")
    mask = (pos_wr >= WRATIO_LOWER) & (pos_wr < WRATIO_UPPER)
    positives = positives_all[mask].reset_index(drop=True)
    pos_wr_filtered = pos_wr[mask].reset_index(drop=True)
    log.info("Positives in [%d,%d): %d (of %d total).",
             WRATIO_LOWER, WRATIO_UPPER, len(positives), len(positives_all))
    # Loading DBSCAN negatives
    dbscan = read_parquet(GS_DBSCAN_PQ)
    neg_pool = dbscan[dbscan["to_link"].eq(False)].reset_index(drop=True)
    neg_wr = _compute_wratio_bulk(neg_pool, "negative")
    neg_pool = neg_pool.copy()
    neg_pool["_wr"] = neg_wr
    # Distribution matching with 8 bins in [60, 100)
    n_bins = 8
    bin_edges = np.linspace(60, 100, n_bins + 1)
    pos_hist, _ = np.histogram(pos_wr_filtered, bins=bin_edges)
    pos_fracs = pos_hist / pos_hist.sum()
    n_target = min(len(positives), len(neg_pool))
    log.info("Target negatives: %d (min of %d pos, %d neg pool).",
             n_target, len(positives), len(neg_pool))
    neg_pool["_bin"] = pd.cut(neg_pool["_wr"], bins=bin_edges, right=False, labels=False)
    neg_pool = neg_pool.dropna(subset=["_bin"])
    neg_pool["_bin"] = neg_pool["_bin"].astype(int)
    targets = (pos_fracs * n_target).astype(int)
    targets[np.argmax(pos_fracs)] += n_target - targets.sum()
    sampled_parts = []
    shortfall = 0
    available_bins = []
    for i in range(n_bins):
        bin_df = neg_pool[neg_pool["_bin"] == i]
        if len(bin_df) < targets[i]:
            shortfall += targets[i] - len(bin_df)
            sampled_parts.append(bin_df)
        else:
            available_bins.append((i, bin_df, targets[i]))
    if shortfall > 0 and available_bins:
        total_surplus = sum(len(bdf) - t for _, bdf, t in available_bins)
        for i, bin_df, base_target in available_bins:
            surplus = len(bin_df) - base_target
            extra = int(round(shortfall * surplus / max(total_surplus, 1)))
            final_n = min(base_target + extra, len(bin_df))
            sampled_parts.append(bin_df.sample(n=final_n, random_state=RANDOM_STATE))
    else:
        for i, bin_df, target in available_bins:
            sampled_parts.append(bin_df.sample(n=target, random_state=RANDOM_STATE))
    neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_wr", "_bin"])
    log.info("Distribution-matched negatives: %d.", len(neg_sampled))
    # Combining positives + negatives
    train = pd.concat([
        positives[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
        neg_sampled[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
    ], ignore_index=True)
    log.info("Training set: %d pairs (pos=%d, neg=%d).",
             len(train), train["to_link"].sum(), (~train["to_link"]).sum())
    return train


# ═════════════════════════════════════════════════════════════════════════════
# Feature computation (base + interaction + length stats)
# ═════════════════════════════════════════════════════════════════════════════
def compute_interaction_features_vectorized(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise differences and products among 6 similarity scores.

    Vectorized over the whole DataFrame for efficiency on large datasets.
    """
    seen: Counter = Counter()
    interaction_cols: dict[str, np.ndarray] = {}
    score_names = [s for s in _SIM_SCORES if s in feat_df.columns]
    for i, j in itertools.combinations(range(len(score_names)), 2):
        a_name, b_name = score_names[i], score_names[j]
        a_vals = feat_df[a_name].values
        b_vals = feat_df[b_name].values
        diff_col = sanitize(f"{a_name} - {b_name}", seen)
        interaction_cols[diff_col] = a_vals - b_vals
        prod_col = sanitize(f"{a_name} * {b_name}", seen)
        interaction_cols[prod_col] = a_vals * b_vals
    return pd.DataFrame(interaction_cols, index=feat_df.index)


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base + interaction + length features for every row.

    Uses row-by-row computation for base features (23 columns), then
    vectorized interaction features (30 columns) and length stats (5 columns).
    """
    n = len(df)
    log.info("Computing base features for %d pairs...", n)
    feat_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = str(row["variant_a"]), str(row["variant_b"])
        feats = compute_pair_features(a, b)
        feat_rows.append(feats)
        if (i + 1) % 50000 == 0:
            log.info("  Base features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    feat_df = pd.DataFrame(feat_rows, index=df.index)
    # Adding base features to main DataFrame
    for col in feat_df.columns:
        if col not in df.columns:
            df[col] = feat_df[col]
    # Computing interaction features vectorized
    log.info("Computing interaction features (vectorized)...")
    interaction_df = compute_interaction_features_vectorized(feat_df)
    for col in interaction_df.columns:
        if col not in df.columns:
            df[col] = interaction_df[col]
    log.info("  %d interaction features added.", len(interaction_df.columns))
    # Computing length stats
    if "variants" not in df.columns:
        df["variants"] = df["variant_a"].astype(str) + "{" + df["variant_b"].astype(str)
    df = pd.concat([df, df["variants"].apply(stats.length_stats)], axis=1)
    return df


def prune_features(
    X: pd.DataFrame,
    variance_threshold: float = 0.001,
    corr_cutoff: float = 0.95,
    min_features: int = 20,
) -> list[str]:
    """Applies variance and correlation pruning, returns surviving column names."""
    var_df, selected = stats.variance_testing(X, variance_threshold)
    X_pruned = X[selected]
    log.info("After variance pruning: %d → %d features.", len(X.columns), len(X_pruned.columns))
    X_pruned = stats.iterative_correlation_dropper(X_pruned, corr_cutoff, var_df, min_features)
    log.info("After correlation pruning: → %d features.", len(X_pruned.columns))
    return list(X_pruned.columns)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════
def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Finds the threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _evaluate_at_threshold(y_true, y_prob, threshold):
    """Computes metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═════════════════════════════════════════════════════════════════════════════
def run_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    """Trains all models and evaluates with default + optimal thresholds."""
    target = "to_link"
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info("Train: %d | Test: %d | Features: %d | spw: %.2f",
             len(X_train), len(X_test), len(num_cols), spw)
    log.info("Test distribution: pos=%d, neg=%d (%.1f%% positive)",
             y_test.sum(), (y_test == 0).sum(), 100 * y_test.mean())
    catalogue = _build_model_catalogue(spw, device, random_state=RANDOM_STATE)
    results = []
    for model_name, clf in catalogue.items():
        log.info("─── Training %s ───", model_name)
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
        # Finding high-precision threshold (P ≥ 0.80)
        best_hi = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for t in np.arange(0.50, 0.99, 0.01):
            m = _evaluate_at_threshold(y_test, y_prob, t)
            if m["precision"] >= 0.80 and m["f1"] > best_hi["f1"]:
                best_hi = m
        results.append({
            "model": model_name, "auc": auc,
            "default_f1": default_m["f1"], "default_prec": default_m["precision"], "default_rec": default_m["recall"],
            "opt_thr": optimal_m["threshold"], "opt_f1": optimal_m["f1"],
            "opt_prec": optimal_m["precision"], "opt_rec": optimal_m["recall"],
            "hiprec_thr": best_hi["threshold"], "hiprec_f1": best_hi["f1"],
            "hiprec_prec": best_hi["precision"], "hiprec_rec": best_hi["recall"],
        })
        log.info("%s → AUC=%.4f | def F1=%.4f | opt F1=%.4f (thr=%.3f) | hi-P F1=%.4f (thr=%.3f)",
                 model_name, auc, default_m["f1"], optimal_m["f1"], opt_thr,
                 best_hi["f1"], best_hi["threshold"])
        y_pred_opt = (y_prob >= opt_thr).astype(int)
        print(f"\n=== {model_name} (optimal thr={opt_thr:.3f}) ===")
        print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
        # Feature importance for tree-based models
        if model_name in ("RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "GradientBoosting"):
            clf_fitted = pipeline.named_steps["clf"]
            if hasattr(clf_fitted, "feature_importances_"):
                imps = sorted(zip(num_cols, clf_fitted.feature_importances_), key=lambda x: x[1], reverse=True)
                print(f"Top 10 features ({model_name}):")
                for name, imp in imps[:10]:
                    print(f"  {name:<35} {imp:.4f}")
    # Printing summary table
    print("\n" + "=" * 120)
    print(f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
          f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
          f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x["opt_f1"], reverse=True):
        print(f"{r['model']:<22} {r['auc']:>6.4f} | "
              f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} {r['default_f1']:>6.4f} | "
              f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} {r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
              f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} {r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}")
    print("=" * 120)
    # Baseline comparison
    print("\n── Exp 6 baseline (gs_mb_dbscan, ~388K train, 2158:1 ratio) ──")
    print("ExtraTrees:  AUC=0.8920, opt F1=0.7050 (P=0.750, R=0.667, thr=0.940)")
    # Selecting best model by c9r composite score (0.4×HiP_P + 0.3×HiP_F1 + 0.3×AUC)
    best = max(results, key=lambda x: 0.4 * x["hiprec_prec"] + 0.3 * x["hiprec_f1"] + 0.3 * x["auc"])
    score = 0.4 * best["hiprec_prec"] + 0.3 * best["hiprec_f1"] + 0.3 * best["auc"]
    print(f"\nBest model by c9r score: {best['model']} (score={score:.4f})")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 11 pipeline."""
    log.info("=== Experiment 11: full model catalogue on gs_mb_max data ===")
    # ── Data assembly ──────────────────────────────────────────────────────
    train_full = assemble_training()
    # ── AVC test set ──────────────────────────────────────────────────────
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1,
    )
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)]
    test_df = test_raw.drop(columns=["_wr"]).reset_index(drop=True)
    log.info("AVC test: %d pairs (pos=%d, neg=%d).",
             len(test_df), test_df["to_link"].sum(), (~test_df["to_link"]).sum())
    # ── Feature computation ────────────────────────────────────────────────
    log.info("Computing features for training set...")
    train_df = add_all_features(train_full)
    log.info("Computing features for test set...")
    test_df = add_all_features(test_df)
    # ── Feature pruning ───────────────────────────────────────────────────
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    all_num = [c for c in train_df.columns
               if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_features(train_df[all_num])
    # Ensuring test_df has the same columns
    missing_in_test = [c for c in num_cols if c not in test_df.columns]
    if missing_in_test:
        log.warning("Columns missing in test_df (filling with 0): %s", missing_in_test)
        for c in missing_in_test:
            test_df[c] = 0.0
    # ── Run experiment ────────────────────────────────────────────────────
    log.info("Running Experiment 11 (all models on gs_mb_max data)...")
    results = run_experiment(train_df, test_df, num_cols)
    return results


if __name__ == "__main__":
    main()
