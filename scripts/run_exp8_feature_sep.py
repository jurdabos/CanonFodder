"""
Runs Experiment 8: Feature-separated training with distribution-matched negatives.

Separates whole-string similarity from the classification feature set.
Whole-string scores are used only for training-data construction (distribution
matching), not as model features. The model trains on token-level,
character-level, and cross-tier interaction features (max 1 whole-string
factor per interaction term).

Steps:
1. Sample 5,000 positives from gs_mb.parquet
2. Compute WRatio for positives → defines the target distribution
3. Distribution-match 5,000 negatives from gs_mb_dbscan.parquet
4. Compute base + cross-tier + non-WS interaction features
5. Drop pure whole-string features from training columns
6. Prune, train all 8 models, threshold sweep against AVC test set
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

from helpers.io import AVC_PQ, GS_MB_PQ, PQ_DIR, read_parquet, sanitize
from helpers.features import compute_pair_features
from helpers import cluster, stats
from helpers.device import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
N_TARGET = 5000

# ── Feature tier definitions ──────────────────────────────────────────────────
WHOLE_STRING_FEATURES = [
    "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
    "WRatio", "QRatio", "norm_levenshtein", "jaro_winkler",
    "length_ratio", "abs_len_diff",
]
NON_WS_FEATURES = [
    # Token-level (5)
    "token_count_diff", "token_jaccard", "shared_token_ratio",
    "lcs_token_len", "token_order_displacement",
    # Character-level (8)
    "bigram_jaccard", "trigram_jaccard",
    "edit_inserts", "edit_deletes", "edit_replaces",
    "shared_prefix_len", "shared_suffix_len", "script_mismatch",
]
# Bounded [0,1] similarity features suitable for interaction products
NON_WS_SIMILARITY = [
    "token_jaccard", "shared_token_ratio", "bigram_jaccard", "trigram_jaccard",
]
# Whole-string scores used for cross-tier interactions
WS_SIM_SCORES = [
    "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
    "WRatio", "QRatio",
]


# ═════════════════════════════════════════════════════════════════════════════
# Step 1–3: Data assembly with distribution matching
# ═════════════════════════════════════════════════════════════════════════════
def _compute_wratio_col(df: pd.DataFrame) -> pd.Series:
    """Computes WRatio for each row."""
    return df.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1,
    )


def distribution_match_negatives(
    positives: pd.DataFrame,
    neg_pool: pd.DataFrame,
    n_target: int = N_TARGET,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Samples negatives whose WRatio distribution matches the positives'.

    Uses fine-grained histogram bins in [60, 100) (the DBSCAN overlap range).
    Iteratively redistributes shortfall from capped bins.
    """
    log.info("Computing WRatio for %d positives...", len(positives))
    pos_wr = _compute_wratio_col(positives)
    log.info("Computing WRatio for %d negatives (may take a minute)...", len(neg_pool))
    neg_wr = _compute_wratio_col(neg_pool)
    neg_pool = neg_pool.copy()
    neg_pool["_wr"] = neg_wr
    # Defining bins within the overlap range [60, 100)
    bin_edges = np.linspace(60, 100, n_bins + 1)
    # Computing positive distribution within [60, 100)
    pos_in_range = pos_wr[(pos_wr >= 60) & (pos_wr < 100)]
    pos_hist, _ = np.histogram(pos_in_range, bins=bin_edges)
    pos_fracs = pos_hist / pos_hist.sum()
    log.info("Positive WRatio distribution in [60,100): %s", dict(zip(
        [f"[{bin_edges[i]:.0f},{bin_edges[i+1]:.0f})" for i in range(n_bins)],
        pos_hist,
    )))
    # Binning negatives
    neg_pool["_bin"] = pd.cut(neg_pool["_wr"], bins=bin_edges, right=False, labels=False)
    neg_pool = neg_pool.dropna(subset=["_bin"])
    neg_pool["_bin"] = neg_pool["_bin"].astype(int)
    # Iterating to distribute n_target across bins
    targets = (pos_fracs * n_target).astype(int)
    # Adjusting rounding errors
    targets[np.argmax(pos_fracs)] += n_target - targets.sum()
    sampled_parts = []
    shortfall = 0
    available_bins = []
    for i in range(n_bins):
        bin_df = neg_pool[neg_pool["_bin"] == i]
        if len(bin_df) < targets[i]:
            shortfall += targets[i] - len(bin_df)
            sampled_parts.append(bin_df)
            log.info("  Bin [%.0f,%.0f): target=%d, available=%d (capped)",
                     bin_edges[i], bin_edges[i + 1], targets[i], len(bin_df))
        else:
            available_bins.append((i, bin_df, targets[i]))
    # Redistributing shortfall proportionally among non-capped bins
    if shortfall > 0 and available_bins:
        total_surplus = sum(len(bdf) - t for _, bdf, t in available_bins)
        for i, bin_df, base_target in available_bins:
            surplus = len(bin_df) - base_target
            extra = int(round(shortfall * surplus / max(total_surplus, 1)))
            final_n = min(base_target + extra, len(bin_df))
            sampled_parts.append(bin_df.sample(n=final_n, random_state=RANDOM_STATE))
            log.info("  Bin [%.0f,%.0f): target=%d+%d=%d, available=%d",
                     bin_edges[i], bin_edges[i + 1], base_target, extra, final_n, len(bin_df))
    else:
        for i, bin_df, target in available_bins:
            sampled_parts.append(bin_df.sample(n=target, random_state=RANDOM_STATE))
            log.info("  Bin [%.0f,%.0f): target=%d, available=%d",
                     bin_edges[i], bin_edges[i + 1], target, len(bin_df))
    neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_wr", "_bin"])
    log.info("Distribution-matched negatives: %d (target was %d).", len(neg_sampled), n_target)
    return neg_sampled


def assemble_training_data() -> pd.DataFrame:
    """Builds the Exp 8 training set: 5K positives + 5K distribution-matched negatives."""
    # Sampling 5K positives from gs_mb.parquet
    gs = read_parquet(GS_MB_PQ)
    positives = gs[gs["to_link"] == True].sample(n=N_TARGET, random_state=RANDOM_STATE)
    log.info("Sampled %d positives from gs_mb.parquet.", len(positives))
    # Loading DBSCAN negatives
    dbscan = read_parquet(GS_DBSCAN_PQ)
    neg_pool = dbscan[dbscan["to_link"] == False].reset_index(drop=True)
    log.info("DBSCAN negative pool: %d pairs.", len(neg_pool))
    # Distribution matching
    neg_sampled = distribution_match_negatives(positives, neg_pool)
    # Combining
    train = pd.concat([
        positives[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
        neg_sampled[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
    ], ignore_index=True)
    log.info("Training set: %d pairs (pos=%d, neg=%d).",
             len(train), train["to_link"].sum(), (~train["to_link"]).sum())
    return train


# ═════════════════════════════════════════════════════════════════════════════
# Step 4–5: Cross-tier and non-WS interaction features
# ═════════════════════════════════════════════════════════════════════════════
def compute_cross_tier_interactions(base_feats: dict[str, float]) -> dict[str, float]:
    """Computes interaction features with at most 1 whole-string factor.

    Two types of interactions:
    1. Cross-tier: 1 whole-string score × 1 non-WS feature (products only)
    2. Non-WS only: all pairwise products among non-WS features
    """
    seen: Counter = Counter()
    interactions: dict[str, float] = {}
    # Cross-tier: whole-string × non-WS (products)
    for ws in WS_SIM_SCORES:
        ws_val = base_feats.get(ws, 0.0)
        for nws in NON_WS_FEATURES:
            nws_val = base_feats.get(nws, 0.0)
            col = sanitize(f"{ws} * {nws}", seen)
            interactions[col] = ws_val * nws_val
    # Non-WS × non-WS (products among all 13 features)
    for a_name, b_name in itertools.combinations(NON_WS_FEATURES, 2):
        a_val = base_feats.get(a_name, 0.0)
        b_val = base_feats.get(b_name, 0.0)
        col = sanitize(f"{a_name} * {b_name}", seen)
        interactions[col] = a_val * b_val
    return interactions


def add_exp8_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base features, cross-tier interactions, and length stats.

    Whole-string base features are computed (needed for interactions)
    but will be excluded from the final training columns.
    """
    # Computing base features (all 3 tiers)
    log.info("Computing base features for %d pairs...", len(df))
    base_rows = df.apply(
        lambda r: pd.Series(compute_pair_features(str(r["variant_a"]), str(r["variant_b"]))),
        axis=1,
    )
    for col in base_rows.columns:
        if col not in df.columns:
            df[col] = base_rows[col]
    # Computing cross-tier + non-WS interactions
    log.info("Computing cross-tier and non-WS interaction features...")
    interaction_rows = df.apply(
        lambda r: pd.Series(compute_cross_tier_interactions(
            compute_pair_features(str(r["variant_a"]), str(r["variant_b"]))
        )),
        axis=1,
    )
    for col in interaction_rows.columns:
        if col not in df.columns:
            df[col] = interaction_rows[col]
    # Synthesising variants column for length_stats
    if "variants" not in df.columns:
        df["variants"] = df["variant_a"].astype(str) + "{" + df["variant_b"].astype(str)
    df = pd.concat([df, df["variants"].apply(stats.length_stats)], axis=1)
    return df


def select_training_columns(df: pd.DataFrame) -> list[str]:
    """Selects numeric columns, excluding whole-string features and metadata."""
    exclude = {"variants", "to_link", "variant_a", "variant_b", "source", "_key"}
    exclude.update(WHOLE_STRING_FEATURES)
    all_num = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    log.info("Candidate training features (excl. whole-string): %d", len(all_num))
    return all_num


# ═════════════════════════════════════════════════════════════════════════════
# Step 6: Experiment runner (reused from Exp 6 with minor tweaks)
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


def run_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    """Trains all 8 models and evaluates with default + optimal thresholds."""
    from corefunc.canon.experiment_runner import _build_model_catalogue
    target = "to_link"
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info("Train: %d | Test: %d | Features: %d | spw: %.2f", len(X_train), len(X_test), len(num_cols), spw)
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
        # Printing feature importances for tree-based models
        if model_name in ("RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "GradientBoosting"):
            clf_fitted = pipeline.named_steps["clf"]
            if hasattr(clf_fitted, "feature_importances_"):
                imps = sorted(zip(num_cols, clf_fitted.feature_importances_), key=lambda x: x[1], reverse=True)
                print(f"Top 10 features ({model_name}):")
                for name, imp in imps[:10]:
                    print(f"  {name:<45} {imp:.4f}")
    # Printing summary table
    print("\n" + "=" * 130)
    print(f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
          f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
          f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}")
    print("-" * 130)
    for r in sorted(results, key=lambda x: x["opt_f1"], reverse=True):
        print(f"{r['model']:<22} {r['auc']:>6.4f} | "
              f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} {r['default_f1']:>6.4f} | "
              f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} {r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
              f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} {r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}")
    print("=" * 130)
    best = max(results, key=lambda x: x["auc"])
    print(f"\nBest model by AUC: {best['model']} (AUC={best['auc']:.4f})")
    # Comparison with Exp 6 baseline
    print("\n── Exp 6 baseline (ExtraTrees): AUC=0.8920, opt F1=0.7050 (thr=0.940) ──")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 8 pipeline."""
    log.info("=== Experiment 8: Feature-separated training ===")
    log.info("Whole-string features used ONLY for data setup, not as model features.")
    # Step 1–3: Assembling training data
    train_raw = assemble_training_data()
    # Step 4: Computing features
    log.info("Step 4: Computing features for %d training pairs...", len(train_raw))
    train_df = add_exp8_features(train_raw)
    # Building AVC test set
    log.info("Building AVC test set...")
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)].drop(columns=["_wr"])
    log.info("AVC test set: %d pairs (pos=%d, neg=%d).",
             len(test_raw), test_raw["to_link"].sum(), (test_raw["to_link"] == False).sum())
    test_df = add_exp8_features(test_raw.reset_index(drop=True))
    # Step 5: Selecting and pruning features (excluding whole-string)
    all_num = select_training_columns(train_df)
    log.info("Pre-pruning feature count: %d", len(all_num))
    num_cols = prune_features(train_df[all_num])
    # Logging which feature tiers survived
    ws_in_final = [c for c in num_cols if c in WHOLE_STRING_FEATURES]
    if ws_in_final:
        log.warning("Whole-string features leaked into final set: %s", ws_in_final)
    cross_tier = [c for c in num_cols if "_mul_" in c and any(ws in c for ws in WS_SIM_SCORES)]
    non_ws_int = [c for c in num_cols if "_mul_" in c and c not in cross_tier]
    base_feats = [c for c in num_cols if c not in cross_tier and c not in non_ws_int]
    log.info("Final features: %d base, %d cross-tier interactions, %d non-WS interactions = %d total",
             len(base_feats), len(cross_tier), len(non_ws_int), len(num_cols))
    # Ensuring test_df has all columns
    missing_in_test = [c for c in num_cols if c not in test_df.columns]
    if missing_in_test:
        log.warning("Columns missing in test_df (filling with 0): %s", missing_in_test)
        for c in missing_in_test:
            test_df[c] = 0.0
    # Step 6: Running experiment
    log.info("Step 6: Running Experiment 8...")
    results = run_experiment(train_df, test_df, num_cols)
    return results


if __name__ == "__main__":
    main()
