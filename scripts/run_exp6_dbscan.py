"""
Runs Experiment 6: DBSCAN-seeded, MBDB-verified training with interaction features.

Steps:
1. DBSCAN grid search over ~20K scrobble artist names (anchor = 177 positive AVC groups)
2. Extract candidate pairs from non-singleton clusters
3. Query local MBDB mirror to verify each pair (alias → positive, different MBID → negative)
4. Build gs_mb_dbscan.parquet filtered to WRatio [60, 100)
5. Compute base + interaction features, prune, run holdout experiment + threshold sweep
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
from rapidfuzz import fuzz, process
from sklearn.base import clone
from sklearn.cluster import DBSCAN
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

from corefunc.mb_local import _psql_csv, _escape_pg, check_local_mb
from helpers.io import AVC_PQ, PQ_DIR, read_parquet, dump_parquet, SCROBBLE_PQ, sanitize
from helpers.features import compute_pair_features
from helpers import cluster, stats
from helpers.device import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"

# ── Similarity scores used for interaction features ───────────────────────────
_SIM_SCORES = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio", "QRatio"]


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: DBSCAN grid search
# ═════════════════════════════════════════════════════════════════════════════
def step1_dbscan_grid_search() -> tuple[float, np.ndarray, list[str]]:
    """Runs anchor-constrained DBSCAN ε grid search over scrobble artist names.

    Returns (best_eps, labels, artist_names).
    """
    log.info("Step 1: Loading scrobble artist names...")
    scrobbles = read_parquet(SCROBBLE_PQ)
    artist_names = sorted(scrobbles["artist_name"].dropna().unique().tolist())
    n = len(artist_names)
    log.info("Unique artist names: %d", n)
    # Building anchor index sets from positive AVC groups
    avc = read_parquet(AVC_PQ)
    pos_groups = avc[(avc["to_link"].notna()) & (avc["to_link"] == True)]
    name2idx = {name: i for i, name in enumerate(artist_names)}
    anchor_idx_sets = []
    for _, row in pos_groups.iterrows():
        variants = row["artist_variants_text"].split("{")
        idxs = [name2idx[v.strip()] for v in variants if v.strip() in name2idx]
        if len(idxs) >= 2:
            anchor_idx_sets.append(idxs)
    log.info("Anchor sets (positive AVC groups with ≥2 names in scrobble pool): %d", len(anchor_idx_sets))
    # Computing WRatio distance matrix
    log.info("Computing %d×%d WRatio distance matrix (this may take a few minutes)...", n, n)
    sim_matrix = process.cdist(
        artist_names, artist_names, scorer=fuzz.WRatio,
        score_cutoff=0, workers=-1,
    ) / 100.0
    dist = 1.0 - sim_matrix
    # Grid search for smallest ε
    log.info("Running DBSCAN ε grid search [0.05, 1.0)...")
    eps_range = np.arange(0.05, 1.0, 0.01)
    best_eps = None
    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
        if cluster.anchors_ok(labels, anchor_idx_sets):
            best_eps = eps
            break
    if best_eps is None:
        raise RuntimeError("No ε in [0.05, 1.0) co-locates all anchor sets.")
    log.info("Best ε = %.2f (all %d anchors satisfied).", best_eps, len(anchor_idx_sets))
    # Running final DBSCAN at best_eps
    labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    log.info("DBSCAN result: %d clusters, %d noise points (%.1f%%).", n_clusters, n_noise, 100 * n_noise / n)
    return best_eps, labels, artist_names


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Extract candidate pairs from clusters
# ═════════════════════════════════════════════════════════════════════════════
def step2_extract_cluster_pairs(labels: np.ndarray, artist_names: list[str]) -> pd.DataFrame:
    """Generates all pairwise combinations from non-singleton DBSCAN clusters."""
    df = pd.DataFrame({"artist": artist_names, "label": labels})
    clustered = df[df["label"] != -1]
    groups = clustered.groupby("label")["artist"].apply(list).tolist()
    log.info("Non-singleton clusters: %d (sizes: min=%d, max=%d, median=%.0f)",
             len(groups), min(len(g) for g in groups), max(len(g) for g in groups),
             np.median([len(g) for g in groups]))
    rows = []
    for group in groups:
        for a, b in itertools.combinations(sorted(set(group)), 2):
            rows.append({"variant_a": a, "variant_b": b})
    pairs_df = pd.DataFrame(rows)
    log.info("Candidate pairs from DBSCAN clusters: %d", len(pairs_df))
    return pairs_df


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: MBDB verification
# ═════════════════════════════════════════════════════════════════════════════
def step3_mbdb_verify(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Queries local MBDB to label each pair as positive, negative, or ambiguous.

    Positive: both names resolve to the same MBID (via artist name or alias).
    Negative: both resolve to different MBIDs.
    Ambiguous: one or both names not found → dropped.
    """
    if not check_local_mb():
        raise RuntimeError("Local MBDB mirror not reachable.")
    # Collecting all unique names
    all_names = sorted(set(pairs_df["variant_a"]) | set(pairs_df["variant_b"]))
    log.info("Querying MBDB for %d unique names...", len(all_names))
    # Building name → set of MBIDs mapping (via artist name AND alias)
    name_to_mbids: dict[str, set[str]] = {}
    batch_size = 150
    for i in range(0, len(all_names), batch_size):
        batch = all_names[i:i + batch_size]
        values = ",".join(f"'{_escape_pg(n)}'" for n in batch)
        # Querying: name matches artist.name OR alias.name
        sql = f"""\
SELECT DISTINCT q.lookup_name, a.gid::text AS mbid
FROM (
    SELECT name AS lookup_name, id AS artist_id
    FROM musicbrainz.artist
    WHERE name IN ({values})
    UNION
    SELECT aa.name AS lookup_name, aa.artist AS artist_id
    FROM musicbrainz.artist_alias aa
    WHERE aa.name IN ({values})
) q
JOIN musicbrainz.artist a ON a.id = q.artist_id"""
        result = _psql_csv(sql)
        if not result.empty:
            for _, row in result.iterrows():
                name_to_mbids.setdefault(row["lookup_name"], set()).add(row["mbid"])
        if (i // batch_size) % 10 == 0:
            log.info("  MBDB lookup progress: %d/%d names...", min(i + batch_size, len(all_names)), len(all_names))
    found = sum(1 for v in name_to_mbids.values() if v)
    log.info("MBDB resolution: %d/%d names found.", found, len(all_names))
    # Labelling pairs
    rows = []
    n_pos, n_neg, n_ambig = 0, 0, 0
    for _, pair in pairs_df.iterrows():
        a, b = pair["variant_a"], pair["variant_b"]
        mbids_a = name_to_mbids.get(a, set())
        mbids_b = name_to_mbids.get(b, set())
        if not mbids_a or not mbids_b:
            n_ambig += 1
            continue
        # Checking for shared MBIDs
        shared = mbids_a & mbids_b
        if shared:
            rows.append({"variant_a": a, "variant_b": b, "to_link": True, "source": "dbscan_mbdb_pos"})
            n_pos += 1
        else:
            rows.append({"variant_a": a, "variant_b": b, "to_link": False, "source": "dbscan_mbdb_neg"})
            n_neg += 1
    log.info("MBDB verification: %d positive, %d negative, %d ambiguous (dropped).", n_pos, n_neg, n_ambig)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Build gs_mb_dbscan.parquet
# ═════════════════════════════════════════════════════════════════════════════
def step4_build_parquet(verified_df: pd.DataFrame) -> pd.DataFrame:
    """Filters to WRatio [60, 100) and saves gs_mb_dbscan.parquet."""
    # Computing WRatio
    verified_df["_wratio"] = verified_df.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1,
    )
    mask = (verified_df["_wratio"] >= WRATIO_LOWER) & (verified_df["_wratio"] < WRATIO_UPPER)
    filtered = verified_df[mask].drop(columns=["_wratio"]).reset_index(drop=True)
    pos = filtered["to_link"].sum()
    neg = len(filtered) - pos
    log.info("gs_mb_dbscan.parquet: %d pairs (pos=%d, neg=%d) in WRatio [%d, %d).",
             len(filtered), pos, neg, WRATIO_LOWER, WRATIO_UPPER)
    dump_parquet(filtered, GS_DBSCAN_PQ)
    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Interaction features
# ═════════════════════════════════════════════════════════════════════════════
def compute_interaction_features(base_feats: dict[str, float]) -> dict[str, float]:
    """Computes pairwise differences and products among the 6 similarity scores."""
    seen: Counter = Counter()
    interaction: dict[str, float] = {}
    sim_vals = {k: base_feats[k] for k in _SIM_SCORES if k in base_feats}
    score_names = list(sim_vals.keys())
    for i, j in itertools.combinations(range(len(score_names)), 2):
        a_name, b_name = score_names[i], score_names[j]
        a_val, b_val = sim_vals[a_name], sim_vals[b_name]
        # Pairwise difference
        diff_col = sanitize(f"{a_name} - {b_name}", seen)
        interaction[diff_col] = a_val - b_val
        # Pairwise product
        prod_col = sanitize(f"{a_name} * {b_name}", seen)
        interaction[prod_col] = a_val * b_val
    return interaction


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base + interaction features and length stats."""
    # Computing base features
    base_rows = df.apply(
        lambda r: pd.Series(compute_pair_features(str(r["variant_a"]), str(r["variant_b"]))),
        axis=1,
    )
    for col in base_rows.columns:
        if col not in df.columns:
            df[col] = base_rows[col]
    # Computing interaction features
    interaction_rows = df.apply(
        lambda r: pd.Series(compute_interaction_features(
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
# Step 6: Experiment runner with threshold sweep
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


def step6_run_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
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
    catalogue = _build_model_catalogue(spw, device, random_state=47)
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
        # Feature importance for tree-based single models
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
    # Detailed threshold sweep for best model by AUC
    best = max(results, key=lambda x: x["auc"])
    print(f"\nBest model by AUC: {best['model']} (AUC={best['auc']:.4f})")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 6 pipeline."""
    log.info("=== Experiment 6: DBSCAN-seeded, MBDB-verified, interaction features ===")
    # Step 1: DBSCAN
    best_eps, labels, artist_names = step1_dbscan_grid_search()
    # Step 2: Extract pairs
    pairs_df = step2_extract_cluster_pairs(labels, artist_names)
    # Step 3: MBDB verification
    verified_df = step3_mbdb_verify(pairs_df)
    if verified_df.empty:
        log.error("No MBDB-verified pairs produced. Aborting.")
        return
    # Step 4: Build parquet
    train_raw = step4_build_parquet(verified_df)
    # Step 5: Features
    log.info("Step 5: Computing features for %d training pairs...", len(train_raw))
    train_df = add_all_features(train_raw)
    # Building AVC test set (same filter as Exp 5)
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
    test_df = add_all_features(test_raw.reset_index(drop=True))
    # Pruning features
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    all_num = [c for c in train_df.columns
               if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_features(train_df[all_num])
    # Ensuring test_df has the same columns (some interaction cols might be missing)
    missing_in_test = [c for c in num_cols if c not in test_df.columns]
    if missing_in_test:
        log.warning("Columns missing in test_df (filling with 0): %s", missing_in_test)
        for c in missing_in_test:
            test_df[c] = 0.0
    # Step 6: Run experiment
    log.info("Step 6: Running Experiment 6...")
    results = step6_run_experiment(train_df, test_df, num_cols)
    return results


if __name__ == "__main__":
    main()
