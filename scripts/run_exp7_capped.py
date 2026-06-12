"""
Runs Experiment 7: DBSCAN-seeded training with cluster capping and negative subsampling.

Builds on Exp 6 but:
- Caps cluster sizes at MAX_CLUSTER_SIZE (default 30) to eliminate the mega-cluster
- Subsamples negatives to a target neg:pos ratio for balanced training
- Reuses interaction features and pruning from Exp 6
"""

from __future__ import annotations

import itertools
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from corefunc.mb_local import _escape_pg, _psql_csv, check_local_mb  # noqa: E402
from helpers import cluster  # noqa: E402
from helpers.device import get_device  # noqa: E402
from helpers.io import AVC_PQ, PQ_DIR, SCROBBLE_PQ, dump_parquet, read_parquet  # noqa: E402
from scripts.run_exp6_dbscan import (  # noqa: E402
    WRATIO_LOWER,  # noqa: E402
    WRATIO_UPPER,
    _evaluate_at_threshold,
    _optimal_threshold,  # noqa: E402
    add_all_features,  # noqa: E402
    prune_features,
)  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

MAX_CLUSTER_SIZE = 30
NEG_POS_RATIO = 10  # target negative : positive ratio
RANDOM_STATE = 47
GS_DBSCAN_CAPPED_PQ = PQ_DIR / "gs_mb_dbscan_capped.parquet"


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: DBSCAN with cluster capping
# ═════════════════════════════════════════════════════════════════════════════
def step1_dbscan_capped() -> tuple[float, list[list[str]]]:
    """Runs DBSCAN and caps oversized clusters.

    Returns (best_eps, capped_groups).
    """
    log.info("Step 1: DBSCAN grid search + cluster capping (max=%d)...", MAX_CLUSTER_SIZE)
    scrobbles = read_parquet(SCROBBLE_PQ)
    artist_names = sorted(scrobbles["artist_name"].dropna().unique().tolist())
    n = len(artist_names)
    log.info("Unique artist names: %d", n)
    # Building anchor sets
    avc = read_parquet(AVC_PQ)
    pos_groups = avc[(avc["to_link"].notna()) & (avc["to_link"].eq(True))]
    name2idx = {name: i for i, name in enumerate(artist_names)}
    anchor_idx_sets = []
    # Collecting anchor names to protect from random subsampling
    anchor_names: set[str] = set()
    for _, row in pos_groups.iterrows():
        variants = [v.strip() for v in row["artist_variants_text"].split("{") if v.strip()]
        idxs = [name2idx[v] for v in variants if v in name2idx]
        if len(idxs) >= 2:
            anchor_idx_sets.append(idxs)
            anchor_names.update(v for v in variants if v in name2idx)
    log.info("Anchor sets: %d (protecting %d anchor names from subsampling).", len(anchor_idx_sets), len(anchor_names))
    # Computing distance matrix
    log.info("Computing %d×%d distance matrix...", n, n)
    sim_matrix = process.cdist(artist_names, artist_names, scorer=fuzz.WRatio, score_cutoff=0, workers=-1) / 100.0
    dist = 1.0 - sim_matrix
    # Grid search
    log.info("Running ε grid search [0.05, 1.0)...")
    best_eps = None
    for eps in np.arange(0.05, 1.0, 0.01):
        labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
        if cluster.anchors_ok(labels, anchor_idx_sets):
            best_eps = eps
            break
    if best_eps is None:
        raise RuntimeError("No ε satisfies all anchor constraints.")
    labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
    log.info("Best ε = %.2f", best_eps)
    # Building groups and capping
    df = pd.DataFrame({"artist": artist_names, "label": labels})
    clustered = df[df["label"] != -1]
    raw_groups = clustered.groupby("label")["artist"].apply(list).tolist()
    n_raw = len(raw_groups)
    sizes_before = [len(g) for g in raw_groups]
    oversized = sum(1 for s in sizes_before if s > MAX_CLUSTER_SIZE)
    log.info(
        "Raw clusters: %d (oversized > %d: %d, max size: %d).", n_raw, MAX_CLUSTER_SIZE, oversized, max(sizes_before)
    )
    # Capping: for clusters > MAX_CLUSTER_SIZE, keeping anchor names + random sample of the rest
    rng = np.random.default_rng(RANDOM_STATE)
    capped_groups = []
    for group in raw_groups:
        if len(group) <= MAX_CLUSTER_SIZE:
            capped_groups.append(group)
        else:
            # Splitting into anchor names and non-anchor names
            anchors_in = [n for n in group if n in anchor_names]
            others = [n for n in group if n not in anchor_names]
            budget = MAX_CLUSTER_SIZE - len(anchors_in)
            if budget > 0 and others:
                sampled = rng.choice(others, size=min(budget, len(others)), replace=False).tolist()
            else:
                sampled = []
            capped = anchors_in + sampled
            capped_groups.append(capped)
    sizes_after = [len(g) for g in capped_groups]
    total_names = sum(sizes_after)
    log.info(
        "Capped clusters: %d groups, %d total names (max size: %d, median: %.0f).",
        len(capped_groups),
        total_names,
        max(sizes_after),
        np.median(sizes_after),
    )
    return best_eps, capped_groups


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Extract pairs from capped clusters
# ═════════════════════════════════════════════════════════════════════════════
def step2_extract_pairs(capped_groups: list[list[str]]) -> pd.DataFrame:
    """Generates all pairwise combinations from capped clusters."""
    rows = []
    for group in capped_groups:
        for a, b in itertools.combinations(sorted(set(group)), 2):
            rows.append({"variant_a": a, "variant_b": b})
    pairs_df = pd.DataFrame(rows)
    log.info("Candidate pairs from capped clusters: %d", len(pairs_df))
    return pairs_df


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: MBDB verification (reused from Exp 6)
# ═════════════════════════════════════════════════════════════════════════════
def step3_mbdb_verify(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Queries local MBDB to label each pair as positive, negative, or ambiguous."""
    if not check_local_mb():
        raise RuntimeError("Local MBDB mirror not reachable.")
    all_names = sorted(set(pairs_df["variant_a"]) | set(pairs_df["variant_b"]))
    log.info("Querying MBDB for %d unique names...", len(all_names))
    name_to_mbids: dict[str, set[str]] = {}
    batch_size = 150
    for i in range(0, len(all_names), batch_size):
        batch = all_names[i : i + batch_size]
        values = ",".join(f"'{_escape_pg(n)}'" for n in batch)
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
            log.info("  MBDB progress: %d/%d names...", min(i + batch_size, len(all_names)), len(all_names))
    found = sum(1 for v in name_to_mbids.values() if v)
    log.info("MBDB resolution: %d/%d names found.", found, len(all_names))
    rows = []
    n_pos, n_neg, n_ambig = 0, 0, 0
    for _, pair in pairs_df.iterrows():
        a, b = pair["variant_a"], pair["variant_b"]
        mbids_a = name_to_mbids.get(a, set())
        mbids_b = name_to_mbids.get(b, set())
        if not mbids_a or not mbids_b:
            n_ambig += 1
            continue
        if mbids_a & mbids_b:
            rows.append({"variant_a": a, "variant_b": b, "to_link": True, "source": "dbscan_mbdb_pos"})
            n_pos += 1
        else:
            rows.append({"variant_a": a, "variant_b": b, "to_link": False, "source": "dbscan_mbdb_neg"})
            n_neg += 1
    log.info("MBDB verification: %d pos, %d neg, %d ambiguous.", n_pos, n_neg, n_ambig)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: WRatio filtering + negative subsampling
# ═════════════════════════════════════════════════════════════════════════════
def step4_filter_and_subsample(verified_df: pd.DataFrame) -> pd.DataFrame:
    """Filters to WRatio [60, 100), then subsamples negatives to target ratio."""
    # Computing WRatio and filtering
    verified_df["_wratio"] = verified_df.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    mask = (verified_df["_wratio"] >= WRATIO_LOWER) & (verified_df["_wratio"] < WRATIO_UPPER)
    filtered = verified_df[mask].copy()
    pos_df = filtered[filtered["to_link"].eq(True)]
    neg_df = filtered[filtered["to_link"].eq(False)]
    n_pos = len(pos_df)
    n_neg_raw = len(neg_df)
    log.info("After WRatio filter: %d pos, %d neg (ratio %.0f:1).", n_pos, n_neg_raw, n_neg_raw / max(n_pos, 1))
    # Stratified negative subsampling: maintaining WRatio band distribution
    target_neg = min(n_pos * NEG_POS_RATIO, n_neg_raw)
    if n_neg_raw > target_neg:
        # Stratifying by WRatio bands to preserve difficulty distribution
        neg_df = neg_df.copy()
        neg_df["_band"] = pd.cut(neg_df["_wratio"], bins=[60, 70, 80, 90, 95, 100], right=False)
        sampled_parts = []
        band_counts = neg_df["_band"].value_counts()
        total_in_bands = band_counts.sum()
        for band, count in band_counts.items():
            band_df = neg_df[neg_df["_band"] == band]
            # Proportional allocation
            n_sample = max(1, int(round(target_neg * count / total_in_bands)))
            n_sample = min(n_sample, len(band_df))
            sampled_parts.append(band_df.sample(n=n_sample, random_state=RANDOM_STATE))
        neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_band"])
        log.info("Subsampled negatives: %d → %d (target ratio %d:1).", n_neg_raw, len(neg_sampled), NEG_POS_RATIO)
    else:
        neg_sampled = neg_df
        log.info("No subsampling needed: %d negatives ≤ target %d.", n_neg_raw, target_neg)
    combined = pd.concat([pos_df, neg_sampled], ignore_index=True).drop(columns=["_wratio"]).reset_index(drop=True)
    pos_final = combined["to_link"].sum()
    neg_final = len(combined) - pos_final
    log.info(
        "Final training set: %d pairs (pos=%d, neg=%d, ratio %.1f:1).",
        len(combined),
        pos_final,
        neg_final,
        neg_final / max(pos_final, 1),
    )
    dump_parquet(combined, GS_DBSCAN_CAPPED_PQ)
    return combined


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Run experiment with threshold sweep
# ═════════════════════════════════════════════════════════════════════════════
def step5_run_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    """Trains all 8 models and evaluates with default + optimal + high-precision thresholds."""
    from corefunc.canon.experiment_runner import _build_model_catalogue

    target = "to_link"
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info(
        "Train: %d | Test: %d | Features: %d | spw: %.2f | device: %s",
        len(X_train),
        len(X_test),
        len(num_cols),
        spw,
        device,
    )
    log.info(
        "Test distribution: pos=%d, neg=%d (%.1f%% positive)", y_test.sum(), (y_test == 0).sum(), 100 * y_test.mean()
    )
    catalogue = _build_model_catalogue(spw, device, random_state=RANDOM_STATE)
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
        results.append(
            {
                "model": model_name,
                "auc": auc,
                "default_f1": default_m["f1"],
                "default_prec": default_m["precision"],
                "default_rec": default_m["recall"],
                "opt_thr": optimal_m["threshold"],
                "opt_f1": optimal_m["f1"],
                "opt_prec": optimal_m["precision"],
                "opt_rec": optimal_m["recall"],
                "hiprec_thr": best_hi["threshold"],
                "hiprec_f1": best_hi["f1"],
                "hiprec_prec": best_hi["precision"],
                "hiprec_rec": best_hi["recall"],
            }
        )
        log.info(
            "%s → AUC=%.4f | def F1=%.4f | opt F1=%.4f (thr=%.3f) | hi-P F1=%.4f (thr=%.3f, P=%.3f)",
            model_name,
            auc,
            default_m["f1"],
            optimal_m["f1"],
            opt_thr,
            best_hi["f1"],
            best_hi["threshold"],
            best_hi["precision"],
        )
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
                    print(f"  {name:<40} {imp:.4f}")
    # Printing summary
    print("\n" + "=" * 120)
    print(
        f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
        f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
        f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}"
    )
    print("-" * 120)
    for r in sorted(results, key=lambda x: x["opt_f1"], reverse=True):
        print(
            f"{r['model']:<22} {r['auc']:>6.4f} | "
            f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} {r['default_f1']:>6.4f} | "
            f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} {r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
            f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} {r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}"
        )
    print("=" * 120)
    # Selecting best model by c9r composite score (0.4×HiP_P + 0.3×HiP_F1 + 0.3×AUC)
    best = max(results, key=lambda x: 0.4 * x["hiprec_prec"] + 0.3 * x["hiprec_f1"] + 0.3 * x["auc"])
    score = 0.4 * best["hiprec_prec"] + 0.3 * best["hiprec_f1"] + 0.3 * best["auc"]
    print(f"\nBest model by c9r score: {best['model']} (score={score:.4f})")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs Experiment 7: capped DBSCAN + subsampled negatives."""
    log.info("=== Experiment 7: DBSCAN capped (max %d) + neg subsampling (%d:1) ===", MAX_CLUSTER_SIZE, NEG_POS_RATIO)
    # Step 1: DBSCAN + capping
    best_eps, capped_groups = step1_dbscan_capped()
    # Step 2: Extract pairs
    pairs_df = step2_extract_pairs(capped_groups)
    # Step 3: MBDB verification
    verified_df = step3_mbdb_verify(pairs_df)
    if verified_df.empty:
        log.error("No MBDB-verified pairs. Aborting.")
        return
    # Step 4: Filter + subsample
    train_raw = step4_filter_and_subsample(verified_df)
    # Step 5a: Features
    log.info("Computing features for %d training pairs...", len(train_raw))
    train_df = add_all_features(train_raw)
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
    log.info(
        "AVC test: %d pairs (pos=%d, neg=%d).",
        len(test_raw),
        test_raw["to_link"].sum(),
        (test_raw["to_link"].eq(False)).sum(),
    )
    test_df = add_all_features(test_raw.reset_index(drop=True))
    # Pruning
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    all_num = [
        c
        for c in train_df.columns
        if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_features(train_df[all_num])
    missing_in_test = [c for c in num_cols if c not in test_df.columns]
    if missing_in_test:
        log.warning("Filling missing test columns with 0: %s", missing_in_test)
        for c in missing_in_test:
            test_df[c] = 0.0
    # Step 5b: Run experiment
    log.info("Running Experiment 7...")
    results = step5_run_experiment(train_df, test_df, num_cols)
    return results


if __name__ == "__main__":
    main()
