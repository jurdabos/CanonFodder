"""Quick ExtraTrees detail analysis for Exp 6 model report."""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp6_dbscan import (
    add_all_features, prune_features, GS_DBSCAN_PQ,
    WRATIO_LOWER, WRATIO_UPPER,
)
from helpers.io import AVC_PQ, read_parquet
from helpers import cluster, stats
from rapidfuzz import fuzz
import pandas as pd


def main():
    """Produces detailed ExtraTrees analysis for the Exp 6 model report."""
    # Loading training data
    train_raw = read_parquet(GS_DBSCAN_PQ)
    print(f"Training: {len(train_raw)} pairs (pos={train_raw['to_link'].sum()}, neg={(~train_raw['to_link']).sum()})")
    train_df = add_all_features(train_raw)
    # Building AVC test set
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)].drop(columns=["_wr"])
    test_df = add_all_features(test_raw.reset_index(drop=True))
    # Pruning
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    all_num = [c for c in train_df.columns
               if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")]
    num_cols = prune_features(train_df[all_num])
    print(f"Features after pruning: {len(num_cols)}")
    X_train, y_train = train_df[num_cols], train_df[target].astype(int).values
    X_test, y_test = test_df[num_cols], test_df[target].astype(int).values
    # Training ExtraTrees
    et = ExtraTreesClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=47, n_jobs=-1)
    pre = ColumnTransformer([("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
                            remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    pipeline = Pipeline([("prep", pre), ("clf", et)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC: {auc:.4f}")
    print(f"Test: {len(y_test)} pairs ({y_test.sum()} pos, {(y_test == 0).sum()} neg)")
    print(f"\nProbability distribution:\n  min={y_prob.min():.4f}  p25={np.percentile(y_prob, 25):.4f}  "
          f"median={np.median(y_prob):.4f}  p75={np.percentile(y_prob, 75):.4f}  max={y_prob.max():.4f}")
    # Threshold sweep
    print(f"\n{'Thr':>6} | {'Prec':>6} {'Rec':>6} {'F1':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-" * 60)
    for thr in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        print(f"{thr:>6.2f} | {p:>6.3f} {r:>6.3f} {f1:>6.3f} | {tp:>4} {fp:>4} {fn:>4} {tn:>4}")
    # Optimal F1
    precs, recs, thrs = precision_recall_curve(y_test, y_prob)
    f1s = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    opt_thr = thrs[best_idx]
    print(f"\nOptimal threshold: {opt_thr:.6f}")
    print(f"\n=== Classification report at optimal threshold ({opt_thr:.4f}) ===")
    y_pred_opt = (y_prob >= opt_thr).astype(int)
    print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
    # Feature importance
    et_fitted = pipeline.named_steps["clf"]
    importances = sorted(zip(num_cols, et_fitted.feature_importances_), key=lambda x: x[1], reverse=True)
    print("Top 15 features:")
    for name, imp in importances[:15]:
        print(f"  {name:<40} {imp:.4f}")
    # Printing which features are interaction features
    print("\nInteraction features in top 15:")
    for name, imp in importances[:15]:
        if "_minus_" in name or "_mul_" in name:
            print(f"  {name:<40} {imp:.4f} (interaction)")


if __name__ == "__main__":
    main()
