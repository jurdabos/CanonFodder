"""Quick RF detail analysis for model report."""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp5_filtered import build_filtered_test, build_filtered_train, _add_features


def main():
    """Produces detailed RF analysis for the model report."""
    train_raw = build_filtered_train()
    test_raw = build_filtered_test()
    train_df = _add_features(train_raw)
    test_df = _add_features(test_raw)
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    num_cols = [
        c for c in train_df.columns
        if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    X_train, y_train = train_df[num_cols], train_df[target].astype(int).values
    X_test, y_test = test_df[num_cols], test_df[target].astype(int).values
    # Training RF
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=47, n_jobs=-1)
    pre = ColumnTransformer([("num", Pipeline([("scaler", RobustScaler())]), num_cols)], remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    pipeline = Pipeline([("prep", pre), ("clf", rf)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC: {auc:.4f}")
    print(f"Test: {len(y_test)} pairs ({y_test.sum()} pos, {(y_test == 0).sum()} neg)")
    print(f"\nProbability distribution:\n  min={y_prob.min():.4f}  p25={np.percentile(y_prob, 25):.4f}  "
          f"median={np.median(y_prob):.4f}  p75={np.percentile(y_prob, 75):.4f}  max={y_prob.max():.4f}")
    # Sweeping thresholds
    print(f"\n{'Thr':>6} | {'Prec':>6} {'Rec':>6} {'F1':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-" * 60)
    for thr in [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.992, 0.995, 0.998]:
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        print(f"{thr:>6.3f} | {p:>6.3f} {r:>6.3f} {f1:>6.3f} | {tp:>4} {fp:>4} {fn:>4} {tn:>4}")
    # Finding optimal F1 threshold
    precs, recs, thrs = precision_recall_curve(y_test, y_prob)
    f1s = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    opt_thr = thrs[best_idx]
    print(f"\nOptimal threshold: {opt_thr:.6f}")
    print(f"\n=== Classification report at optimal threshold ({opt_thr:.4f}) ===")
    y_pred_opt = (y_prob >= opt_thr).astype(int)
    print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
    # Feature importance
    rf_fitted = pipeline.named_steps["clf"]
    importances = sorted(zip(num_cols, rf_fitted.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name:<30} {imp:.4f}")


if __name__ == "__main__":
    main()
