"""
Supplies statistical and ML-related utility functions for c9r.

All DB-dependent functions have been removed in the v0.6 migration.
Only pure-computation helpers remain.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

log = logging.getLogger(__name__)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Returns Cramér's V between two categorical pandas Series."""
    from scipy import stats

    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))


def drop_high_corr_features(cm: pd.DataFrame, threshold: float, var_table: pd.DataFrame):
    """Identifies and returns highly correlated column pairs and the drop list."""
    high_corr_pairs = []
    features_to_drop = []
    variance_dict = dict(zip(var_table["features"], var_table["variances"]))
    for _i_ in range(len(cm.columns)):
        for j in range(_i_):
            if abs(cm.iloc[_i_, j]) > threshold:
                high_corr_pairs.append((cm.columns[_i_], cm.columns[j]))
                feature_i = cm.columns[_i_]
                feature_j = cm.columns[j]
                if variance_dict[feature_i] < variance_dict[feature_j]:
                    features_to_drop.append(feature_i)
                else:
                    features_to_drop.append(feature_j)
    return high_corr_pairs, features_to_drop


def iterative_correlation_dropper(
    current_data: pd.DataFrame,
    cutoff: float,
    varframe: pd.DataFrame,
    min_features: int = 8,
) -> pd.DataFrame:
    """
    Iteratively drops correlated columns until *min_features* remain.

    Parameters
    ----------
    current_data : DataFrame to prune.
    cutoff : absolute correlation threshold.
    varframe : DataFrame with per-feature variances.
    min_features : minimal column count to keep.
    """
    while len(current_data.columns) > min_features:
        # Calculating the correlation matrix
        corr_matrix = current_data.corr(method="spearman")
        mask = np.triu(np.ones(corr_matrix.shape), k=0)
        corr_matrix = corr_matrix.where(mask == 0)
        corr_pairs = corr_matrix.stack()
        vs_corr_pairs = corr_pairs[abs(corr_pairs) > cutoff].sort_values(ascending=False, key=abs)
        if vs_corr_pairs.empty:
            break
        f1, f2 = vs_corr_pairs.index[0]
        if f1 not in current_data.columns or f2 not in current_data.columns:
            continue
        var_f1 = varframe.loc[varframe["features"] == f1, "variances"].values[0]
        var_f2 = varframe.loc[varframe["features"] == f2, "variances"].values[0]
        feature_to_drop = f1 if var_f1 > var_f2 else f2
        current_data = current_data.drop(columns=[feature_to_drop])
        log.debug("Dropped feature: %s (corr=%.3f)", feature_to_drop, vs_corr_pairs.iloc[0])
        if len(current_data.columns) <= min_features:
            break
    return current_data


def length_stats(input_text: str) -> pd.Series:
    """Computes length-based features from a '{'-delimited variant string."""
    if not isinstance(input_text, str):
        return pd.Series({"sig_len": 0, "n_variants": 0, "avg_name_len": 0.0, "max_name_len": 0, "var_len": 0.0})
    parts = re.split(r"{", input_text)
    lens = [len(p.strip()) for p in parts if p.strip()]
    return pd.Series(
        {
            "sig_len": sum(lens),
            "n_variants": len(lens),
            "avg_name_len": np.mean(lens),
            "max_name_len": np.max(lens),
            "var_len": np.std(lens),
        }
    )


def missing_value_ratio(col: pd.Series) -> float:
    """Returns the percentage of missing values in a pandas Series."""
    return (col.isnull().sum() / len(col)) * 100


def show_cm_and_report(y_true, y_pred, title: str = "") -> None:
    """Prints confusion matrix and classification report."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    if title:
        print(f"\n{title}")
    print(cm_df.to_string())
    print(classification_report(y_true, y_pred, target_names=["no link", "link"]))


def variance_testing(dframe: pd.DataFrame, varthresh: float):
    """Applies sklearn.VarianceThreshold and returns (variance_df, selected_cols)."""
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=varthresh)
    selector.fit_transform(dframe)
    variance_df = pd.DataFrame({"features": dframe.columns, "variances": selector.variances_})
    variance_df = variance_df.sort_values(by="variances", ascending=False)
    selected_features = dframe.columns[selector.get_support(indices=True)]
    return variance_df, selected_features


def winsorization_outliers(df) -> list:
    """Returns numeric outliers detected via 1st and 99th percentile."""
    out = []
    q1, q3 = np.percentile(df, 1), np.percentile(df, 99)
    for n in df:
        if n > q3 or n < q1:
            out.append(n)
    return out
