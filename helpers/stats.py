"""
Supplies small statistical helpers for feature selection and outlier checks.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate

def cramers_v(x, y):
    """
    Returns Cramér’s V between two categorical pandas Series.
    """
    from scipy import stats
    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))


def drop_high_corr_features(cm, threshold, var_table):
    """
    Identifies and returns highly correlated column pairs and the drop list.
    """
    high_corr_pairs = []
    features_to_drop = []
    variance_dict = dict(zip(var_table['features'], var_table['variances']))
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


def iterative_correlation_dropper(current_data, cutoff, varframe, min_features=8):
    """
    Iteratively drops correlated columns until `min_features` remain.
    Args:
        current_data: dataframe to prune
        cutoff: absolute correlation threshold
        varframe: dataframe with per-feature variances
        min_features: minimal column count to keep
    Returns:
        dataframe with reduced multicollinearity
    """
    all_dropped_features = []
    all_corr_pairs = []
    while len(current_data.columns) > min_features:
        # Calculating the correlation matrix
        corr_matrix = current_data.corr(method='spearman')
        # Masking to ignore redundant correlations
        mask = np.triu(np.ones(corr_matrix.shape), k=0)
        corr_matrix = corr_matrix.where(mask == 0)
        # Finding pairs with correlation above cutoff
        corr_pairs = corr_matrix.stack()
        vs_corr_pairs = corr_pairs[abs(corr_pairs) > cutoff].sort_values(ascending=False, key=abs)
        if vs_corr_pairs.empty:
            print(f"No more pairs above an absolute correlation of {cutoff}. Stopping.")
            break
        f1, f2 = vs_corr_pairs.index[0]
        if f1 not in current_data.columns or f2 not in current_data.columns:
            continue
        var_f1 = varframe.loc[varframe['features'] == f1, 'variances'].values[0]
        var_f2 = varframe.loc[varframe['features'] == f2, 'variances'].values[0]
        feature_to_drop = f1 if var_f1 > var_f2 else f2
        current_data = current_data.drop(columns=[feature_to_drop])
        all_dropped_features.append(feature_to_drop)
        all_corr_pairs.append((f1, f2))
        print(f"Dropped feature: {feature_to_drop} (Correlation: {vs_corr_pairs.iloc[0]:.3f})")
        # Stopping condition to retain a minimum number of features
        if len(current_data.columns) <= min_features:
            print(f"Reached the minimum number of features: {min_features}. Stopping.")
            break
    print(f"Final feature count: {len(current_data.columns)}")
    return current_data


def missing_value_ratio(col):
    """
    Returns the percentage of missing values in a pandas Series.
    """
    return (col.isnull().sum() / len(col)) * 100


def show_cm_and_report(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    if title:
        print(f"\n{title}")
    print(tabulate(cm_df, headers="keys", tablefmt="pretty"))
    print(classification_report(y_true, y_pred, target_names=["no link", "link"]))


def variance_testing(dframe, varthresh):
    """
    Applies sklearn.VarianceThreshold and returns (variance_df selected_cols).
    """
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=varthresh)
    _ = selector.fit_transform(dframe)
    variances = selector.variances_
    variance_df = pd.DataFrame({"features": dframe.columns, "variances": variances})
    variance_df = variance_df.sort_values(by="variances", ascending=False)
    selected_features = dframe.columns[selector.get_support(indices=True)]
    return variance_df, selected_features


def winsorization_outliers(df):
    """
    Prints and returns numeric outliers detected via 1st and 99th percentile.
    """
    out = []
    for n in df:
        q1 = np.percentile(df, 1)
        q3 = np.percentile(df, 99)
        if n > q3 or n < q1:
            out.append(n)
    print("Outliers:", out)
    return out
