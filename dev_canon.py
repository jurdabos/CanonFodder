# %% Basic setup
from dotenv import load_dotenv

load_dotenv()
from DB import SessionLocal
from DB.models import ArtistVariantsCanonized
import featuretools as ft
from hdbscan import HDBSCAN
from helpers import cli
from helpers import io
from helpers import cluster
from helpers import stats
import itertools
import json
import logging
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
import mbAPI
mbAPI.init()
import numpy as np
import os

os.environ["MPLBACKEND"] = "TkAgg"
import pandas as pd
from pathlib import Path
from rapidfuzz import process, distance as rf_dist
from rapidfuzz.distance import Levenshtein
import re
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import ParameterGrid

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text
import sys
from tabulate import tabulate

if '__file__' in globals():
    HERE = Path(__file__).resolve().parent  # running from a file
else:
    HERE = Path.cwd()  # running in a console/notebook
JSON_DIR = HERE / "JSON"
PALETTES_FILE = JSON_DIR / "palettes.json"
LOGGER = logging.getLogger(__name__)

# %% Display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{: .2f}".format
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = io.register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# %% Step 1: Input & pre-processing
print("=================================================")
print("Welcome to the CanonFodder canonization workflow!")
print("=================================================\n")
data, latest_filename = io.latest_parquet(return_df=True)
if data is None or data.empty:
    print("No DataFrame was loaded; cannot proceed with EDA.")
    sys.exit()
if data is None or data.empty:
    sys.exit("🚫  No scrobble data found – aborting EDA.")
data.columns = ["Artist", "Album", "Song", "Datetime"]
data.dropna(subset=["Datetime"], inplace=True)
data = data.drop_duplicates(["Artist", "Album", "Song", "Datetime"])

# %% Step 2: Variant clustering
LOGGER.info("Fetching already canonised artist-name variants…")
with SessionLocal() as sess:
    canon_rows = (
        sess.query(ArtistVariantsCanonized)
        .filter(ArtistVariantsCanonized.to_link.is_(True))
        .all()
    )
SEPARATOR = "{"


def _split_variants(raw: str) -> list[str]:
    """
    Split the artist_variants field into its individual names.
    • Primary separator is “{”   (the new, unambiguous choice)
    • Strips whitespace and ignores empty items.
    """
    return [v.strip()
            for v in re.split(rf"[{re.escape(SEPARATOR)}]", raw)
            if v.strip()]


variant_to_canon: dict[str, str] = {}
for row in canon_rows:
    # `artist_variants` keeps the variants in one string, {-separated (a{b{c).
    variants: list[str] = [v.strip() for v in row.artist_variants.split("{") if v.strip()]
    for variant in _split_variants(row.artist_variants):
        if variant and variant != row.canonical_name:
            variant_to_canon[variant] = row.canonical_name
if variant_to_canon:
    data["Artist"] = data["Artist"].replace(variant_to_canon)
artist_counts = data["Artist"].value_counts()
artist_counts_df = artist_counts.reset_index()
artist_counts_df.columns = ["Artist", "Count"]
count_threshold = 3
mandatory = {v.strip() for group in cluster.variant_sets for v in group}
fltrd_artcount = artist_counts_df[artist_counts_df["Count"] >= count_threshold]
fltrd_artcount = pd.concat([
    fltrd_artcount,
    artist_counts_df[artist_counts_df["Artist"].isin(mandatory)]
]).drop_duplicates(subset=["Artist"])
artist_names = fltrd_artcount["Artist"].str.strip().tolist()
n = len(artist_names)
sim_matrix = process.cdist(
        artist_names,
        artist_names,
        score_cutoff=0,
        workers=-1,
) / 100.0
dist = 1.0 - sim_matrix
name2idx = {name: i for i, name in enumerate(artist_names)}
anchor_idx_sets = [
    [name2idx[n.strip()] for n in group if n.strip() in name2idx]
    for group in cluster.variant_sets
]
anchor_idx_sets = [s for s in anchor_idx_sets if len(s) >= 2]   # ignore degenerate
eps_range = np.arange(0.05, 1.0, 0.01)
best_eps = None
for eps in eps_range:
    labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
    if cluster.anchors_ok(labels, anchor_idx_sets):
        best_eps = eps
        break
if best_eps is None:
    raise ValueError("No ε in the range puts every variant_set in one cluster.")
print(f"Chosen ε = {best_eps:.2f} (all anchors satisfied)")
labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
clusters = (
    pd.DataFrame({"Artist": artist_names, "label": labels})
      .query("label != -1")
      .groupby("label")["Artist"].apply(list)
      .tolist()
)

# %% Step 3: Gold standard creation
with SessionLocal() as sess:
    unhandled = 0
    for group in clusters:
        if len(group) <= 1:
            continue
        sig = "|".join(sorted(group))
        row = sess.execute(
            text("SELECT 1 FROM artist_variants_canonized WHERE artist_variants = :sig"),
            {"sig": sig},
        ).first()
        if row is None:
            unhandled += 1
print(f"Number of groups identified by DBSCAN: {len(clusters)}")
print(f"Number of groups NOT yet handled by user: {unhandled}")
data, fltrd_artcount = cli.unify_artist_names_cli(
    data=data,
    fltrd_artcount=fltrd_artcount,
    similar_artist_groups=clusters,
)
print("Done unifying. Next steps coming later. Now exiting...")


# %%
# Building composite features with ft of 6 rapidfuzz function outputs


# %%
# Logic for calculating and visualizing Cramér's V for similarity score metrics
categorical_columns = data.select_dtypes(include=["object"]).columns
cramers_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 == col2:
            cramers_matrix.loc[col1, col2] = 1.0
        else:
            cramers_matrix.loc[col1, col2] = stats.cramers_v(data[col1], data[col2])
cramers_matrix = cramers_matrix.astype(float)
cramers_matrix.columns = [
    helper.short_labels.get(col, col) for col in cramers_matrix.columns
]
cramers_matrix.index = [
    helper.short_labels.get(col, col) for col in cramers_matrix.index
]
plt.figure(figsize=(12, 8))
sns.heatmap(
    cramers_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, mask=None
)
plt.gca().xaxis.set_tick_params(labeltop=True, labelbottom=False)
plt.xticks(rotation=45, ha="center", fontsize=9)
plt.yticks(rotation=0, ha="right", fontsize=10)
plt.tight_layout()
plt.show()
plt.close()

# %%
# Checking the correlation matrix using Spearman's
vs_corr = data_VT.corr(method="spearman")
plt.figure(figsize=(20, 18))
sns.heatmap(
    vs_corr,
    annot=True,
    annot_kws={"size": 8},
    cmap="winter",
    fmt=".2f",
    cbar=True,
    linewidths=0.5,
    mask=np.triu(vs_corr),
)
plt.title("")
plt.xticks(rotation=90)
plt.show()
plt.close()

# %%
# Identifying attributes with very strong correlation
vs_corr_pairs, features_to_drop_vs = stats.drop_high_corr_features(
    vs_corr, vs_threshold, variance_df
)

# %%
# Dropping the deselected tables
drop_table = pd.DataFrame(
    {
        "Very strongly correlated feature pair": [
            f"{pair[0]} - {pair[1]}" for pair in vs_corr_pairs
        ],
        "Feature to drop": features_to_drop_vs,
    }
)
print(tabulate(drop_table, headers="keys", tablefmt="pretty"))
data_VS_corr = data_VT.drop(columns=features_to_drop_vs)

# %%
# Recalculating for strong correlations
corr_matrix_shrunken = data_VS_corr.corr(method="spearman")
s_threshold = 0.6
high_corr_pairs_s, features_to_drop_s = stats.drop_high_corr_features(
    corr_matrix_shrunken, s_threshold, variance_df
)
drop_table_s = pd.DataFrame(
    {
        "Strongly correlated feature pair": [
            f"{pair[0]} - {pair[1]}" for pair in high_corr_pairs_s
        ],
        "Feature to drop": features_to_drop_s,
    }
)
print(tabulate(drop_table_s, headers="keys", tablefmt="pretty"))
data_corr_final = data_VS_corr.drop(columns=features_to_drop_s)
summary_dcf = data_corr_final.describe(include="all").transpose()
summary_dcf = summary_dcf.drop(columns=["25%", "50%", "75%"])
summary_dcf.index.name = "Featnames"
print(tabulate(summary_dcf, headers="keys", tablefmt="pretty"))

# %%
# DFS with featuretools
trans_primitives = [
    "add_numeric",
    "subtract_numeric",
    "divide_numeric",
    "multiply_numeric",
]
agg_primitives = ["mean", "std", "min", "max", "count"]
LLEed_data["index_col"] = range(1, len(LLEed_data) + 1)
es = ft.EntitySet(id="DataBase")
es = es.add_dataframe(
    dataframe_name="TableEntity", dataframe=LLEed_data, index="index_col"
)
print(es)
# noinspection UnusedPrimitiveWarning
features, feature_names = ft.dfs(
    entityset=es,
    target_dataframe_name="TableEntity",
    max_depth=1,
    trans_primitives=trans_primitives,
    agg_primitives=agg_primitives,
)
data_with_dfs = features.reset_index()
print("Number of features:", len(feature_names))
print(f"Original DFS_data shape: {data_with_dfs.shape}")
cols_to_drop = [col for col in data_with_dfs.columns if "index_col" in col]
data_with_dfs = data_with_dfs.drop(columns=cols_to_drop)
print(f"Shape of DFS_data after dropping index column: {data_with_dfs.shape}")
nan_count = data_with_dfs.isna().sum()
inf_count = (data_with_dfs == np.inf).sum()
nan_columns = nan_count[nan_count >= 1000]
if not nan_columns.empty:
    print("Columns with at least 1000 NaN values:")
    print(nan_columns)
inf_columns = inf_count[inf_count >= 545]
if not inf_columns.empty:
    print("Columns with at least 545 infinite values:")
    print(inf_columns)
data_with_dfs_onlyNaN = data_with_dfs.replace([np.inf, -np.inf], np.nan)
nan_count_onlyNaN = data_with_dfs_onlyNaN.isna().sum()
inf_count_onlyNaN = (data_with_dfs_onlyNaN == np.inf).sum()
nan_columns_onlyNaN = nan_count_onlyNaN[nan_count_onlyNaN >= 1125]
if not nan_columns_onlyNaN.empty:
    print("Columns with at least 1125 NaN values:")
    print(nan_columns_onlyNaN)
inf_columns_onlyNaN = inf_count_onlyNaN[inf_count_onlyNaN >= 545]
if not inf_columns_onlyNaN.empty:
    print("Columns with at least 545 infinite values:")
    print(inf_columns_onlyNaN)
data_dfs_var = data_with_dfs_onlyNaN.dropna(axis=1)
print(f"Original DFS_data shape: {data_with_dfs_onlyNaN.shape}")
print(f"Shape of data after removing NaN & inc columns: {data_dfs_var.shape}")
vari_threshold = 0.05
selector2 = VarianceThreshold(threshold=vari_threshold)
_ = selector2.fit_transform(data_dfs_var)
variances2 = selector2.variances_
variance2_df = pd.DataFrame({"features": data_dfs_var.columns, "variances": variances2})
variance2_df = variance2_df.sort_values(by="variances", ascending=False)
selected_features2 = data_dfs_var.columns[selector2.get_support(indices=True)]
final_features2 = list(set(selected_features2))
data_DFS_var = pd.DataFrame(data_dfs_var[final_features2], columns=final_features2)
print(f"Filtered dataset size: {data_DFS_var.shape[0], data_DFS_var.shape[1]}")

# %%
# Iterative correlation-based filtering to exclude very strongly correlated features
data_dfs_varcor = stats.iterative_correlation_dropper(
    data_DFS_var, vs_threshold, variance2_df
)

# %%
# Iterative correlation-based filtering to exclude strongly correlated features
data_dfs_varcor = stats.iterative_correlation_dropper(
    data_dfs_varcor, s_threshold, variance2_df
)

print(f"Before dataset size: {data_DFS_var.shape[0], data_DFS_var.shape[1]}")
print(f"After dataset size: {data_dfs_varcor.shape[0], data_dfs_varcor.shape[1]}")


# %%
# Scaling
scaler2 = MinMaxScaler()
data_dfs_scaled = scaler2.fit_transform(data_dfs_varcor)
data_dfs_scaled = pd.DataFrame(data_dfs_scaled, columns=data_dfs_varcor.columns)
summary_dfsmms = data_dfs_scaled.describe(include="all").transpose()
summary_dfsmms = summary_dfsmms.drop(columns=["25%", "50%", "75%"])
summary_dfsmms.index.name = "Fnames"
print(tabulate.tabulate(summary_dfsmms, headers="keys", tablefmt="pretty"))

# %%
# Baseline k-means
kmeans_01 = KMeans(n_clusters=5, random_state=44)
labels_01 = kmeans_01.fit_predict(data_dfs_scaled)
centers_01 = kmeans_01.cluster_centers_
cluster_counts = pd.Series(labels_01).value_counts()
print(cluster_counts)
data_with_clusters = data_dfs_scaled.copy()
data_with_clusters["Cluster"] = labels_01
radii = []
for i, center in enumerate(centers_01):
    # Extracting points belonging to the current cluster
    cluster_points = data_dfs_scaled[labels_01 == i]
    # Computing distances to the cluster center
    distances = cdist(cluster_points, [center])
    # Calculating the maximum radius (farthest point)
    max_radius = distances.max()
    radii.append(max_radius)
print("Radii for each cluster:", radii)

# %%
# Feature importance calculation
X = data_with_clusters.drop(columns=["Cluster"])
y = data_with_clusters["Cluster"].astype("category")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=43
)
model = RandomForestClassifier(random_state=41)
model.fit(X_train, y_train)
importances = model.feature_importances_
feature_importances = pd.DataFrame(
    {"features": X.columns, "importance": importances}
).sort_values(by="importance", ascending=False)
print(
    tabulate(feature_importances.head(20), headers="keys", tablefmt="pretty")
)

# %%
# QA calculation clustering
QA_name_01 = "Base_km_5c_44"
QA_labels_01 = labels_01
QA_data_01 = data_dfs_scaled
QA_cc_01 = centers_01
QA_model_01 = kmeans_01
QA_results = pd.DataFrame(
    columns=[
        "Clustering Name",
        "Noise Percentage",
        "Weighted WSS",
        "Silhouette Score",
        "BIC",
    ]
)
QA_metrics_01 = cluster.calculate_clustering_metrics(
    QA_name_01, QA_labels_01, QA_data_01, QA_cc_01, QA_model_01
)
QA_metrics_01_df = pd.DataFrame([QA_metrics_01])
if QA_results.empty:
    QA_results = QA_metrics_01_df.dropna(axis=1, how="all")
    print(
        f"Initialized QA_results with the first QA entry: '{QA_metrics_01_df['Clustering Name'].iloc[0]}'."
    )
else:
    if (
            QA_metrics_01_df["Clustering Name"].iloc[0]
            in QA_results["Clustering Name"].values
    ):
        print(
            f"QA entry with Clustering Name '{QA_metrics_01_df['Clustering Name'].iloc[0]}' already exists."
        )
    else:
        QA_metrics_01_df = QA_metrics_01_df.dropna(axis=1, how="all")
        QA_results = pd.concat([QA_results, QA_metrics_01_df], ignore_index=True)
        print(
            f"Added new QA entry for Clustering Name '{QA_metrics_01_df['Clustering Name'].iloc[0]}'."
        )

# %%
# Exploring the data to get information about ranges (and a possible further round of scaling)
summary_df = data_dfs_scaled.describe(include="all").transpose()
summary_df = summary_df.drop(columns=["25%", "50%", "75%"])
summary_df.index.name = "Featnames"
summary_df["range"] = summary_df["max"] - summary_df["min"]
sorted_summary = summary_df.sort_values(by="range", ascending=False)
sorted_summary = sorted_summary.drop(columns=["std", "count"])
largest_ranges = sorted_summary.head(10)
print("Features with the 10 Largest Ranges:")
print(tabulate(largest_ranges, headers="keys", tablefmt="pretty"))
selected_components = set()
selected_features = []
for _, row in feature_importances.iterrows():
    feature_name = row["features"]
    feature_importance = row["importance"]
    components = set(feature_name.replace("*", "+").replace("-", "+").split(" + "))
    if not components & selected_components:
        selected_features.append(feature_name)
        selected_components.update(components)
    if len(selected_features) == 7:
        break
print("Selected features:", selected_features)

# %%
# Building the main df with the top 7 selected attributes based on feature importance
existing_features = [
    feature for feature in selected_features if feature in data_dfs_scaled.columns
]
filtered_data = data_dfs_scaled[existing_features]
summary_dfsfilt = filtered_data.describe(include="all").transpose()
summary_dfsfilt = summary_dfsfilt.drop(columns=["25%", "50%", "75%"])
summary_dfsfilt.index.name = "Filtnames"
print(tabulate(summary_dfsfilt, headers="keys", tablefmt="pretty"))

# %%
# DBSCANning
X = filtered_data.values
dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusterer.fit(X)
labels_DBSCAN = dbscan_clusterer.labels_
label_map = {-1: "Noise"}
categories = [label_map.get(label, f"Cluster {label}") for label in labels_DBSCAN]
data_with_DBSCAN = filtered_data.copy()
data_with_DBSCAN["Cluster"] = labels_DBSCAN
num_noise = np.sum(labels_DBSCAN == -1)
print(f"Noise points: {num_noise}")
cluster_counts = pd.Series(labels_DBSCAN).value_counts()
print("Cluster Summary:")
print(cluster_counts)

# %%
# Adding DBSCAN results to QA table
QA_name_02 = "DBSCAN_eps0.5_minsam5"
QA_labels_02 = labels_DBSCAN
QA_data_02 = filtered_data
clusters_DBSCAN = data_with_DBSCAN[data_with_DBSCAN["Cluster"] != -1]
cluster_centers_DBSCAN = clusters_DBSCAN.groupby("Cluster").mean()
QA_cc_02 = cluster_centers_DBSCAN
QA_model_02 = dbscan_clusterer
QA_metrics_02 = helper.calculate_clustering_metrics(
    QA_name_02, QA_labels_02, QA_data_02, QA_cc_02, QA_model_02
)
QA_metrics_02_df = pd.DataFrame([QA_metrics_02])
if QA_results.empty:
    QA_results = QA_metrics_02_df.dropna(axis=1, how="all")
    print(
        f"Initialized QA_results with the first QA entry: '{QA_metrics_02_df['Clustering Name'].iloc[0]}'."
    )
else:
    if (
            QA_metrics_02_df["Clustering Name"].iloc[0]
            in QA_results["Clustering Name"].values
    ):
        print(
            f"QA entry with Clustering Name '{QA_metrics_02_df['Clustering Name'].iloc[0]}' already exists."
        )
    else:
        QA_metrics_02_df = QA_metrics_02_df.dropna(axis=1, how="all")
        QA_results = pd.concat([QA_results, QA_metrics_02_df], ignore_index=True)
        print(
            f"Added new QA entry for Clustering Name '{QA_metrics_02_df['Clustering Name'].iloc[0]}'."
        )

# %%
# Print description of clusters
print(data_with_DBSCAN["Cluster"].describe())

# %%
# HDBSCANning
X2 = filtered_data.values
clusterer2 = HDBSCAN(min_cluster_size=10, min_samples=15, cluster_selection_epsilon=0.0)
clusterer2.fit(X2)
labels_HDBSCAN = clusterer2.labels_
probabilities_HDBSCAN = clusterer2.probabilities_
label_map = {-1: "Noise", -2: "Infinite Values", -3: "Missing Values"}
categories = [label_map.get(label, f"Cluster {label}") for label in labels_HDBSCAN]
data_with_HDBSCAN = filtered_data.copy()
data_with_HDBSCAN["Cluster"] = labels_HDBSCAN
if probabilities_HDBSCAN is not None:
    data_with_HDBSCAN["Probability"] = probabilities_HDBSCAN
num_noise = np.sum(labels_HDBSCAN == -1)
num_infinite = np.sum(labels_HDBSCAN == -2)
num_missing = np.sum(labels_HDBSCAN == -3)
print(f"Noise points: {num_noise}")
print(f"Entries with infinite values: {num_infinite}")
print(f"Entries with missing values: {num_missing}")
cluster_counts = pd.Series(labels_HDBSCAN).value_counts()
print("Cluster summary:")
print(cluster_counts)
print("Cluster size range:", cluster_counts.min(), "to", cluster_counts.max())

# %%
# Probability check
print(data_with_HDBSCAN["Cluster"].describe())
print(data_with_HDBSCAN["Probability"].describe())
unique_prob = np.unique(probabilities_HDBSCAN)
count_zero_prob = (data_with_HDBSCAN["Probability"] == 0).sum()
count_less_than_0_9 = (data_with_HDBSCAN["Probability"] < 0.9).sum()
count_less_than_0_9_but_more_than_0 = count_less_than_0_9 - count_zero_prob
print(
    f"Number of entries with probability less than 0.9 (excluding noise points): {count_less_than_0_9_but_more_than_0}"
)

# %%
# Adding first HDBSCAN results to QA table
QA_name_03 = "HDBSCAN_eps0_minsam15_minclsize10"
QA_labels_03 = labels_HDBSCAN
QA_data_03 = filtered_data
clusters_HDBSCAN = filtered_data[data_with_HDBSCAN["Cluster"] != -1]
cluster_centers_HDBSCAN = clusters_HDBSCAN.groupby(data_with_HDBSCAN["Cluster"]).mean()
# clusters_HDBSCAN = data_with_HDBSCAN[data_with_HDBSCAN['Cluster'] != -1]
# cluster_centers_HDBSCAN = clusters_HDBSCAN.groupby('Cluster').mean()
QA_cc_03 = cluster_centers_HDBSCAN
QA_model_03 = clusterer2
QA_metrics_03 = helper.calculate_clustering_metrics(
    QA_name_03, QA_labels_03, QA_data_03, QA_cc_03, QA_model_03
)
QA_metrics_03_df = pd.DataFrame([QA_metrics_03])
if QA_results.empty:
    QA_results = QA_metrics_03_df.dropna(axis=1, how="all")
    print(
        f"Initialized QA_results with the first QA entry: '{QA_metrics_03_df['Clustering Name'].iloc[0]}'."
    )
else:
    if (
            QA_metrics_03_df["Clustering Name"].iloc[0]
            in QA_results["Clustering Name"].values
    ):
        print(
            f"QA entry with clustering '{QA_metrics_03_df['Clustering Name'].iloc[0]}' already exists."
        )
    else:
        QA_metrics_03_df = QA_metrics_03_df.dropna(axis=1, how="all")
        QA_results = pd.concat([QA_results, QA_metrics_03_df], ignore_index=True)
        print(
            f"Added new QA entry for clustering '{QA_metrics_03_df['Clustering Name'].iloc[0]}'."
        )

# %%
# HDBSCANning (with mcs 50)
X3 = filtered_data.values
clusterer3 = HDBSCAN(min_cluster_size=50, min_samples=15, cluster_selection_epsilon=0.0)
clusterer3.fit(X3)
labels_HDBSCAN_2 = clusterer3.labels_
probabilities_HDBSCAN_2 = clusterer3.probabilities_
label_map = {-1: "Noise", -2: "Infinite Values", -3: "Missing Values"}
categories = [label_map.get(label, f"Cluster {label}") for label in labels_HDBSCAN_2]
data_with_HDBSCAN_2 = filtered_data.copy()
data_with_HDBSCAN_2["Cluster"] = labels_HDBSCAN_2
if probabilities_HDBSCAN_2 is not None:
    data_with_HDBSCAN_2["Probability"] = probabilities_HDBSCAN_2
num_noise = np.sum(labels_HDBSCAN_2 == -1)
num_infinite = np.sum(labels_HDBSCAN_2 == -2)
num_missing = np.sum(labels_HDBSCAN_2 == -3)
print(f"Noise points: {num_noise}")
print(f"Entries with infinite values: {num_infinite}")
print(f"Entries with missing values: {num_missing}")
cluster_counts = pd.Series(labels_HDBSCAN_2).value_counts()
print("Cluster Summary:")
print(cluster_counts)
print("Cluster size range:", cluster_counts.min(), "to", cluster_counts.max())
print(data_with_HDBSCAN_2["Cluster"].describe())
print(data_with_HDBSCAN_2["Probability"].describe())
unique_prob = np.unique(probabilities_HDBSCAN)
count_zero_prob = (data_with_HDBSCAN_2["Probability"] == 0).sum()
count_less_than_0_9 = (data_with_HDBSCAN_2["Probability"] < 0.9).sum()
count_less_than_0_9_but_more_than_0 = count_less_than_0_9 - count_zero_prob
print(
    f"Number of entries with probability less than 0.9 (excluding noise points): {count_less_than_0_9_but_more_than_0}"
)
QA_name_04 = "HDBSCAN_eps0_minsam15_minclsize50"
QA_labels_04 = labels_HDBSCAN_2
QA_data_04 = filtered_data
clusters_HDBSCAN_2 = filtered_data[data_with_HDBSCAN_2["Cluster"] != -1]
cluster_centers_HDBSCAN_2 = clusters_HDBSCAN_2.groupby(
    data_with_HDBSCAN_2["Cluster"]
).mean()
QA_cc_04 = cluster_centers_HDBSCAN_2
QA_model_04 = clusterer3
QA_metrics_04 = helper.calculate_clustering_metrics(
    QA_name_04, QA_labels_04, QA_data_04, QA_cc_04, QA_model_04
)
QA_metrics_04_df = pd.DataFrame([QA_metrics_04])
if QA_results.empty:
    QA_results = QA_metrics_04_df.dropna(axis=1, how="all")
    print(
        f"Initialized QA_results with the first QA entry: '{QA_metrics_04_df['Clustering Name'].iloc[0]}'."
    )
else:
    if (
            QA_metrics_04_df["Clustering Name"].iloc[0]
            in QA_results["Clustering Name"].values
    ):
        print(
            f"QA entry with clustering '{QA_metrics_04_df['Clustering Name'].iloc[0]}' already exists."
        )
    else:
        QA_metrics_04_df = QA_metrics_04_df.dropna(axis=1, how="all")
        QA_results = pd.concat([QA_results, QA_metrics_04_df], ignore_index=True)
        print(
            f"Added new QA entry for clustering '{QA_metrics_04_df['Clustering Name'].iloc[0]}'."
        )

# %%
# HDBSCANning (with mcs 100)
X4 = filtered_data.values
clusterer4 = HDBSCAN(
    min_cluster_size=100, min_samples=15, cluster_selection_epsilon=0.0
)
clusterer4.fit(X4)
labels_HDBSCAN_3 = clusterer4.labels_
probabilities_HDBSCAN_3 = clusterer4.probabilities_
label_map = {-1: "Noise", -2: "Infinite Values", -3: "Missing Values"}
categories = [label_map.get(label, f"Cluster {label}") for label in labels_HDBSCAN_3]
data_with_HDBSCAN_3 = filtered_data.copy()
data_with_HDBSCAN_3["Cluster"] = labels_HDBSCAN_3
if probabilities_HDBSCAN_3 is not None:
    data_with_HDBSCAN_3["Probability"] = probabilities_HDBSCAN_3
num_noise = np.sum(labels_HDBSCAN_3 == -1)
num_infinite = np.sum(labels_HDBSCAN_3 == -2)
num_missing = np.sum(labels_HDBSCAN_3 == -3)
print(f"Noise points: {num_noise}")
print(f"Entries with infinite values: {num_infinite}")
print(f"Entries with missing values: {num_missing}")
cluster_counts = pd.Series(labels_HDBSCAN_3).value_counts()
print("Cluster Summary:")
print(cluster_counts)
print("Cluster size range:", cluster_counts.min(), "to", cluster_counts.max())
print(data_with_HDBSCAN_3["Cluster"].describe())
print(data_with_HDBSCAN_3["Probability"].describe())
unique_prob = np.unique(probabilities_HDBSCAN_3)
count_zero_prob = (data_with_HDBSCAN_3["Probability"] == 0).sum()
count_less_than_0_9 = (data_with_HDBSCAN_3["Probability"] < 0.9).sum()
count_less_than_0_9_but_more_than_0 = count_less_than_0_9 - count_zero_prob
print(
    f"Number of entries with probability less than 0.9 (excluding noise points): {count_less_than_0_9_but_more_than_0}"
)
QA_name_05 = "HDBSCAN_eps0_minsam15_minclsize100"
QA_labels_05 = labels_HDBSCAN_3
QA_data_05 = filtered_data
clusters_HDBSCAN_3 = filtered_data[data_with_HDBSCAN_3["Cluster"] != -1]
cluster_centers_HDBSCAN_3 = clusters_HDBSCAN_3.groupby(
    data_with_HDBSCAN_3["Cluster"]
).mean()
QA_cc_05 = cluster_centers_HDBSCAN_3
QA_model_05 = clusterer4
QA_metrics_05 = helper.calculate_clustering_metrics(
    QA_name_05, QA_labels_05, QA_data_05, QA_cc_05, QA_model_05
)
QA_metrics_05_df = pd.DataFrame([QA_metrics_05])
if QA_results.empty:
    QA_results = QA_metrics_05_df.dropna(axis=1, how="all")
    print(
        f"Initialized QA_results with the first QA entry: '{QA_metrics_05_df['Clustering Name'].iloc[0]}'."
    )
else:
    if (
            QA_metrics_05_df["Clustering Name"].iloc[0]
            in QA_results["Clustering Name"].values
    ):
        print(
            f"QA entry with clustering '{QA_metrics_05_df['Clustering Name'].iloc[0]}' already exists."
        )
    else:
        QA_metrics_05_df = QA_metrics_05_df.dropna(axis=1, how="all")
        QA_results = pd.concat([QA_results, QA_metrics_05_df], ignore_index=True)
        print(
            f"Added new QA entry for clustering '{QA_metrics_05_df['Clustering Name'].iloc[0]}'."
        )

# %%
# Choosing final clustering
data_with_HDBSCAN_2.info()
data_without_noise = data_with_HDBSCAN_2[data_with_HDBSCAN_2["Cluster"] != -1]
features = [
    col for col in data_without_noise.columns if col not in ["Cluster", "Probability"]
]
plt.figure(figsize=(12, 6))
for feature in features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Cluster", y=feature, data=data_without_noise)
    plt.title(f"Distribution of {feature} by Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.close()


# %% Step X: Building a labelled training set directly from  collated sqlite table
# -------------------------------------------------------------------------
triples = []
for sig, canon in artist_link_rows.itertuples(index=False):
    names = [s.strip() for s in sig.split("|") if s.strip()]
    if len(names) < 2:  # to ignore singletons / malformed rows
        continue
    # label = 1 if the group should be unified, 0 if it must be kept separated
    link_flag = 0 if canon.strip() == "__SKIP__" else 1
    # every unordered pair inside the group becomes one training example
    triples.extend(
        (link_flag, a, b) for a, b in itertools.combinations(names, r=2)
    )
# Creating dataframe and dropping duplicates just in case
df_pairs = (
    pd.DataFrame(triples, columns=["to_link", "A", "B"])
    .drop_duplicates()
    .sample(frac=1, random_state=43)
    .reset_index(drop=True)
)
NEG_TARGET = 500
# Build a fast lookup: group id  → list(variants)
groups = (artist_link_rows.assign(gid=range(len(artist_link_rows)))
          .explode('group_signature')
          .assign(variant=lambda d: d.group_signature.str.strip())
          .loc[lambda d: d.variant.ne('')]  # keep non‑empty
          .groupby('gid')['variant'].apply(list)
          )
rng = np.random.default_rng(43)
gids = groups.index.to_numpy()
neg_pairs = [
    (0, *rng.choice(groups[g1]), *rng.choice(groups[g2]))[:3]  # (flag,A,B)
    for g1, g2 in rng.choice(gids, size=(NEG_TARGET, 2), replace=True)
    if g1 != g2
]
df_pairs = (
    pd.concat([df_pairs, pd.DataFrame(neg_pairs, columns=df_pairs.columns)])
    .drop_duplicates()
    .sample(frac=1, random_state=43)
    .reset_index(drop=True)
)
print(f"After synthesis: {len(df_pairs)} pairs  "
      f"(pos= {df_pairs.to_link.sum()}, "
      f"neg= {len(df_pairs) - df_pairs.to_link.sum()})")


# %% Step X+1: Teaching decision rule to logistic regression or other classifier
score_df = (
    df_pairs
    .apply(lambda r: pd.Series(cluster.fuzzy_scores(r["A"], r["B"])), axis=1)
    .astype("float32")
)
full = pd.concat([df_pairs, score_df], axis=1)
full.head()
X = score_df.values
y = full["to_link"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=500, class_weight='balanced').fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
print("AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]).round(3))
# coefficients → human‑readable weights
w = pd.Series(clf.coef_[0], index=score_df.columns).round(2)
print(w.sort_values(ascending=False))

# Picking practical thresholds
full["proba"] = clf.predict_proba(X)[:, 1]  # 0‑1 similarity proxy
full = full.sort_values("proba", ascending=False)

# Empirical threshold where FP = 0
cut = full.loc[full["to_link"] == 1, "proba"].min()
print(f"Cut‑off that keeps all true links: {cut:.2f}")
print(full[["A", "B", "to_link", "proba"]].head(10))

# %% Step X+2: I WILL WRITE SOMETHING INFORMATIVE HERE IF IT WORKS
artist_names = fltrd_artcount["Artist"].str.str.strip().tolist()

match, prob = cluster.most_similar(
    "bohren & der club of gore",
    artist_names,
    clf,
    threshold=cut,
)
print(match, prob)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
