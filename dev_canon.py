# %% Basic setup
from dotenv import load_dotenv
from sklearn.svm import SVC

load_dotenv()
from DB import SessionLocal
from DB.models import ArtistVariantsCanonized
import featuretools as ft
from helpers import cli
from helpers import io
from helpers import cluster
from helpers import stats
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
from rapidfuzz import process
import re
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, classification_report, pairwise_distances, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sqlalchemy import insert, select, text, update
import sys
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")
import woodwork as ww

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

# %% Input & pre-processing
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

# %% Variant clustering
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
    variants: list[str] = [v.strip() for v in row.artist_variants_text.split("{") if v.strip()]
    for variant in _split_variants(row.artist_variants_text):
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
anchor_idx_sets = [s for s in anchor_idx_sets if len(s) >= 2]  # ignore degenerate
eps_range = np.arange(0.05, 1.0, 0.01)
best_eps = None
for eps in eps_range:
    labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
    if cluster.anchors_ok(labels, anchor_idx_sets):
        best_eps = eps
        break
if best_eps is None:
    raise ValueError("No ε in the range puts every variant_set in one cluster.")
# print(f"Chosen ε = {best_eps:.2f} (all anchors satisfied)")
# Manually updating best_eps here based on results from previous iterations,
# since, in the meantime, variants contained in "variant_sets", which plays an important part in ε identification,
# have been canonized.
best_eps = 0.23
labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
clusters = (
    pd.DataFrame({"Artist": artist_names, "label": labels})
    .query("label != -1")
    .groupby("label")["Artist"].apply(list)
    .tolist()
)

# %% Gold standard creation
# This step previously identified a lot of relevant clusters, which I manually clustered to form a gold-standard table.
# Cf. section 2.3.3 of the assignment
with SessionLocal() as sess:
    unhandled = 0
    for group in clusters:
        if len(group) <= 1:
            continue
        sig = "|".join(sorted(group))
        row = sess.execute(
            text("SELECT 1 FROM artist_variants_canonized WHERE artist_variants_text = :sig"),
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
print("Done unifying. Next steps coming.")

# %%
# Building 6 rapidfuzz function outputs
avc = pd.read_parquet("PQ/avc.parquet")
expanded_rows = []
for _, row in avc.iterrows():
    expanded_rows.extend(cluster.expand_pairs(row))
goldstandard = pd.DataFrame(expanded_rows, columns=[
    'variants', 'variant_a', 'variant_b', 'to_link'
])
score_df = goldstandard.apply(
    lambda sor: pd.Series(cluster.fuzzy_scores(sor['variant_a'], sor['variant_b'])),
    axis=1
)
goldstandard = pd.concat([goldstandard, score_df], axis=1)
goldstandard.info()
goldstandard.describe()
goldstandard["to_link"] = goldstandard["to_link"].astype("int64")
base_feats = [
    "ratio", "partial_ratio",
    "token_sort_ratio", "token_set_ratio",
    "WRatio", "QRatio"
]
gs = goldstandard[[
    "variants", "to_link",
    "ratio", "partial_ratio",
    "token_sort_ratio", "token_set_ratio",
    "WRatio", "QRatio"
]]

# %%
# Variance-pruning and checking the correlation matrix using Spearman's
X = gs.drop(columns=["variants", "to_link"])
selector = VarianceThreshold(threshold=0.01)
_ = selector.fit_transform(X)
variances = selector.variances_
variance_df = pd.DataFrame({
    "features": X.columns,
    "variances": variances
})
pruned_features_df = stats.iterative_correlation_dropper(
    current_data=X,
    cutoff=0.85,
    varframe=variance_df,
    min_features=3
)
gs_VTed = pd.concat([
    gs[["variants", "to_link"]],
    pruned_features_df
], axis=1)
gs_corr = pruned_features_df.corr(method="spearman")
plt.figure(figsize=(20, 18))
sns.heatmap(
    gs_corr,
    annot=True,
    annot_kws={"size": 8},
    cmap="winter",
    fmt=".2f",
    cbar=True,
    linewidths=0.5,
    mask=np.triu(gs_corr),
)
plt.title("")
plt.xticks(rotation=90)

# %% FTing
gs_VTed_num = gs_VTed.drop("variants", axis='columns')
gs_VTed_num = gs_VTed_num.drop("to_link", axis='columns')
gs_VTed_num["pair_id"] = gs_VTed_num.index
gs_VTed_num.ww.init(
    name="similarities",
    index="pair_id",
    logical_types={col: 'Double' for col in gs_VTed_num.columns if col != "pair_id"}
)
es = ft.EntitySet(id="artist_pairs")
es = es.add_dataframe(
    dataframe_name="similarities",
    dataframe=gs_VTed_num,
    index="pair_id"
)
trans_primitives = ["add_numeric", "subtract_numeric", "divide_numeric", "multiply_numeric"]
features, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="similarities",
    trans_primitives=trans_primitives,
    max_depth=1
)
features.reset_index(drop=True, inplace=True)
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(axis=1, how='any', inplace=True)
print(features.head())
print(f"Final features shape: {features.shape}")

# %%
# Identifying attributes with very strong correlation
X = features
selector = VarianceThreshold(threshold=0.01)
_ = selector.fit_transform(X)
variances = selector.variances_
variance_df = pd.DataFrame({
    "features": X.columns,
    "variances": variances
})
pruned_features_df = stats.iterative_correlation_dropper(
    current_data=X,
    cutoff=0.85,
    varframe=variance_df,
    min_features=3
)

# %%
# Recalculating for strong correlations
features = pruned_features_df
corr_matrix_shrunken = features.corr(method="spearman")
s_threshold = 0.75
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
data_corr_final = features.drop(columns=features_to_drop_s)
summary_dcf = data_corr_final.describe(include="all").transpose()
summary_dcf = summary_dcf.drop(columns=["25%", "50%", "75%"])
summary_dcf.index.name = "Featnames"
print(tabulate(summary_dcf, headers="keys", tablefmt="pretty"))

# %%
# Scaling
scaler2 = MinMaxScaler()
data_dfs_scaled = scaler2.fit_transform(data_corr_final)
data_dfs_scaled = pd.DataFrame(data_dfs_scaled, columns=data_corr_final.columns)
summary_dfsmms = data_dfs_scaled.describe(include="all").transpose()
summary_dfsmms = summary_dfsmms.drop(columns=["25%", "50%", "75%"])
summary_dfsmms.index.name = "Fnames"
print(tabulate(summary_dfsmms, headers="keys", tablefmt="pretty"))

# %%
# Feature importance calculation
X = data_dfs_scaled
y = gs_VTed["to_link"].astype("category")
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
# Use only the features with the 3 highest feature importance score
TOP_K = 3
TOP_K = min(TOP_K, len(feature_importances))
selected_base = (
    feature_importances  # DataFrame with cols ["features","importance"]
    .head(TOP_K)  # to keep the K highest
    .loc[:, "features"]  # to take the names only
    .tolist()
)
print(f"Selected top-{TOP_K} features:", ", ".join(selected_base))
df_selected = data_dfs_scaled[selected_base].copy()
final_ft_df = pd.concat(
    [gs[['variants', 'to_link']].reset_index(drop=True), df_selected.reset_index(drop=True)],
    axis=1
)
selected_comp_feats_plus_target = final_ft_df.drop("variants", axis='columns')
(Path("JSON") / "selected_features.json").write_text(
    json.dumps({"top_k": TOP_K, "features": selected_base}, indent=2)
)

# %% Let's now try a decision tree-based approach
X = df_selected
y = final_ft_df["to_link"]
tree = DecisionTreeClassifier(max_depth=2, random_state=44)
tree.fit(X, y)
print(export_text(tree, feature_names=X.columns.tolist()))
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=X.columns.tolist(), class_names=["no link", "link"], filled=True)

# %%
# Schema to test deterministic idea
rules = cluster.tree_to_rule_list(tree, X.columns)
print("\nExtracted rule set:")
for cond, p1 in rules:
    print(f"If {cond}  =>  P(to_link=1)={p1:.2f}")
final_ft_df["auto_unify"] = 0
for cond, _ in rules:
    final_ft_df.loc[final_ft_df.eval(cond), "auto_unify"] = 1
y_true = final_ft_df["to_link"]
y_pred = final_ft_df["auto_unify"]
cm = confusion_matrix(final_ft_df["to_link"], final_ft_df["auto_unify"])
cm_df = pd.DataFrame(cm,
                     index=["Actual 0", "Actual 1"],
                     columns=["Pred 0", "Pred 1"])
print("\nClassification report:")
print(classification_report(final_ft_df["to_link"],
                            final_ft_df["auto_unify"],
                            target_names=["no link", "link"]))
cm_percent = (cm / cm.sum(axis=1, keepdims=True)) * 100

# %%
# Checking misclassifications
mis_mask_tree = final_ft_df["to_link"] != final_ft_df["auto_unify"]
mis_tree = gs.loc[mis_mask_tree, ["variants", "to_link"] + base_feats].copy()
mis_tree["predicted"] = final_ft_df.loc[mis_mask_tree, "auto_unify"].values
print("\n🚨  Tree mis-classified rows:")
print(tabulate(mis_tree.head(20), headers="keys", tablefmt="pretty"))

# %% Let"s try a decision tree on the original fuzz scores
RATIO_TH = 0.865
full = pd.concat(
    [gs[["variants", "to_link"]], gs[base_feats]],
    axis=1
)
full["auto_unify"] = 0
full.loc[full["ratio"] > RATIO_TH, "auto_unify"] = 1  # assigning only the col
stats.show_cm_and_report(full["to_link"], full["auto_unify"],
                         title=f"Single-feature rule (ratio > {RATIO_TH})")
mis_mask_rule = full["to_link"] != full["auto_unify"]
mis_rule = full.loc[mis_mask_rule, ["variants", "to_link"] + base_feats + ["auto_unify"]]
print("\nRule mis-classified rows:")
print(tabulate(mis_rule.head(20), headers="keys", tablefmt="pretty"))

# %%
# SVM experiment (linear kernel)
X = gs[base_feats].values.astype("float32")
y = gs["to_link"].values
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, gs.index, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
svm = SVC(kernel="linear", class_weight="balanced",
          probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
print("\n=== SVM on base RapidFuzz scores ===")
stats.show_cm_and_report(y_test, svm.predict(X_test_sc))

# mis-classified rows in training set
train_pred = svm.predict(X_train_sc)
mis_mask_svm = train_pred != y_train
mis_svm = gs.loc[idx_train[mis_mask_svm], ["variants", "to_link"] + base_feats].copy()
mis_svm["predicted"] = train_pred[mis_mask_svm]
print("\nSVM mis-classified training rows:")
print(tabulate(mis_svm.head(20), headers="keys", tablefmt="pretty"))

# weights for interpretability
coef_series = pd.Series(svm.coef_[0], index=base_feats).round(3)
print("\nSVM linear weights:")
print(tabulate(coef_series.sort_values(ascending=False).to_frame("weight"),
               headers="keys", tablefmt="pretty"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
