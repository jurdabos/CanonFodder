from dotenv import load_dotenv
load_dotenv()
from DB import engine, SessionLocal
import helper
import json
import logging
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["MPLBACKEND"] = "TkAgg"
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sqlalchemy import select, text
import sys
HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "JSON"
PALETTES_FILE = JSON_DIR / "palettes.json"
os.environ.pop("FLASK_APP", None)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# %% Pandas display options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{: .2f}".format

# %% display options for a specific style
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = helper.register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# %% Step 1: Loading the data from the latest parquet file
print("===========================================")
print("Welcome to the CanonFodder data profiling workflow!")
print("We'll load your scrobble data, apply any previously saved artist name unifications,")
print("then explore on forward.")
print("===========================================\n")
logging.info("I am happy for you being here!")
print("Step 1: Loading and cleaning your scrobble data. Please wait...")
data, latest_filename = helper.load_latest_parquet_in_pq()
if data is None or data.empty:
    print("No DataFrame was loaded; cannot proceed with EDA.")
    sys.exit()
# Naming the columns
data.columns = ["Artist", "Album", "Song", "Datetime", "Country"]
data.info()
helper.missing_value_ratio(data)
data.dropna(subset=["Datetime"], inplace=True)

# %% Step 2: Checking for duplicates
before_count = len(data)
data = data.drop_duplicates(subset=["Artist", "Album", "Song", "Datetime"], keep="first")
after_count = len(data)
removed_count = before_count - after_count
logging.info(f"Removed {removed_count} duplicate rows.")

# %% Step 3: Connect to the DB and apply previously stored unifications/canonizations
with SessionLocal() as sess:
    logging.info("Fetching all previously handled groups …")
    handled = sess.execute(
        text("""
             SELECT group_signature, canonical_name
             FROM groups_handled
             WHERE canonical_name != '__SKIP__'
        """)
    ).all()
for group_sig, canonical_name in handled:
    # Converting the signature back to a list of old names
    old_names = group_sig.split("|")
    for old_name in old_names:
        if old_name != canonical_name:
            data.loc[data["Artist"] == old_name, "Artist"] = canonical_name

# %% Step 4: Now do EDA tasks on the already-updated data
artist_counts = data["Artist"].value_counts()
top_artists = artist_counts.head(10)
print(top_artists)
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors_10 = helper.register_custom_palette("colorpalette_10", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors_10))
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_artists.values,
    y=top_artists.index,
    palette=custom_colors_10[: len(top_artists)],
    hue=top_artists.index,
    legend=False,
)
plt.title("Top 10 Artists by Scrobbles", fontsize=16)
plt.xlabel("Scrobbles", fontsize=14)
plt.ylabel("Artist", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Analyzing artist counts
print(artist_counts.describe())
# Convert artist_counts to DataFrame
artist_counts_df = artist_counts.reset_index()
artist_counts_df.columns = ["Artist", "Count"]
count_threshold = 2
# Shrinking of the data set can be easily undone here by setting count_threshold to 1 in future flows
fltrd_artcount = artist_counts_df[artist_counts_df["Count"] >= count_threshold]
print(fltrd_artcount.describe())

# %%
# Artist names EDA
artist_names = fltrd_artcount["Artist"].str.lower().str.strip().tolist()
n = len(artist_names)
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = (
            fuzz.token_sort_ratio(artist_names[i], artist_names[j]) / 100
        )
bohren_indices = [i for i, name in enumerate(artist_names) if "bohren" in name]
for eps in np.arange(0.01, 1.0, 0.01):
    dbscan = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = dbscan.fit_predict(1 - similarity_matrix)
    if labels[bohren_indices[0]] == labels[bohren_indices[1]] and labels[bohren_indices[0]] != -1:
        print(f"Bohren occurrences are grouped together at epsilon: {eps}")
        break
print("\n--- Running DBSCAN to find new clusters of similar artists ---")
dbscan = DBSCAN(eps=0.12, min_samples=2, metric="precomputed")
labels = dbscan.fit_predict(1 - similarity_matrix)
clusters = {}
for label, artist in zip(labels, fltrd_artcount["Artist"]):
    if label != -1:
        clusters.setdefault(label, []).append(artist)
similar_artist_groups = list(clusters.values())
print(f"Number of groups identified: {len(similar_artist_groups)}")

# %%
# DBing artist name canonization
with SessionLocal() as sess:
    unhandled = 0
    for group in similar_artist_groups:
        if len(group) <= 1:
            continue
        sig = "|".join(sorted(group))
        row = sess.execute(
            text("SELECT 1 FROM groups_handled WHERE group_signature = :sig"),
            {"sig": sig},
        ).first()
        if row is None:
            unhandled += 1

print(f"Number of groups identified by DBSCAN: {len(similar_artist_groups)}")
print(f"Number of groups NOT yet handled by user: {unhandled}")

# %%
# CLI for picking canonical names for similar artist names
data, fltrd_artcount = helper.unify_artist_names_cli(
    data=data,
    fltrd_artcount=fltrd_artcount,
    similar_artist_groups=similar_artist_groups,
)
print("Done unifying. Next steps coming later. Now exiting...")

# gender_values = data.Gender.value_counts().sort_values(ascending=False).to_frame()
# gender_values = gender_values.rename(columns={"Gender": "count"})
# print(gender_values)
sys.exit()

# %%
# EDA of Country column
country_count = data.Country.value_counts().sort_values(ascending=False).to_frame()[:15]
country_count = country_count.rename(columns={"Country": "count"})
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=country_count.index,
    y="count",
    data=country_count,
    palette="Spectral",
    hue=country_count.index,
)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=10,
        color="black",
    )
ax.set_xticks(range(len(country_count)))
ax.set_xticklabels(country_count.index, rotation=45, ha="right", fontsize=12)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_title("Top 15 Countries", fontsize=16)
ax.set_xlabel("Country", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()
plt.show()
plt.close()


# %%
# Calculating and visualizing Cramér's V
categorical_columns = data.select_dtypes(include=["object"]).columns
cramers_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 == col2:
            cramers_matrix.loc[col1, col2] = 1.0
        else:
            cramers_matrix.loc[col1, col2] = helper.cramers_v(data[col1], data[col2])
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
# Getting a printout of the unique values in benefits, care_options, seek_help and wellness_program
benefits_values = data.benefits.value_counts().sort_values(ascending=False).to_frame()
benefits_values = benefits_values.rename(columns={"Benefits": "count"})
print(benefits_values)
care_options_values = (
    data.care_options.value_counts().sort_values(ascending=False).to_frame()
)
care_options_values = care_options_values.rename(columns={"Care options": "count"})
print(care_options_values)
seek_help_values = data.seek_help.value_counts().sort_values(ascending=False).to_frame()
seek_help_values = seek_help_values.rename(columns={"Seek help": "count"})
print(seek_help_values)
wellness_program_values = (
    data.wellness_program.value_counts().sort_values(ascending=False).to_frame()
)
wellness_program_values = wellness_program_values.rename(
    columns={"Wellness program": "count"}
)
print(wellness_program_values)

# %%
# Aggregating features benefits, care_options, seek_help and wellness_program into support_available
mapping = {"No": 0, "Don't know": 0.125, "Not sure": 0.125, "Yes": 0.25}
data["support_available"] = (
    data["benefits"].map(mapping)
    + data["care_options"].map(mapping)
    + data["seek_help"].map(mapping)
    + data["wellness_program"].map(mapping)
)

# %%
# Getting a printout of the unique values in support_available
support_available_values = (
    data.support_available.value_counts().sort_values(ascending=False).to_frame()
)
support_available_values = support_available_values.rename(
    columns={"Support available": "count"}
)
print(support_available_values)

# %%
# Dropping columns with cross-association related to available support (based on Cramér's V)
data.drop(
    ["benefits", "care_options", "seek_help", "wellness_program"], axis=1, inplace=True
)

# %%
# Aggregating features coworkers, mental_health_consequence, and supervisor into disc_MH_atwork
mapping = {"No": 0, "Some of them": 1 / 6, "Maybe": 1 / 6, "Yes": 1 / 3}
data["disc_MH_atwork"] = (
    data["coworkers"].map(mapping)
    + data["mental_health_consequence"].map(mapping)
    + data["supervisor"].map(mapping)
)

# %%
# Getting a printout of the unique values in disc_MH_atwork
disc_MH_atwork_values = (
    data.disc_MH_atwork.value_counts().sort_values(ascending=False).to_frame()
)
disc_MH_atwork_values = disc_MH_atwork_values.rename(
    columns={"Discussing mental health at work": "count"}
)
print(disc_MH_atwork_values)

# %%
# Dropping columns with cross-association related to discussing mental health at work (based on Cramér's V)
data.drop(
    ["coworkers", "mental_health_consequence", "supervisor"], axis=1, inplace=True
)

# %%
# Saving the altered data table after categorical feature selection
# filename = f'''./data_versions/survey_aftercfs_{datetime.now().strftime(
#     "%Y%m%d_%H%M%S")}.csv'''
# data.to_csv(filename, index=False)

# %%
# Creating an informative summary DataFrame for categorical features before one-hot encoding
categorical_features = data.select_dtypes(include=["object"])
cat_summary = pd.DataFrame(
    {
        "feature": categorical_features.columns,
        "unique": [categorical_features[col].nunique() for col in categorical_features],
        "categories": [
            list(categorical_features[col].unique()) for col in categorical_features
        ],
    }
)
print(tabulate.tabulate(cat_summary, headers="keys", tablefmt="pretty"))

# %%
# Function call to add a grid-based table to Word for case study
# helper.save_as_word_table(cat_summary,
#                           "./data_versions/categorical_summary_with_grid.docx")

# %%
# Starting 1HE by specifying categorical columns for memory and performance benefits
categorical_columns = [
    "family_history",
    "Gender",
    "Country",
    "self_employed",
    "remote_work",
    "treatment",
    "phys_health_consequence",
    "mental_health_interview",
    "phys_health_interview",
    "obs_consequence",
    "anonymity",
    "leave",
    "mental_vs_physical",
]
data[categorical_columns] = data[categorical_columns].astype("category")

# %% One-hot-encoding categorical features and passing in short label list for the prefixes
data_1hot = pd.get_dummies(
    data[categorical_columns],
    prefix=[helper.short_labels[col] for col in categorical_columns],
    drop_first=False,
)

# %%
# Building the data table with encoded categorical features
non_categorical_columns = data.select_dtypes(include=["number"]).columns
data_encoded = pd.concat([data[non_categorical_columns], data_1hot], axis=1)
data_encoded.drop(columns=categorical_columns, inplace=True, errors="ignore")

# %%
# Converting everything to float and saving the altered data table
data_encoded_float = data_encoded.astype("float64")
# filename = f'''./data_versions/survey_afterhotenc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'''
# data_encoded_float.to_csv(filename, index=False)

# %%
# Verifying the resulting dataframe
summary = data_encoded_float.describe(include="all").transpose()
summary = summary.drop(columns=["25%", "50%", "75%"])
print(tabulate.tabulate(summary, headers="keys", tablefmt="pretty"))

# %%
# Robust scaling
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data_encoded_float)
scaled_data = pd.DataFrame(scaled_data, columns=data_encoded_float.columns)

# %%
# Verifying the resulting dataframe
summary_robsca = scaled_data.describe(include="all").transpose()
summary_robsca = summary_robsca.drop(columns=["25%", "50%", "75%"])
summary_robsca.index.name = "Fn"
print(tabulate.tabulate(summary_robsca, headers="keys", tablefmt="pretty"))

# %%
# Setting up and fitting for variance testing
threshold = 0.01
variance_df, selected_features = helper.variance_testing(scaled_data, threshold)
threshold = 0.05
variance_df, selected_features = helper.variance_testing(scaled_data, threshold)

# %%
# Manually retaining specific features
manually_retained_features = [
    "Continent_Asia",
    "Continent_Africa",
    "Continent_Australia and Oceania",
    "Gen_other",
]
final_features = list(set(selected_features).union(manually_retained_features))
data_VT = pd.DataFrame(scaled_data[final_features], columns=final_features)

# %%
# Creating barplot visual based on variances
colors = ["teal" if var >= threshold else "salmon" for var in variance_df["variances"]]
plt.figure(figsize=(12, 6))
plt.bar(x=variance_df["features"], height=variance_df["variances"], color=colors)
plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
plt.xticks(rotation=90)
plt.ylabel("Feature variance")
plt.title("Feature variance comparison")
plt.legend()
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
# Setting threshold for very strong
vs_threshold = 0.8
mask = np.triu(np.ones(vs_corr.shape), k=1)

# %%
# Identifying attributes with very strong correlation
vs_corr_pairs, features_to_drop_vs = helper.drop_high_corr_features(
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
print(tabulate.tabulate(drop_table, headers="keys", tablefmt="pretty"))

# %%
# Dropping features with very strong correlation
data_VS_corr = data_VT.drop(columns=features_to_drop_vs)

# %%
# Recalculating for strong correlations
corr_matrix_shrunken = data_VS_corr.corr(method="spearman")
s_threshold = 0.6
high_corr_pairs_s, features_to_drop_s = helper.drop_high_corr_features(
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
print(tabulate.tabulate(drop_table_s, headers="keys", tablefmt="pretty"))

# %%
# Dropping the features identified in the second iteration
data_corr_final = data_VS_corr.drop(columns=features_to_drop_s)

# %%
# Verifying the resulting dataframe
summary_dcf = data_corr_final.describe(include="all").transpose()
summary_dcf = summary_dcf.drop(columns=["25%", "50%", "75%"])
summary_dcf.index.name = "Featnames"
print(tabulate.tabulate(summary_dcf, headers="keys", tablefmt="pretty"))

# %%
# Preparing a target variable for 2D MDS
feature_name = "Age"
target_variable = data_corr_final[feature_name]

# %%
# Applying 2D MDS to visualize the data
mds = MDS(2, random_state=0)
data_2d = mds.fit_transform(data_corr_final)
plt.figure(figsize=(10, 6))
plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=target_variable, cmap="winter", s=50)
plt.colorbar(label=f"Scaled {feature_name} feature")
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title(f"2D MDS projection with coloring by {feature_name}")
plt.show()
plt.close()

# %%
# Applying 3D MDS to visualize the data
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mds = MDS(3, random_state=1)
data_3d = mds.fit_transform(data_corr_final)
scat = ax.scatter(
    data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=target_variable, cmap="winter", s=50
)
cbar = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Robustly scaled version of Age")
plt.title("3D MDS projection")
plt.show()
plt.close()

# %%
# Locating outliers in the 2D plot w/ more than 3 stds from the mean
distances_2 = np.sqrt(np.sum(data_2d**2, axis=1))
threshold_2 = distances_2.mean() + 3 * distances_2.std()
outlier_indices_2 = np.where(distances_2 > threshold_2)[0]
print(f"Outlier Indices: {outlier_indices_2}")
print(f"Outlier Distances: {distances_2[outlier_indices_2]}")

# %%
# Locating outliers in the 3D plot w/ more than 3 stds from the mean
distances_3 = np.sqrt(np.sum(data_3d**2, axis=1))
threshold_3 = distances_3.mean() + 3 * distances_3.std()
outlier_indices_3 = np.where(distances_3 > threshold_3)[0]
print(f"Outlier Indices: {outlier_indices_3}")
print(f"Outlier Distances: {distances_3[outlier_indices_3]}")

# %%
# Investigating outlier data for outlier_indices_3, which contains all values of outlier_indices_2
outlier_data = data_corr_final.iloc[outlier_indices_3]
same_value_columns = []
for col in outlier_data.columns:
    unique_values = outlier_data[col].nunique()
    if unique_values == 1:
        same_value_columns.append(col)
print("Columns where all outlier entries have the same value:")
for col in same_value_columns:
    common_value = outlier_data[col].iloc[0]
    print(f"Column: {col}; common value: {common_value}")

# %%
# Removing outliers
print(f"Before dataset size: {data_corr_final.shape[0], data_corr_final.shape[1]}")
MDS_clean_rows = data_corr_final.drop(index=outlier_indices_3)
MDS_clean_data = MDS_clean_rows.drop(same_value_columns, axis=1)
print(f"After dataset size: {MDS_clean_data.shape[0], MDS_clean_data.shape[1]}")

# %%
# Verifying the resulting dataframe
summary_MDS = MDS_clean_data.describe(include="all").transpose()
summary_MDS = summary_MDS.drop(columns=["25%", "50%", "75%"])
summary_MDS.index.name = "Data after MDS clean"
print(tabulate.tabulate(summary_MDS, headers="keys", tablefmt="pretty"))

# %%
# Preparing a target for 2D LLE
feature_name = "disc_MH_atwork"
LLE_2D_target = MDS_clean_data[feature_name]

# %%
# Applying 2D LLE
embedding = manifold.LocallyLinearEmbedding(
    n_neighbors=10, n_components=2, random_state=791
)
LLE_data_2d = embedding.fit_transform(MDS_clean_data)
plt.figure(figsize=(8, 6))
plt.scatter(
    LLE_data_2d[:, 0], LLE_data_2d[:, 1], c=LLE_2D_target, cmap="Spectral", s=15
)
plt.title("LLE projection of processed survey data")
plt.xlabel("LLE Component 1")
plt.ylabel("LLE Component 2")
plt.colorbar(label=f"Colored by {feature_name}")
plt.show()
plt.close()

# %%
# Preparing a target for 3D LLE
selected_feature = "support_available"
LLE_3D_target = MDS_clean_data[selected_feature]

# %%
# Applying 3D LLE
embedding_3D = manifold.LocallyLinearEmbedding(
    n_neighbors=17, n_components=3, random_state=7
)
LLE_data_3d = embedding_3D.fit_transform(MDS_clean_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scat = ax.scatter(
    LLE_data_3d[:, 0],
    LLE_data_3d[:, 1],
    LLE_data_3d[:, 2],
    c=LLE_3D_target,
    cmap="Spectral",
    s=50,
)
cbar = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10, pad=0.18)
cbar.set_label("support available")
plt.title("3D LLE projection")
plt.show()
plt.close()

# %%
# Using LLE outliers for data wrangling
threshold_2d = 2.4
threshold_3d = 2.4
mean_2d = np.mean(LLE_data_2d, axis=0)
std_2d = np.std(LLE_data_2d, axis=0)
mean_3d = np.mean(LLE_data_3d, axis=0)
std_3d = np.std(LLE_data_3d, axis=0)
outliers_2d_mask = np.any(np.abs(LLE_data_2d - mean_2d) > threshold_2d * std_2d, axis=1)
outliers_2d_indices = np.where(outliers_2d_mask)[0]
print(f"Number of outliers in 2D LLE: {len(outliers_2d_indices)}")
outliers_3d_mask = np.any(np.abs(LLE_data_3d - mean_3d) > threshold_3d * std_3d, axis=1)
outliers_3d_indices = np.where(outliers_3d_mask)[0]
print(f"Number of outliers in 3D LLE: {len(outliers_3d_indices)}")
shared_outlier_indices = np.intersect1d(outliers_2d_indices, outliers_3d_indices)
print(f"Number of shared outliers: {len(shared_outlier_indices)}")
shared_outliers_data = MDS_clean_data.iloc[shared_outlier_indices]
shared_values = shared_outliers_data.nunique()
drop_candidates = shared_values[shared_values == 1].index
print(f"Columns with constant values in shared outliers: {list(drop_candidates)}")

# %%
# Visualizing outliers
plt.figure(figsize=(8, 6))
plt.scatter(LLE_data_2d[:, 0], LLE_data_2d[:, 1], c="lightgrey", s=15, label="Inliers")
plt.scatter(
    LLE_data_2d[outliers_2d_mask, 0],
    LLE_data_2d[outliers_2d_mask, 1],
    c="red",
    s=20,
    label="Outliers",
)
plt.title("Outliers in 2D LLE projection")
plt.legend()
plt.show()
plt.close()

# %%
# Removing outlier entries
threshold_2d_removal = 3.2
threshold_3d_removal = 3.2
outliers_2d_removal_mask = np.any(
    np.abs(LLE_data_2d - mean_2d) > threshold_2d_removal * std_2d, axis=1
)
outliers_2d_removal_count = np.sum(outliers_2d_removal_mask)
print(f"Number of stricter outliers in 2D LLE for removal: {outliers_2d_removal_count}")
outliers_3d_removal_mask = np.any(
    np.abs(LLE_data_3d - mean_3d) > threshold_3d_removal * std_3d, axis=1
)
outliers_3d_removal_count = np.sum(outliers_3d_removal_mask)
print(f"Number of stricter outliers in 3D LLE for removal: {outliers_3d_removal_count}")
combined_outlier_mask = outliers_2d_removal_mask | outliers_3d_removal_mask
total_outliers_to_remove = np.sum(combined_outlier_mask)
print(f"Total number of entries to remove: {total_outliers_to_remove}")
LLEed_data = MDS_clean_data[~combined_outlier_mask]
print(f"Original dataset size: {MDS_clean_data.shape}")
print(f"Filtered dataset size: {LLEed_data.shape}")

# %%
# Composite DFS call from helper module planned for next iterations
# final_features, clusters, feature_importances = helper.automated_feature_engineering_and_clustering(
#     df=LLEed_data,
#     entityset_name="MentalHealthSurvey",
#     corr_threshold=0.8,
#     min_cluster_size=10,
#     min_samples=5
# )
# %%
# DFS
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

# %%
# Checking for NaN and inf values in the dataset
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

# %%
# Converting infs to NaN
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

# %%
# Removing columns with NaN or inf values
data_dfs_var = data_with_dfs_onlyNaN.dropna(axis=1)
print(f"Original DFS_data shape: {data_with_dfs_onlyNaN.shape}")
print(f"Shape of data after removing NaN & inc columns: {data_dfs_var.shape}")

# %%
# Another iteration of variance filtering
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
data_dfs_varcor = helper.iterative_correlation_dropper(
    data_DFS_var, vs_threshold, variance2_df
)

# %%
# Iterative correlation-based filtering to exclude strongly correlated features
data_dfs_varcor = helper.iterative_correlation_dropper(
    data_dfs_varcor, s_threshold, variance2_df
)

# %%
# Verifying removal
print(f"Before dataset size: {data_DFS_var.shape[0], data_DFS_var.shape[1]}")
print(f"After dataset size: {data_dfs_varcor.shape[0], data_dfs_varcor.shape[1]}")
# filename = f'''./data_versions/survey_afterdfs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'''
# data_dfs_varcor.to_csv(filename, index=False)

# %%
# Another round of scaling
scaler2 = MinMaxScaler()
data_dfs_scaled = scaler2.fit_transform(data_dfs_varcor)
data_dfs_scaled = pd.DataFrame(data_dfs_scaled, columns=data_dfs_varcor.columns)
summary_dfsmms = data_dfs_scaled.describe(include="all").transpose()
summary_dfsmms = summary_dfsmms.drop(columns=["25%", "50%", "75%"])
summary_dfsmms.index.name = "Fnames"
print(tabulate.tabulate(summary_dfsmms, headers="keys", tablefmt="pretty"))

# %%
# Baseline k-means iter 01
kmeans_01 = KMeans(n_clusters=5, random_state=44)
labels_01 = kmeans_01.fit_predict(data_dfs_scaled)
centers_01 = kmeans_01.cluster_centers_
cluster_counts = pd.Series(labels_01).value_counts()
print(cluster_counts)
data_with_clusters = data_dfs_scaled.copy()
data_with_clusters["Cluster"] = labels_01

# %%
# Calculating radius of k-means clusters
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
importances = model.feature_importances_
feature_importances = pd.DataFrame(
    {"features": X.columns, "importance": importances}
).sort_values(by="importance", ascending=False)
# file_name = f'''./data_versions/survey_featimpiter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'''
# feature_importances.to_csv(file_name, index=False)
print(
    tabulate.tabulate(feature_importances.head(20), headers="keys", tablefmt="pretty")
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
QA_metrics_01 = helper.calculate_clustering_metrics(
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
# print(tabulate.tabulate(summary_df, headers='keys', tablefmt='pretty'))
summary_df["range"] = summary_df["max"] - summary_df["min"]
sorted_summary = summary_df.sort_values(by="range", ascending=False)
sorted_summary = sorted_summary.drop(columns=["std", "count"])
largest_ranges = sorted_summary.head(10)
print("Features with the 10 Largest Ranges:")
print(tabulate.tabulate(largest_ranges, headers="keys", tablefmt="pretty"))

# %%
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
print(tabulate.tabulate(summary_dfsfilt, headers="keys", tablefmt="pretty"))

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
# QA_file_name = f'''./data_versions/qa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'''
# QA_results.to_csv(QA_file_name, index=False)

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
    # plt.show()
plt.close()

# %%
# User can choose to query filtering per country they lived in
with get_session() as s:
    current_country = (
        s.scalars(
            select(UserCountry)
            .where(UserCountry.end_date.is_(None))
        )
        .first()
    )
    if current_country:
        tracks = s.scalars(
            select(Scrobble)
            .where(
                Scrobble.play_time >= current_country.start_date,
                # Any "end_date" check only if it's not NULL:
                current_country.end_date.is_(None)
                | (Scrobble.play_time < current_country.end_date)
            )
        ).all()
