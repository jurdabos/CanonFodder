# %%
from dotenv import load_dotenv
load_dotenv()
from DB import engine, SessionLocal
import helper
import itertools
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text
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

# %% Display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{: .2f}".format
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = helper.register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# %% Step 1: Loading the data from the latest parquet
print("===========================================")
print("Welcome to the CanonFodder canonization workflow!")
print("===========================================\n")
print("Step 1: Loading your scrobble data. Please wait...")
data, latest_filename = helper.load_latest_parquet_in_pq()
if data is None or data.empty:
    print("No DataFrame was loaded; cannot proceed with EDA.")
    sys.exit()
# Naming the columns
data.columns = ["Artist", "Album", "Song", "Datetime", "Country"]
data.info()
data.dropna(subset=["Datetime"], inplace=True)
before_count = len(data)
data = data.drop_duplicates(subset=["Artist", "Album", "Song", "Datetime"], keep="first")
after_count = len(data)
removed_count = before_count - after_count
logging.info(f"Removed {removed_count} duplicate rows.")

# %% Step 2: Connecting to artist unification data table
with SessionLocal() as sess:
    artist_link_rows = pd.read_sql(text("SELECT group_signature, canonical_name FROM groups_handled"), sess.bind)

# %% Step 3: Before-canonization top artists
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

# %% Step 4: Before-canonization artist stats
print(artist_counts.describe())
artist_counts_df = artist_counts.reset_index()
artist_counts_df.columns = ["Artist", "Count"]
count_threshold = 2
# Shrinking of the data set can be easily undone here by setting count_threshold to 1 in future flows
fltrd_artcount = artist_counts_df[artist_counts_df["Count"] >= count_threshold]
print(fltrd_artcount.describe())

# %% Step 5: Building a labelled training set directly from manually collated sqlite table
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
NEG_TARGET = 5000
# Build a fast lookup: group id  → list(variants)
groups = (artist_link_rows.assign(gid=range(len(artist_link_rows)))
          .explode('group_signature')
          .assign(variant=lambda d: d.group_signature.str.strip())
          .loc[lambda d: d.variant.ne('')]  # keep non‑empty
          .groupby('gid')['variant'].apply(list)
          )
rng  = np.random.default_rng(43)
gids = groups.index.to_numpy()
neg_pairs = [
    (0, *rng.choice(groups[g1]), *rng.choice(groups[g2]))[:3]   # (flag,A,B)
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
      f"neg= {len(df_pairs)-df_pairs.to_link.sum()})")

# %% Step 6: Teaching decision rule to logistic regression
score_df = (
    df_pairs
    .apply(lambda r: pd.Series(helper.fuzzy_scores(r["A"], r["B"])), axis=1)
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

# %% Step 7: I WILL WRITE SOMETHING INFORMATIVE HERE IF IT WORKS
artist_names = fltrd_artcount["Artist"].str.lower().str.strip().tolist()

match, prob = helper.most_similar(
        "bohren & der club of gore",
        artist_names,
        clf,
        threshold=cut,
)
print(match, prob)

# %% Step 8: Artist names similarity check with DBSCAN
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
dbscan = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
labels = dbscan.fit_predict(1 - similarity_matrix)
clusters = {}
for label, artist in zip(labels, fltrd_artcount["Artist"]):
    if label != -1:
        clusters.setdefault(label, []).append(artist)
similar_artist_groups = list(clusters.values())
print(f"Number of groups identified: {len(similar_artist_groups)}")
