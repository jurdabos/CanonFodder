# %% Basic setup
from dotenv import load_dotenv

load_dotenv()
from DB import SessionLocal
from DB.models import ArtistVariantsCanonized
from collections import Counter
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
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sqlalchemy import insert, select, text, update
import sys
from tabulate import tabulate
import warnings
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")
import woodwork as ww

if '__file__' in globals():
    HERE = Path(__file__).resolve().parent  # running from a file
else:
    HERE = Path.cwd()  # running in a console/notebook
JSON_DIR = HERE / "JSON"
PALETTES_FILE = JSON_DIR / "palettes.json"
LOGGER = logging.getLogger(__name__)
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
LOGGER.info("Fetching already canonised artist name variants…")
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
anchor_idx_sets = [s for s in anchor_idx_sets if len(s) >= 2]
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
# since, in the meantime, variants contained in the DBSCAN input set have been canonized.
best_eps = 0.23
labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
clusters = (
    pd.DataFrame({"Artist": artist_names, "label": labels})
    .query("label != -1")
    .groupby("label")["Artist"].apply(list)
    .tolist()
)

# %%
# ------------------------------------------------------------------
# 0.  Load gold-standard pairs
# ------------------------------------------------------------------
# This step previously identified a lot of relevant clusters, which I manually handled to form a gold-standard table.
# Cf. section 2.3.3 of the assignment
with SessionLocal() as sess:
    unhandled = 0
    for group in clusters:
        if len(group) <= 1:
            continue
        sig = cli.make_signature(group)  # <- unified
        row = sess.execute(
            text("SELECT 1 FROM artist_variants_canonized "
                 "WHERE artist_variants_text = :sig"),
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
print("Done unifying. ML steps coming.")

gs = pd.read_parquet("PQ/avc.parquet")
# Exploding into pairwise rows if not yet done
if {"variant_a", "variant_b"}.issubset(gs.columns) is False:
    rows = []
    for _, row in gs.iterrows():
        rows.extend(cluster.expand_pairs(row))
    gs = pd.DataFrame(rows, columns=[
        "variants", "variant_a", "variant_b", "to_link"
    ])
all_scores = {
    "ratio", "partial_ratio",
    "token_sort_ratio", "token_set_ratio",
    "WRatio", "QRatio"
}
if all_scores.difference(gs.columns):
    scores = gs.apply(
        lambda r: pd.Series(cluster.fuzzy_scores(r["variant_a"],
                                                 r["variant_b"])),
        axis=1
    )
    # Overwriting only the missing columns
    for col in all_scores:
        if col not in gs.columns:
            gs[col] = scores[col]

# %%
# ------------------------------------------------------------------
# 1.  New engineered features
# ------------------------------------------------------------------
gs = pd.concat([gs, gs["variants"].apply(stats.length_stats)], axis=1)

# %%
# ------------------------------------------------------------------
# 2.  Feature / target split
# ------------------------------------------------------------------
target = "to_link"
num_cols = [c for c in gs.columns if c not in ["variants", target,
                                               "variant_a", "variant_b"]]
X = gs[num_cols]
y = gs[target].astype(int)

# ------------------------------------------------------------------
# 3.  Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=47, stratify=y
)

# %%
# ------------------------------------------------------------------
# 4.  Pipeline: RobustScaler ➜ XGBoost
# ------------------------------------------------------------------
numeric_pipe = Pipeline([
    ("scaler", RobustScaler())
])
pre = ColumnTransformer(
    [("num", numeric_pipe, num_cols)],
    remainder="drop"
)
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.75,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='logloss',
    random_state=49,
    n_jobs=-1
)
model = Pipeline([
    ("prep", pre),
    ("xgb", xgb)
])

# %%
# ------------------------------------------------------------------
# 5.  Training
# ------------------------------------------------------------------
model.fit(X_train, y_train)

# %%
# ------------------------------------------------------------------
# 6.  Evaluation
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n=== XGBoost report (held-out 25 %) ===")
print(classification_report(y_test, y_pred,
                            target_names=["no link", "link"]))
print("AUC :", round(roc_auc_score(y_test, y_prob), 3))

# %%
# ------------------------------------------------------------------
# 7.  Feature importance (gain)
# ------------------------------------------------------------------
booster = model.named_steps["xgb"]
raw_imp = booster.get_booster().get_score(importance_type="gain")
feat_names = model.named_steps["prep"].get_feature_names_out()
imp_series = (pd.Series(raw_imp)
              .rename(index=lambda k: feat_names[int(k[1:])])
              .astype(float)
              .sort_values(ascending=False))
TOP_SHOW = 15
imp_df = (imp_series.head(TOP_SHOW)
          .reset_index()
          .rename(columns={"index": "feature", 0: "gain"}))
print("\nTop-gain features")
print(tabulate(imp_df, headers="keys", tablefmt="pretty", showindex=False))

# %%
# ------------------------------------------------------------------
# 8.  Save model & mapping
# ------------------------------------------------------------------
Path("models").mkdir(exist_ok=True)
model_path = Path("models/xgb.json")
booster.save_model(model_path)
(Path("models/xgb_columns.json")
 .write_text(json.dumps(num_cols, indent=2)))
print(f"\n✅  Model saved to {model_path}")

# %%
# ---------------------------------------------------------------
# 9.  Mis-classified rows
# ---------------------------------------------------------------
stats.show_misclassified(
    gs_df=gs,
    model_pipe=model,
    X_matrix=gs[num_cols],
    extra_cols=list(all_scores) + ["sig_len", "avg_name_len"],
    top_n=40
)

# TEST set only
stats.show_misclassified(gs, model, gs[num_cols],
                         only_test=True, idx_test=idx_test,
                         extra_cols=list(all_scores), top_n=15)

# %%

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
