"""
Runs Experiment 12: album/track-enriched model catalogue.

Augments the Exp 11 feature set with discography overlap features derived
from MBDB release_groups (≈albums) and recordings (≈tracks).  For training
pairs the discography comes from MBDB; for the AVC test set it comes from
scrobble.parquet.  Positive training pairs use subsampled discographies
(30–70 % per side) to bridge the MBDB→scrobble domain gap.

Compares XGBoost, RF, ExtraTrees, LightGBM, GradientBoosting against
Exp 6 ExtraTrees (AUC=0.892, F1=0.705) and Exp 11 RF (AUC=0.884, F1=0.643).
"""
from __future__ import annotations
import itertools
import logging
import random
import sys
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from corefunc.canon.experiment_runner import _build_model_catalogue
from corefunc.mb_local import _psql_csv, check_local_mb
from helpers.io import AVC_PQ, PQ_DIR, read_parquet, dump_parquet, sanitize, SCROBBLE_PQ
from helpers.features import compute_pair_features
from helpers import cluster, stats
from helpers.device import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_MB_MAX_PQ = PQ_DIR / "gs_mb_max.parquet"
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
DISCOGRAPHY_PQ = PQ_DIR / "mbdb_discography.parquet"
FUZZY_TRACK_THR = 80
SUBSAMPLE_LO = 0.30
SUBSAMPLE_HI = 0.70
MAX_TRACKS = 40
MAX_ALBUMS = 20

_SIM_SCORES = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio", "QRatio"]
# Selecting only 5 tree-based models (composites slow, didn't help in Exp 11)
_MODELS_TO_RUN = {"XGBoost", "RandomForest", "ExtraTrees", "LightGBM", "GradientBoosting"}


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: MBDB name→MBID mapping + discography extraction
# ═════════════════════════════════════════════════════════════════════════════
def _build_name_to_mbid() -> dict[str, str]:
    """Queries MBDB for all (name, mbid) pairs and returns name→first_mbid dict."""
    log.info("Building name→MBID mapping from MBDB...")
    sql = """\
SELECT a.gid::text AS mbid, a.name AS name
FROM musicbrainz.artist a
UNION ALL
SELECT a.gid::text AS mbid, aa.name AS name
FROM musicbrainz.artist a
JOIN musicbrainz.artist_alias aa ON aa.artist = a.id"""
    raw = _psql_csv(sql)
    raw = raw.dropna(subset=["name", "mbid"])
    raw["name"] = raw["name"].astype(str)
    # Keeping first MBID per name (deterministic)
    name_to_mbid: dict[str, str] = {}
    for _, row in raw.iterrows():
        name = row["name"]
        if name not in name_to_mbid:
            name_to_mbid[name] = row["mbid"]
    log.info("Name→MBID mapping: %d unique names.", len(name_to_mbid))
    return name_to_mbid


def step1_extract_discographies() -> pd.DataFrame:
    """Extracts per-MBID discographies from MBDB and caches as parquet.

    Returns DataFrame with columns: mbid, albums_str, tracks_str
    ({-delimited strings of unique release_group and recording names).
    """
    if DISCOGRAPHY_PQ.exists():
        existing = read_parquet(DISCOGRAPHY_PQ)
        log.info("mbdb_discography.parquet already cached: %d MBIDs.", len(existing))
        return existing
    if not check_local_mb():
        raise RuntimeError("Local MBDB mirror not reachable.")
    log.info("Extracting discographies from MBDB (this will take a few minutes)...")
    # Getting all artist MBIDs that have aliases (same universe as gs_mb_max)
    mbid_df = _psql_csv("""\
SELECT DISTINCT a.gid::text AS mbid
FROM musicbrainz.artist a
WHERE EXISTS (SELECT 1 FROM musicbrainz.artist_alias aa WHERE aa.artist = a.id)""")
    all_mbids = mbid_df["mbid"].tolist()
    log.info("Artists with aliases: %d MBIDs to extract.", len(all_mbids))
    # Batching discography extraction
    batch_size = 500
    results = []
    for i in range(0, len(all_mbids), batch_size):
        batch = all_mbids[i:i + batch_size]
        values = ",".join(f"'{m}'" for m in batch)
        sql = f"""\
SELECT
    a.gid::text AS mbid,
    STRING_AGG(DISTINCT rg.name, '{{' ORDER BY rg.name) AS albums_str,
    STRING_AGG(DISTINCT rec.name, '{{' ORDER BY rec.name) AS tracks_str
FROM musicbrainz.artist a
JOIN musicbrainz.artist_credit_name acn ON acn.artist = a.id
LEFT JOIN musicbrainz.release_group rg ON rg.artist_credit = acn.artist_credit
LEFT JOIN musicbrainz.release r ON r.release_group = rg.id
LEFT JOIN musicbrainz.medium m ON m.release = r.id
LEFT JOIN musicbrainz.track t ON t.medium = m.id
LEFT JOIN musicbrainz.recording rec ON rec.id = t.recording
WHERE a.gid IN ({values})
GROUP BY a.gid"""
        batch_df = _psql_csv(sql)
        if not batch_df.empty:
            results.append(batch_df)
        if (i // batch_size) % 50 == 0:
            log.info("  Discography extraction: %d/%d MBIDs...", min(i + batch_size, len(all_mbids)), len(all_mbids))
    disco_df = pd.concat(results, ignore_index=True)
    disco_df["albums_str"] = disco_df["albums_str"].fillna("")
    disco_df["tracks_str"] = disco_df["tracks_str"].fillna("")
    log.info("Extracted discographies for %d MBIDs.", len(disco_df))
    dump_parquet(disco_df, DISCOGRAPHY_PQ)
    return disco_df


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Data assembly (reuses Exp 10/11 logic)
# ═════════════════════════════════════════════════════════════════════════════
def _compute_wratio_bulk(df: pd.DataFrame, label: str = "") -> pd.Series:
    """Computes WRatio for every row with progress logging."""
    n = len(df)
    log.info("Computing WRatio for %d %s pairs...", n, label)
    wr = df.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    log.info("  Done. WRatio range: [%.0f, %.0f]", wr.min(), wr.max())
    return wr


def assemble_training() -> pd.DataFrame:
    """Assembles the gs_mb_max-based training set with distribution-matched negatives."""
    positives_all = read_parquet(GS_MB_MAX_PQ)
    log.info("gs_mb_max.parquet: %d pairs.", len(positives_all))
    pos_wr = _compute_wratio_bulk(positives_all, "positive")
    mask = (pos_wr >= WRATIO_LOWER) & (pos_wr < WRATIO_UPPER)
    positives = positives_all[mask].reset_index(drop=True)
    pos_wr_filtered = pos_wr[mask].reset_index(drop=True)
    log.info("Positives in [%d,%d): %d.", WRATIO_LOWER, WRATIO_UPPER, len(positives))
    # Loading DBSCAN negatives
    dbscan = read_parquet(GS_DBSCAN_PQ)
    neg_pool = dbscan[dbscan["to_link"] == False].reset_index(drop=True)
    neg_wr = _compute_wratio_bulk(neg_pool, "negative")
    neg_pool = neg_pool.copy()
    neg_pool["_wr"] = neg_wr
    # Distribution matching
    n_bins = 8
    bin_edges = np.linspace(60, 100, n_bins + 1)
    pos_hist, _ = np.histogram(pos_wr_filtered, bins=bin_edges)
    pos_fracs = pos_hist / pos_hist.sum()
    n_target = min(len(positives), len(neg_pool))
    log.info("Target negatives: %d.", n_target)
    neg_pool["_bin"] = pd.cut(neg_pool["_wr"], bins=bin_edges, right=False, labels=False)
    neg_pool = neg_pool.dropna(subset=["_bin"])
    neg_pool["_bin"] = neg_pool["_bin"].astype(int)
    targets = (pos_fracs * n_target).astype(int)
    targets[np.argmax(pos_fracs)] += n_target - targets.sum()
    sampled_parts = []
    shortfall = 0
    available_bins = []
    for i in range(n_bins):
        bin_df = neg_pool[neg_pool["_bin"] == i]
        if len(bin_df) < targets[i]:
            shortfall += targets[i] - len(bin_df)
            sampled_parts.append(bin_df)
        else:
            available_bins.append((i, bin_df, targets[i]))
    if shortfall > 0 and available_bins:
        total_surplus = sum(len(bdf) - t for _, bdf, t in available_bins)
        for i, bin_df, base_target in available_bins:
            surplus = len(bin_df) - base_target
            extra = int(round(shortfall * surplus / max(total_surplus, 1)))
            final_n = min(base_target + extra, len(bin_df))
            sampled_parts.append(bin_df.sample(n=final_n, random_state=RANDOM_STATE))
    else:
        for i, bin_df, target in available_bins:
            sampled_parts.append(bin_df.sample(n=target, random_state=RANDOM_STATE))
    neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_wr", "_bin"])
    log.info("Distribution-matched negatives: %d.", len(neg_sampled))
    train = pd.concat([
        positives[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
        neg_sampled[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
    ], ignore_index=True)
    log.info("Training set: %d pairs (pos=%d, neg=%d).",
             len(train), train["to_link"].sum(), (~train["to_link"]).sum())
    return train


# ═════════════════════════════════════════════════════════════════════════════
# Discography feature computation
# ═════════════════════════════════════════════════════════════════════════════
def _parse_delimited(s: str) -> list[str]:
    """Splits a {-delimited string into a list, filtering blanks."""
    if not s or pd.isna(s):
        return []
    return [x for x in s.split("{") if x.strip()]


def _jaccard(a: set, b: set) -> float:
    """Returns Jaccard index, 1.0 when both are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _fuzzy_overlap(list_a: list[str], list_b: list[str], threshold: int = FUZZY_TRACK_THR) -> tuple[int, float]:
    """Counts items in list_a that fuzzy-match any item in list_b.

    Uses rapidfuzz.process.extractOne (C-optimised) instead of nested loops.
    Returns (n_matched, ratio).
    """
    if not list_a or not list_b:
        return 0, 0.0
    matched = 0
    for a in list_a:
        result = process.extractOne(a, list_b, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
        if result is not None:
            matched += 1
    total = max(len(set(list_a) | set(list_b)), 1)
    return matched, matched / total


def compute_discography_features(
    albums_a: list[str], albums_b: list[str],
    tracks_a: list[str], tracks_b: list[str],
) -> dict[str, float]:
    """Computes 10 discography overlap features for a single pair."""
    # Capping list sizes to avoid outlier artists with thousands of items
    tracks_a = tracks_a[:MAX_TRACKS]
    tracks_b = tracks_b[:MAX_TRACKS]
    albums_a = albums_a[:MAX_ALBUMS]
    albums_b = albums_b[:MAX_ALBUMS]
    # Track features (primary signal)
    set_ta, set_tb = set(tracks_a), set(tracks_b)
    n_fuzzy_trk, fuzzy_trk_ratio = _fuzzy_overlap(tracks_a, tracks_b)
    # Album features (included but may be weak/inverted)
    set_aa, set_ab = set(albums_a), set(albums_b)
    n_fuzzy_alb, fuzzy_alb_ratio = _fuzzy_overlap(albums_a, albums_b)
    return {
        "fuzzy_track_ratio": fuzzy_trk_ratio,
        "has_fuzzy_track_match": float(n_fuzzy_trk > 0),
        "n_fuzzy_track_matches": n_fuzzy_trk,
        "exact_track_jaccard": _jaccard(set_ta, set_tb),
        "fuzzy_album_ratio": fuzzy_alb_ratio,
        "has_fuzzy_album_match": float(n_fuzzy_alb > 0),
        "min_track_count": min(len(set_ta), len(set_tb)),
        "max_track_count": max(len(set_ta), len(set_tb)),
        "min_album_count": min(len(set_aa), len(set_ab)),
        "max_album_count": max(len(set_aa), len(set_ab)),
    }


def _subsample(items: list[str], rng: random.Random) -> list[str]:
    """Subsamples a list to 30–70 % of its length (minimum 1 item)."""
    if len(items) <= 1:
        return items
    frac = rng.uniform(SUBSAMPLE_LO, SUBSAMPLE_HI)
    k = max(1, int(len(items) * frac))
    return rng.sample(items, k)


def add_discography_features_train(
    df: pd.DataFrame,
    name_to_mbid: dict[str, str],
    mbid_to_disco: dict[str, tuple[list[str], list[str]]],
) -> pd.DataFrame:
    """Adds discography features to training DataFrame.

    For positive pairs (same MBID), subsamples each side's discography
    to simulate partial scrobble coverage.
    """
    n = len(df)
    log.info("Computing discography features for %d training pairs...", n)
    rng = random.Random(RANDOM_STATE)
    disco_rows = []
    n_miss = 0
    for i, (_, row) in enumerate(df.iterrows()):
        va, vb = str(row["variant_a"]), str(row["variant_b"])
        is_pos = bool(row["to_link"])
        mbid_a = name_to_mbid.get(va)
        mbid_b = name_to_mbid.get(vb)
        if mbid_a and mbid_a in mbid_to_disco:
            albums_a, tracks_a = mbid_to_disco[mbid_a]
        else:
            albums_a, tracks_a = [], []
        if mbid_b and mbid_b in mbid_to_disco:
            albums_b, tracks_b = mbid_to_disco[mbid_b]
        else:
            albums_b, tracks_b = [], []
        if not albums_a and not tracks_a and not albums_b and not tracks_b:
            n_miss += 1
        # Subsampling positive pairs to avoid trivial 1.0 overlap
        if is_pos and (tracks_a or albums_a):
            albums_a = _subsample(albums_a, rng)
            albums_b = _subsample(albums_b, rng)
            tracks_a = _subsample(tracks_a, rng)
            tracks_b = _subsample(tracks_b, rng)
        disco_rows.append(compute_discography_features(albums_a, albums_b, tracks_a, tracks_b))
        if (i + 1) % 50000 == 0:
            log.info("  Discography features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    if n_miss > 0:
        log.warning("  %d pairs had no discography data on either side.", n_miss)
    disco_df = pd.DataFrame(disco_rows, index=df.index)
    for col in disco_df.columns:
        df[col] = disco_df[col]
    log.info("  Discography features added: %d columns.", len(disco_df.columns))
    return df


def add_discography_features_test(df: pd.DataFrame) -> pd.DataFrame:
    """Adds discography features to test DataFrame using scrobble data."""
    log.info("Building scrobble discography lookup...")
    scrobbles = read_parquet(SCROBBLE_PQ)
    # Filtering blank album titles
    clean = scrobbles[scrobbles["album_title"].notna() & (scrobbles["album_title"].str.strip() != "")]
    name_to_albums = clean.groupby("artist_name")["album_title"].apply(
        lambda x: sorted(set(x.unique()))
    ).to_dict()
    name_to_tracks = scrobbles.groupby("artist_name")["track_title"].apply(
        lambda x: sorted({t for t in x.dropna().unique() if t.strip()})
    ).to_dict()
    log.info("Scrobble discography: %d names with albums, %d with tracks.",
             len(name_to_albums), len(name_to_tracks))
    # Computing features
    n = len(df)
    log.info("Computing discography features for %d test pairs...", n)
    disco_rows = []
    for _, row in df.iterrows():
        va, vb = str(row["variant_a"]), str(row["variant_b"])
        albums_a = name_to_albums.get(va, [])
        albums_b = name_to_albums.get(vb, [])
        tracks_a = name_to_tracks.get(va, [])
        tracks_b = name_to_tracks.get(vb, [])
        disco_rows.append(compute_discography_features(albums_a, albums_b, tracks_a, tracks_b))
    disco_df = pd.DataFrame(disco_rows, index=df.index)
    for col in disco_df.columns:
        df[col] = disco_df[col]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Base + interaction + length features (from Exp 11)
# ═════════════════════════════════════════════════════════════════════════════
def compute_interaction_features_vectorized(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise diffs and products among 6 similarity scores (vectorized)."""
    seen: Counter = Counter()
    interaction_cols: dict[str, np.ndarray] = {}
    score_names = [s for s in _SIM_SCORES if s in feat_df.columns]
    for i, j in itertools.combinations(range(len(score_names)), 2):
        a_name, b_name = score_names[i], score_names[j]
        diff_col = sanitize(f"{a_name} - {b_name}", seen)
        interaction_cols[diff_col] = feat_df[a_name].values - feat_df[b_name].values
        prod_col = sanitize(f"{a_name} * {b_name}", seen)
        interaction_cols[prod_col] = feat_df[a_name].values * feat_df[b_name].values
    return pd.DataFrame(interaction_cols, index=feat_df.index)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base (23) + interaction (30) + length (5) features."""
    n = len(df)
    log.info("Computing base features for %d pairs...", n)
    feat_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = str(row["variant_a"]), str(row["variant_b"])
        feats = compute_pair_features(a, b)
        feat_rows.append(feats)
        if (i + 1) % 50000 == 0:
            log.info("  Base features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    feat_df = pd.DataFrame(feat_rows, index=df.index)
    for col in feat_df.columns:
        if col not in df.columns:
            df[col] = feat_df[col]
    # Interaction features (vectorized)
    log.info("Computing interaction features...")
    interaction_df = compute_interaction_features_vectorized(feat_df)
    for col in interaction_df.columns:
        if col not in df.columns:
            df[col] = interaction_df[col]
    # Length stats
    if "variants" not in df.columns:
        df["variants"] = df["variant_a"].astype(str) + "{" + df["variant_b"].astype(str)
    df = pd.concat([df, df["variants"].apply(stats.length_stats)], axis=1)
    return df


def prune_features(X: pd.DataFrame) -> list[str]:
    """Applies variance and correlation pruning."""
    var_df, selected = stats.variance_testing(X, 0.001)
    X_pruned = X[selected]
    log.info("After variance pruning: %d → %d features.", len(X.columns), len(X_pruned.columns))
    X_pruned = stats.iterative_correlation_dropper(X_pruned, 0.95, var_df, 20)
    log.info("After correlation pruning: → %d features.", len(X_pruned.columns))
    return list(X_pruned.columns)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════
def _optimal_threshold(y_true, y_prob):
    """Finds the threshold that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1s)]), float(f1s[np.argmax(f1s)])


def _eval_at(y_true, y_prob, thr):
    """Computes metrics at a given threshold."""
    y_pred = (y_prob >= thr).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": thr,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═════════════════════════════════════════════════════════════════════════════
def run_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str]):
    """Trains 5 tree-based models and evaluates with threshold sweep."""
    target = "to_link"
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info("Train: %d | Test: %d | Features: %d | spw: %.2f",
             len(X_train), len(X_test), len(num_cols), spw)
    catalogue = _build_model_catalogue(spw, device, random_state=RANDOM_STATE)
    # Filtering to tree-based models only
    catalogue = {k: v for k, v in catalogue.items() if k in _MODELS_TO_RUN}
    results = []
    for model_name, clf in catalogue.items():
        log.info("─── Training %s ───", model_name)
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop", verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        default_m = _eval_at(y_test, y_prob, 0.5)
        opt_thr, _ = _optimal_threshold(y_test, y_prob)
        optimal_m = _eval_at(y_test, y_prob, opt_thr)
        best_hi = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for t in np.arange(0.50, 0.99, 0.01):
            m = _eval_at(y_test, y_prob, t)
            if m["precision"] >= 0.80 and m["f1"] > best_hi["f1"]:
                best_hi = m
        results.append({
            "model": model_name, "auc": auc,
            "default_f1": default_m["f1"], "default_prec": default_m["precision"], "default_rec": default_m["recall"],
            "opt_thr": optimal_m["threshold"], "opt_f1": optimal_m["f1"],
            "opt_prec": optimal_m["precision"], "opt_rec": optimal_m["recall"],
            "hiprec_thr": best_hi["threshold"], "hiprec_f1": best_hi["f1"],
            "hiprec_prec": best_hi["precision"], "hiprec_rec": best_hi["recall"],
        })
        log.info("%s → AUC=%.4f | def F1=%.4f | opt F1=%.4f (thr=%.3f) | hi-P F1=%.4f (thr=%.3f)",
                 model_name, auc, default_m["f1"], optimal_m["f1"], opt_thr,
                 best_hi["f1"], best_hi["threshold"])
        y_pred_opt = (y_prob >= opt_thr).astype(int)
        print(f"\n=== {model_name} (optimal thr={opt_thr:.3f}) ===")
        print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
        # Feature importance
        clf_fitted = pipeline.named_steps["clf"]
        if hasattr(clf_fitted, "feature_importances_"):
            imps = sorted(zip(num_cols, clf_fitted.feature_importances_), key=lambda x: x[1], reverse=True)
            print(f"Top 15 features ({model_name}):")
            for name, imp in imps[:15]:
                print(f"  {name:<35} {imp:.4f}")
    # Summary table
    print("\n" + "=" * 120)
    print(f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
          f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
          f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x["opt_f1"], reverse=True):
        print(f"{r['model']:<22} {r['auc']:>6.4f} | "
              f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} {r['default_f1']:>6.4f} | "
              f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} {r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
              f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} {r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}")
    print("=" * 120)
    print("\n── Baselines ──")
    print("Exp 6  ExtraTrees (gs_mb_dbscan, 2158:1):    AUC=0.8920, opt F1=0.7050 (P=0.750, R=0.667, thr=0.940)")
    print("Exp 11 RandomForest (gs_mb_max, no disco):    AUC=0.8838, opt F1=0.6429 (P=0.551, R=0.771, thr=0.995)")
    best = max(results, key=lambda x: x["auc"])
    print(f"\nBest model by AUC: {best['model']} (AUC={best['auc']:.4f})")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 12 pipeline."""
    log.info("=== Experiment 12: Album/Track-Enriched Model Catalogue ===")
    # ── Step 1: MBDB discography ──────────────────────────────────────────
    disco_df = step1_extract_discographies()
    mbid_to_disco: dict[str, tuple[list[str], list[str]]] = {}
    for _, row in disco_df.iterrows():
        mbid_to_disco[row["mbid"]] = (
            _parse_delimited(row["albums_str"]),
            _parse_delimited(row["tracks_str"]),
        )
    log.info("Discography lookup: %d MBIDs.", len(mbid_to_disco))
    # Building name→MBID mapping
    name_to_mbid = _build_name_to_mbid()
    # ── Step 2: Data assembly ─────────────────────────────────────────────
    train_full = assemble_training()
    # ── AVC test set ──────────────────────────────────────────────────────
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1,
    )
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)]
    test_df = test_raw.drop(columns=["_wr"]).reset_index(drop=True)
    log.info("AVC test: %d pairs (pos=%d, neg=%d).",
             len(test_df), test_df["to_link"].sum(), (~test_df["to_link"]).sum())
    # ── Step 3: Features ──────────────────────────────────────────────────
    log.info("Computing base features for training set...")
    train_df = add_base_features(train_full)
    log.info("Adding discography features for training set...")
    train_df = add_discography_features_train(train_df, name_to_mbid, mbid_to_disco)
    log.info("Computing base features for test set...")
    test_df = add_base_features(test_df)
    log.info("Adding discography features for test set...")
    test_df = add_discography_features_test(test_df)
    # ── Step 4: Pruning ───────────────────────────────────────────────────
    target = "to_link"
    exclude = {"variants", target, "variant_a", "variant_b", "source", "_key"}
    all_num = [c for c in train_df.columns
               if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_features(train_df[all_num])
    # Ensuring test_df has the same columns
    missing = [c for c in num_cols if c not in test_df.columns]
    if missing:
        log.warning("Columns missing in test_df (filling with 0): %s", missing)
        for c in missing:
            test_df[c] = 0.0
    # ── Step 5: Run experiment ────────────────────────────────────────────
    log.info("Running Experiment 12...")
    results = run_experiment(train_df, test_df, num_cols)
    return results


if __name__ == "__main__":
    main()
