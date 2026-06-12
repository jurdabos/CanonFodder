"""
Runs Experiment 14: unified catalogue lookup with flattened pairs.

Key changes from Exp 13:
1. Pair-level flattening — AVC groups are expanded to pairs *before* splitting,
   then split at the pair level (80/20 stratified) instead of group level.
2. No group-level features — length_stats() is dropped entirely; pair-level
   length info already lives in abs_len_diff and length_ratio (base 23).
3. Unified MBID→MBDB + scrobble fallback catalogue — for each artist name,
   tries MBID lookup in scrobble.parquet first; if found, pulls the full
   solo-credit discography from the local MBDB mirror; otherwise falls back
   to scrobble-side tracks/albums.
4. Solo-credit MBDB query — filters artist_credit.artist_count = 1 to
   exclude collaborative credits that inflated Exp 12 discographies.

Compares XGBoost, RF, ExtraTrees, LightGBM, GradientBoosting against
Exp 6 ExtraTrees (AUC=0.892, F1=0.705) and
Exp 13 XGBoost (AUC=0.9968, opt F1=0.9524).
"""

from __future__ import annotations

import itertools
import logging
import sys
import tempfile
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
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from corefunc.canon.experiment_runner import _build_model_catalogue, _safe_get_params  # noqa: E402
from corefunc.mb_local import _psql_csv, check_local_mb  # noqa: E402
from helpers import cluster, experiment, stats  # noqa: E402
from helpers.device import get_device  # noqa: E402
from helpers.features import compute_pair_features  # noqa: E402
from helpers.io import (  # noqa: E402
    AVC_PQ,  # noqa: E402
    PQ_DIR,
    SCROBBLE_PQ,
    dump_parquet,
    read_parquet,  # noqa: E402
    sanitize,
)  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
TEST_SIZE = 0.20
WRATIO_LOWER = 60
WRATIO_UPPER = 100
FUZZY_TRACK_THR = 80
FUZZY_ALBUM_THR = 80
MAX_TRACKS = 10000
MAX_ALBUMS = 1000
SOLO_DISCO_PQ = PQ_DIR / "mbdb_discography_solo.parquet"
_MBID_BATCH = 500

_SIM_SCORES = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio", "WRatio", "QRatio"]
_MODELS_TO_RUN = {"XGBoost", "RandomForest", "ExtraTrees", "LightGBM", "GradientBoosting"}


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Pair-level flattening and split
# ═════════════════════════════════════════════════════════════════════════════
def step1_flatten_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expands all AVC groups to pairs, then splits at the pair level.

    Returns (train_pairs_df, test_pairs_df) with columns:
    [variant_a, variant_b, to_link].
    """
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].reset_index(drop=True)
    log.info(
        "AVC decided rows: %d (pos=%d, neg=%d).", len(decided), decided["to_link"].sum(), (~decided["to_link"]).sum()
    )
    # Expanding every group to pairwise combinations
    all_rows: list[tuple] = []
    for _, row in decided.iterrows():
        all_rows.extend(cluster.expand_pairs(row))
    all_pairs = pd.DataFrame(all_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    log.info(
        "Expanded to %d pairs (pos=%d, neg=%d).",
        len(all_pairs),
        all_pairs["to_link"].sum(),
        (~all_pairs["to_link"]).sum(),
    )
    # Deduplicating pairs (same (a,b) could appear from overlapping groups)
    all_pairs = all_pairs.drop_duplicates(subset=["variant_a", "variant_b"]).reset_index(drop=True)
    log.info("After dedup: %d pairs.", len(all_pairs))
    # Filtering to WRatio [60, 100)
    all_pairs["_wr"] = all_pairs.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    all_pairs = (
        all_pairs[(all_pairs["_wr"] >= WRATIO_LOWER) & (all_pairs["_wr"] < WRATIO_UPPER)]
        .drop(columns=["_wr", "variants"])
        .reset_index(drop=True)
    )
    log.info(
        "After WRatio [%d,%d) filter: %d pairs (pos=%d, neg=%d).",
        WRATIO_LOWER,
        WRATIO_UPPER,
        len(all_pairs),
        all_pairs["to_link"].sum(),
        (~all_pairs["to_link"]).sum(),
    )
    # Splitting at the pair level with stratification
    train_pairs, test_pairs = train_test_split(
        all_pairs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_pairs["to_link"],
    )
    train_pairs = train_pairs.reset_index(drop=True)
    test_pairs = test_pairs.reset_index(drop=True)
    log.info(
        "Train: %d (pos=%d, neg=%d) | Test: %d (pos=%d, neg=%d)",
        len(train_pairs),
        train_pairs["to_link"].sum(),
        (~train_pairs["to_link"]).sum(),
        len(test_pairs),
        test_pairs["to_link"].sum(),
        (~test_pairs["to_link"]).sum(),
    )
    return train_pairs, test_pairs


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Solo-credit MBDB discography extraction
# ═════════════════════════════════════════════════════════════════════════════
def step2_extract_solo_discographies() -> pd.DataFrame:
    """Extracts per-MBID discographies from MBDB using solo credits only.

    Filters artist_credit.artist_count = 1 to exclude collaborative credits.
    Caches as mbdb_discography_solo.parquet.
    Returns DataFrame with columns: mbid, albums_str, tracks_str.
    """
    if SOLO_DISCO_PQ.exists():
        existing = read_parquet(SOLO_DISCO_PQ)
        log.info("mbdb_discography_solo.parquet cached: %d MBIDs.", len(existing))
        return existing
    if not check_local_mb():
        raise RuntimeError("Local MBDB mirror not reachable.")
    log.info("Extracting solo-credit discographies from MBDB...")
    # Getting all artist MBIDs that have aliases
    mbid_df = _psql_csv("""\
SELECT DISTINCT a.gid::text AS mbid
FROM musicbrainz.artist a
WHERE EXISTS (SELECT 1 FROM musicbrainz.artist_alias aa WHERE aa.artist = a.id)""")
    all_mbids = mbid_df["mbid"].tolist()
    log.info("Artists with aliases: %d MBIDs to extract.", len(all_mbids))
    # Batching discography extraction with solo-credit filter
    results: list[pd.DataFrame] = []
    for i in range(0, len(all_mbids), _MBID_BATCH):
        batch = all_mbids[i : i + _MBID_BATCH]
        values = ",".join(f"'{m}'" for m in batch)
        sql = f"""\
SELECT
    a.gid::text AS mbid,
    STRING_AGG(DISTINCT rg.name, '{{' ORDER BY rg.name) AS albums_str,
    STRING_AGG(DISTINCT rec.name, '{{' ORDER BY rec.name) AS tracks_str
FROM musicbrainz.artist a
JOIN musicbrainz.artist_credit_name acn ON acn.artist = a.id
JOIN musicbrainz.artist_credit ac ON acn.artist_credit = ac.id
LEFT JOIN musicbrainz.release_group rg ON rg.artist_credit = ac.id
LEFT JOIN musicbrainz.release r ON r.release_group = rg.id
LEFT JOIN musicbrainz.medium m ON m.release = r.id
LEFT JOIN musicbrainz.track t ON t.medium = m.id
LEFT JOIN musicbrainz.recording rec ON rec.id = t.recording
WHERE a.gid IN ({values})
  AND ac.artist_count = 1
GROUP BY a.gid"""
        batch_df = _psql_csv(sql)
        if not batch_df.empty:
            results.append(batch_df)
        if (i // _MBID_BATCH) % 50 == 0:
            log.info("  Solo disco extraction: %d/%d MBIDs...", min(i + _MBID_BATCH, len(all_mbids)), len(all_mbids))
    disco_df = pd.concat(results, ignore_index=True)
    disco_df["albums_str"] = disco_df["albums_str"].fillna("")
    disco_df["tracks_str"] = disco_df["tracks_str"].fillna("")
    log.info("Extracted solo-credit discographies for %d MBIDs.", len(disco_df))
    dump_parquet(disco_df, SOLO_DISCO_PQ)
    return disco_df


def _parse_delimited(s: str) -> list[str]:
    """Splits a {-delimited string into a list, filtering blanks."""
    if not s or pd.isna(s):
        return []
    return [x for x in s.split("{") if x.strip()]


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Unified catalogue lookup (MBID→MBDB first, scrobble fallback)
# ═════════════════════════════════════════════════════════════════════════════
def build_unified_lookups(
    mbid_to_disco: dict[str, tuple[list[str], list[str]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Builds name→albums and name→tracks dicts using unified MBID-first lookup.

    For each unique artist name in scrobble.parquet:
      1. If the name has a valid MBID and that MBID is in mbid_to_disco →
         use MBDB solo-credit catalogue.
      2. Otherwise → use scrobble-side tracks/albums.
    Returns (name_to_albums, name_to_tracks).
    """
    log.info("Building unified catalogue lookups...")
    scrobbles = read_parquet(SCROBBLE_PQ)
    # Building name→MBID mapping from scrobble data
    has_mbid = scrobbles["artist_mbid"].notna() & (scrobbles["artist_mbid"].str.len() == 36)
    name_to_mbid: dict[str, str] = (
        scrobbles[has_mbid].drop_duplicates("artist_name").set_index("artist_name")["artist_mbid"].to_dict()
    )
    log.info("  Name→MBID from scrobbles: %d names.", len(name_to_mbid))
    # Building scrobble-side fallback lookups
    clean = scrobbles[scrobbles["album_title"].notna() & (scrobbles["album_title"].str.strip() != "")]
    scrobble_albums = clean.groupby("artist_name")["album_title"].apply(lambda x: sorted(set(x.unique()))).to_dict()
    scrobble_tracks = (
        scrobbles.groupby("artist_name")["track_title"]
        .apply(lambda x: sorted({t for t in x.dropna().unique() if t.strip()}))
        .to_dict()
    )
    # Merging: MBDB-first, scrobble fallback
    all_names = set(scrobbles["artist_name"].unique())
    name_to_albums: dict[str, list[str]] = {}
    name_to_tracks: dict[str, list[str]] = {}
    n_mbdb, n_scrobble, n_empty = 0, 0, 0
    for name in all_names:
        mbid = name_to_mbid.get(name)
        if mbid and mbid in mbid_to_disco:
            albums, tracks = mbid_to_disco[mbid]
            name_to_albums[name] = albums
            name_to_tracks[name] = tracks
            n_mbdb += 1
        else:
            albums = scrobble_albums.get(name, [])
            tracks = scrobble_tracks.get(name, [])
            if albums or tracks:
                n_scrobble += 1
            else:
                n_empty += 1
            name_to_albums[name] = albums
            name_to_tracks[name] = tracks
    log.info("  Unified lookups: %d MBDB, %d scrobble fallback, %d empty.", n_mbdb, n_scrobble, n_empty)
    log.info(
        "  Total: %d names with albums, %d with tracks.",
        sum(1 for v in name_to_albums.values() if v),
        sum(1 for v in name_to_tracks.values() if v),
    )
    return name_to_albums, name_to_tracks


# ═════════════════════════════════════════════════════════════════════════════
# Discography features (album overlap — δίσκος)
# ═════════════════════════════════════════════════════════════════════════════
def _jaccard(a: set, b: set) -> float:
    """Returns Jaccard index, 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _fuzzy_overlap(
    list_a: list[str],
    list_b: list[str],
    threshold: int,
) -> tuple[int, float]:
    """Counts items in list_a that fuzzy-match any item in list_b.

    Uses rapidfuzz.process.extractOne (C-optimised) for speed.
    Returns (n_matched, ratio).
    """
    if not list_a or not list_b:
        return 0, 0.0
    matched = 0
    for a in list_a:
        result = process.extractOne(
            a,
            list_b,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is not None:
            matched += 1
    total = max(len(set(list_a) | set(list_b)), 1)
    return matched, matched / total


def compute_discography_features(albums_a: list[str], albums_b: list[str]) -> dict[str, float]:
    """Computes 5 discography (album) overlap features for a single pair."""
    albums_a = albums_a[:MAX_ALBUMS]
    albums_b = albums_b[:MAX_ALBUMS]
    set_aa, set_ab = set(albums_a), set(albums_b)
    n_fuzzy_alb, fuzzy_alb_ratio = _fuzzy_overlap(albums_a, albums_b, FUZZY_ALBUM_THR)
    return {
        "disco_fuzzy_album_ratio": fuzzy_alb_ratio,
        "disco_has_fuzzy_album_match": float(n_fuzzy_alb > 0),
        "disco_n_fuzzy_album_matches": n_fuzzy_alb,
        "disco_exact_album_jaccard": _jaccard(set_aa, set_ab),
        "disco_min_album_count": min(len(set_aa), len(set_ab)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Melography features (track overlap — μέλος)
# ═════════════════════════════════════════════════════════════════════════════
def compute_melography_features(tracks_a: list[str], tracks_b: list[str]) -> dict[str, float]:
    """Computes 5 melography (track) overlap features for a single pair."""
    tracks_a = tracks_a[:MAX_TRACKS]
    tracks_b = tracks_b[:MAX_TRACKS]
    set_ta, set_tb = set(tracks_a), set(tracks_b)
    n_fuzzy_trk, fuzzy_trk_ratio = _fuzzy_overlap(tracks_a, tracks_b, FUZZY_TRACK_THR)
    return {
        "melo_fuzzy_track_ratio": fuzzy_trk_ratio,
        "melo_has_fuzzy_track_match": float(n_fuzzy_trk > 0),
        "melo_n_fuzzy_track_matches": n_fuzzy_trk,
        "melo_exact_track_jaccard": _jaccard(set_ta, set_tb),
        "melo_min_track_count": min(len(set_ta), len(set_tb)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Combined catalogue feature addition
# ═════════════════════════════════════════════════════════════════════════════
def add_catalogue_features(
    df: pd.DataFrame,
    name_to_albums: dict[str, list[str]],
    name_to_tracks: dict[str, list[str]],
) -> pd.DataFrame:
    """Adds discography + melography features using unified catalogue lookups."""
    n = len(df)
    log.info("Computing catalogue features for %d pairs...", n)
    disco_rows: list[dict] = []
    melo_rows: list[dict] = []
    n_no_disco, n_no_melo = 0, 0
    for i, (_, row) in enumerate(df.iterrows()):
        va, vb = str(row["variant_a"]), str(row["variant_b"])
        albums_a = name_to_albums.get(va, [])
        albums_b = name_to_albums.get(vb, [])
        tracks_a = name_to_tracks.get(va, [])
        tracks_b = name_to_tracks.get(vb, [])
        if not albums_a and not albums_b:
            n_no_disco += 1
        if not tracks_a and not tracks_b:
            n_no_melo += 1
        disco_rows.append(compute_discography_features(albums_a, albums_b))
        melo_rows.append(compute_melography_features(tracks_a, tracks_b))
        if (i + 1) % 200 == 0:
            log.info("  Catalogue features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    if n_no_disco:
        log.warning("  %d pairs had no album data on either side.", n_no_disco)
    if n_no_melo:
        log.warning("  %d pairs had no track data on either side.", n_no_melo)
    disco_df = pd.DataFrame(disco_rows, index=df.index)
    melo_df = pd.DataFrame(melo_rows, index=df.index)
    for col in disco_df.columns:
        df[col] = disco_df[col]
    for col in melo_df.columns:
        df[col] = melo_df[col]
    log.info("  Catalogue features added: %d disco + %d melo columns.", len(disco_df.columns), len(melo_df.columns))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Base + interaction features (no group-level length_stats)
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
    """Computes base (23) + interaction (30) features. No group-level length_stats."""
    n = len(df)
    log.info("Computing base features for %d pairs...", n)
    feat_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = str(row["variant_a"]), str(row["variant_b"])
        feats = compute_pair_features(a, b)
        feat_rows.append(feats)
        if (i + 1) % 200 == 0:
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
    # Deliberately skipping length_stats() — group-level features not wanted
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
    """Trains 5 tree-based models with MLflow tracking and threshold sweep."""
    target = "to_link"
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info("Train: %d | Test: %d | Features: %d | spw: %.2f", len(X_train), len(X_test), len(num_cols), spw)
    log.info(
        "Train distribution: pos=%d, neg=%d (%.1f%% positive)",
        y_train.sum(),
        (y_train == 0).sum(),
        100 * y_train.mean(),
    )
    log.info(
        "Test distribution: pos=%d, neg=%d (%.1f%% positive)", y_test.sum(), (y_test == 0).sum(), 100 * y_test.mean()
    )
    catalogue = _build_model_catalogue(spw, device, random_state=RANDOM_STATE)
    catalogue = {k: v for k, v in catalogue.items() if k in _MODELS_TO_RUN}
    # Initialising MLflow
    experiment.init_experiment()
    results = []
    with experiment.start_run(run_name="exp14_unified_catalogue"):
        experiment.log_params(
            {
                "experiment": 14,
                "experiment_type": "avc_pair_level_split",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "wratio_lower": WRATIO_LOWER,
                "wratio_upper": WRATIO_UPPER,
                "max_tracks": MAX_TRACKS,
                "max_albums": MAX_ALBUMS,
                "fuzzy_track_thr": FUZZY_TRACK_THR,
                "fuzzy_album_thr": FUZZY_ALBUM_THR,
                "n_features": len(num_cols),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_pos": int(y_train.sum()),
                "train_neg": int((y_train == 0).sum()),
                "test_pos": int(y_test.sum()),
                "test_neg": int((y_test == 0).sum()),
                "spw": round(spw, 2),
                "device_probed": device,
                "model_count": len(catalogue),
                "catalogue_source": "unified_mbdb_solo_plus_scrobble",
                "group_features": "none",
            }
        )
        # Logging train/test splits as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "exp14_train.parquet"
            test_path = Path(tmpdir) / "exp14_test.parquet"
            dump_parquet(train_df, train_path)
            dump_parquet(test_df, test_path)
            experiment.log_artifact(train_path)
            experiment.log_artifact(test_path)
        for model_name, clf in catalogue.items():
            log.info("─── Training %s ───", model_name)
            with experiment.start_run(run_name=model_name, nested=True):
                import mlflow

                mlflow.set_tag("model_type", model_name)
                safe_params = _safe_get_params(clf)
                safe_params["device_used"] = device
                experiment.log_params(safe_params)
                pre = ColumnTransformer(
                    [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
                    remainder="drop",
                    verbose_feature_names_out=False,
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
                # Logging metrics to MLflow
                experiment.log_metrics(
                    {
                        "auc": auc,
                        "default_f1": default_m["f1"],
                        "default_precision": default_m["precision"],
                        "default_recall": default_m["recall"],
                        "opt_threshold": optimal_m["threshold"],
                        "opt_f1": optimal_m["f1"],
                        "opt_precision": optimal_m["precision"],
                        "opt_recall": optimal_m["recall"],
                        "hiprec_threshold": best_hi["threshold"],
                        "hiprec_f1": best_hi["f1"],
                        "hiprec_precision": best_hi["precision"],
                        "hiprec_recall": best_hi["recall"],
                    }
                )
                results.append(
                    {
                        "model": model_name,
                        "auc": auc,
                        "default_f1": default_m["f1"],
                        "default_prec": default_m["precision"],
                        "default_rec": default_m["recall"],
                        "opt_thr": optimal_m["threshold"],
                        "opt_f1": optimal_m["f1"],
                        "opt_prec": optimal_m["precision"],
                        "opt_rec": optimal_m["recall"],
                        "hiprec_thr": best_hi["threshold"],
                        "hiprec_f1": best_hi["f1"],
                        "hiprec_prec": best_hi["precision"],
                        "hiprec_rec": best_hi["recall"],
                    }
                )
                log.info(
                    "%s → AUC=%.4f | def F1=%.4f | opt F1=%.4f (thr=%.3f) | hi-P F1=%.4f (thr=%.3f)",
                    model_name,
                    auc,
                    default_m["f1"],
                    optimal_m["f1"],
                    opt_thr,
                    best_hi["f1"],
                    best_hi["threshold"],
                )
                y_pred_opt = (y_prob >= opt_thr).astype(int)
                print(f"\n=== {model_name} (optimal thr={opt_thr:.3f}) ===")
                print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
                # Logging artefacts
                experiment.log_confusion_matrix(y_test, y_pred_opt)
                experiment.log_feature_importance(pipeline, num_cols)
                X_test_transformed = pipeline.named_steps["prep"].transform(X_test)
                experiment.log_shap_summary(pipeline, X_test_transformed, num_cols)
                experiment.log_model(pipeline)
                # Printing feature importance to stdout
                clf_fitted = pipeline.named_steps["clf"]
                if hasattr(clf_fitted, "feature_importances_"):
                    imps = sorted(zip(num_cols, clf_fitted.feature_importances_), key=lambda x: x[1], reverse=True)
                    print(f"Top 15 features ({model_name}):")
                    for name, imp in imps[:15]:
                        print(f"  {name:<40} {imp:.4f}")
                    dm_feats = [(n, v) for n, v in imps if n.startswith(("disco_", "melo_"))]
                    if dm_feats:
                        print("  ── Disco/melo features ──")
                        for name, imp in dm_feats:
                            print(f"  {name:<40} {imp:.4f}")
    # Summary table
    print("\n" + "=" * 130)
    print(
        f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
        f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
        f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}"
    )
    print("-" * 130)
    for r in sorted(results, key=lambda x: x["opt_f1"], reverse=True):
        print(
            f"{r['model']:<22} {r['auc']:>6.4f} | "
            f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} {r['default_f1']:>6.4f} | "
            f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} {r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
            f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} {r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}"
        )
    print("=" * 130)
    print("\n── Baselines ──")
    print("Exp 6  ExtraTrees (gs_mb_dbscan, full AVC test):  AUC=0.8920, opt F1=0.7050 (P=0.750, R=0.667, thr=0.940)")
    print("Exp 13 XGBoost (AVC-only, group split, scrobble): AUC=0.9968, opt F1=0.9524 (P=0.909, R=1.000, thr=0.410)")
    print("Exp 14 note: pair-level split (no group-level features), unified MBDB-solo + scrobble catalogue.")
    # Selecting best model by c9r composite score (0.4×HiP_P + 0.3×HiP_F1 + 0.3×AUC)
    best = max(results, key=lambda x: 0.4 * x["hiprec_prec"] + 0.3 * x["hiprec_f1"] + 0.3 * x["auc"])
    score = 0.4 * best["hiprec_prec"] + 0.3 * best["hiprec_f1"] + 0.3 * best["auc"]
    print(f"\nBest model by c9r score: {best['model']} (score={score:.4f})")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 14 pipeline."""
    log.info("=== Experiment 14: Unified Catalogue Lookup with Flattened Pairs ===")
    log.info("MAX_TRACKS=%d, MAX_ALBUMS=%d", MAX_TRACKS, MAX_ALBUMS)
    # ── Step 1: Pair-level flattening and split ────────────────────────────
    train_pairs, test_pairs = step1_flatten_and_split()
    # ── Step 2: Solo-credit MBDB discography ───────────────────────────────
    disco_df = step2_extract_solo_discographies()
    mbid_to_disco: dict[str, tuple[list[str], list[str]]] = {}
    for _, row in disco_df.iterrows():
        mbid_to_disco[row["mbid"]] = (
            _parse_delimited(row["albums_str"]),
            _parse_delimited(row["tracks_str"]),
        )
    log.info("Solo-credit disco lookup: %d MBIDs.", len(mbid_to_disco))
    # ── Step 3: Unified catalogue lookups ──────────────────────────────────
    name_to_albums, name_to_tracks = build_unified_lookups(mbid_to_disco)
    # ── Step 4: Features ───────────────────────────────────────────────────
    log.info("Computing base features for training set...")
    train_df = add_base_features(train_pairs)
    log.info("Computing base features for test set...")
    test_df = add_base_features(test_pairs)
    log.info("Adding catalogue features for training set...")
    train_df = add_catalogue_features(train_df, name_to_albums, name_to_tracks)
    log.info("Adding catalogue features for test set...")
    test_df = add_catalogue_features(test_df, name_to_albums, name_to_tracks)
    # ── Step 5: Pruning ────────────────────────────────────────────────────
    target = "to_link"
    exclude = {target, "variant_a", "variant_b", "source", "_key"}
    all_num = [
        c
        for c in train_df.columns
        if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_features(train_df[all_num])
    # Reporting disco/melo feature survival
    disco_survived = [c for c in num_cols if c.startswith("disco_")]
    melo_survived = [c for c in num_cols if c.startswith("melo_")]
    log.info(
        "Post-pruning: %d features (%d disco, %d melo survived).",
        len(num_cols),
        len(disco_survived),
        len(melo_survived),
    )
    # Ensuring test_df has the same columns
    missing = [c for c in num_cols if c not in test_df.columns]
    if missing:
        log.warning("Columns missing in test_df (filling with 0): %s", missing)
        for c in missing:
            test_df[c] = 0.0
    # ── Step 6: Run experiment ─────────────────────────────────────────────
    log.info("Running Experiment 14...")
    results = run_experiment(train_df, test_df, num_cols)
    return results


if __name__ == "__main__":
    main()
