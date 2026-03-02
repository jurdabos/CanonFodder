"""
Unified training pipeline for ``c9r train``.

Consolidates the best practices from Experiments 1–15 into a single,
reproducible entry point with full MLflow tracking.  Replaces the legacy
``model.py:train_model()`` as the primary training path.

Pipeline stages:
  1. Pre-verify MLflow tracking is reachable and writable.
  2. Build training data: AVC pair-level flattening → WRatio filter →
     stratified split.
  3. Compute features: base 23 + interaction 30 + catalogue 18 (optional).
  4. Prune features: variance (0.001) + correlation (0.95).
  5. Train: k-fold stratified CV with per-fold nested MLflow runs,
     final model on full training set, held-out evaluation at 3 operating
     points (default 0.5, F1-optimal, high-precision P≥0.80).
"""

from __future__ import annotations
import itertools
import logging
import math
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
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from corefunc.canon.experiment_runner import _build_model_catalogue, _safe_get_params
from helpers import cluster, experiment, stats
from helpers.device import get_device
from helpers.features import compute_pair_features
from helpers.io import (
    AVC_PQ,
    GS_MB_PQ,
    PQ_DIR,
    read_parquet,
    read_scrobble_df,
    dump_parquet,
    sanitize,
)
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
MAX_TRACKS = 10_000
MAX_ALBUMS = 1_000
SOLO_DISCO_PQ = PQ_DIR / "mbdb_discography_solo.parquet"
_SIM_SCORES = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "WRatio",
    "QRatio",
]
_TREE_MODELS = {
    "XGBoost",
    "RandomForest",
    "ExtraTrees",
    "LightGBM",
    "GradientBoosting",
}
DEFAULT_MODELS = ["LightGBM"]


# ═════════════════════════════════════════════════════════════════════════════
# MLflow pre-verification
# ═════════════════════════════════════════════════════════════════════════════
def verify_mlflow() -> None:
    """Tests that the MLflow tracking URI is reachable and writable.

    Starts and immediately ends a dummy run.  Raises RuntimeError on
    failure so callers can abort before expensive feature computation.
    """
    import mlflow

    experiment.init_experiment()
    try:
        with mlflow.start_run(run_name="_verify_mlflow_tracking"):
            mlflow.log_param("_verify", True)
        log.info("MLflow pre-verification passed.")
    except Exception as exc:
        raise RuntimeError(
            f"MLflow tracking verification failed: {exc}.  "
            f"Check that the tracking URI ({experiment.TRACKING_URI}) is accessible."
        ) from exc


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Pair-level data building
# ═════════════════════════════════════════════════════════════════════════════
def build_training_data(
    *,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expands AVC groups to pairs, filters by WRatio, and splits.

    Returns (train_pairs_df, test_pairs_df) with columns
    ``[variant_a, variant_b, to_link]``.
    """
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        raise RuntimeError("avc.parquet is empty or missing — nothing to train on.")
    decided = avc[avc["to_link"].notna()].reset_index(drop=True)
    log.info(
        "AVC decided rows: %d (pos=%d, neg=%d).",
        len(decided),
        decided["to_link"].sum(),
        (~decided["to_link"]).sum(),
    )
    # Expanding to pair level
    all_rows: list[tuple] = []
    for _, row in decided.iterrows():
        all_rows.extend(cluster.expand_pairs(row))
    all_pairs = pd.DataFrame(
        all_rows,
        columns=["variants", "variant_a", "variant_b", "to_link"],
    )
    all_pairs = all_pairs.drop_duplicates(
        subset=["variant_a", "variant_b"],
    ).reset_index(drop=True)
    log.info(
        "Expanded to %d unique pairs (pos=%d, neg=%d).",
        len(all_pairs),
        all_pairs["to_link"].sum(),
        (~all_pairs["to_link"]).sum(),
    )
    # Applying WRatio band filter
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
    # Performing stratified split
    train_pairs, test_pairs = train_test_split(
        all_pairs,
        test_size=test_size,
        random_state=random_state,
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
# Step 2: Catalogue lookups
# ═════════════════════════════════════════════════════════════════════════════
def _parse_delimited(s: str) -> list[str]:
    """Splits a ``{``-delimited string into a list, filtering blanks."""
    if not s or pd.isna(s):
        return []
    return [x for x in s.split("{") if x.strip()]


def _load_catalogue_lookups() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Builds name→albums and name→tracks dicts from MBDB + scrobble fallback.

    Uses the solo-credit discography cache if available; otherwise falls
    back to scrobble-only data.
    """
    # Loading MBDB solo-credit disco cache
    mbid_to_disco: dict[str, tuple[list[str], list[str]]] = {}
    if SOLO_DISCO_PQ.exists():
        disco_df = read_parquet(SOLO_DISCO_PQ)
        for _, row in disco_df.iterrows():
            mbid_to_disco[row["mbid"]] = (
                _parse_delimited(row["albums_str"]),
                _parse_delimited(row["tracks_str"]),
            )
        log.info("Solo-credit disco lookup: %d MBIDs.", len(mbid_to_disco))
    else:
        log.warning(
            "mbdb_discography_solo.parquet not found — using scrobble-only catalogue.",
        )
    # Building unified lookups (Exp 14/15 approach)
    scrobbles = read_scrobble_df()
    if scrobbles is None or scrobbles.empty:
        log.warning("No scrobble data — catalogue features will be zeros.")
        return {}, {}
    has_mbid = scrobbles["artist_mbid"].notna() & (scrobbles["artist_mbid"].str.len() == 36)
    name_to_mbid: dict[str, str] = (
        scrobbles[has_mbid].drop_duplicates("artist_name").set_index("artist_name")["artist_mbid"].to_dict()
    )
    clean = scrobbles[scrobbles["album_title"].notna() & (scrobbles["album_title"].str.strip() != "")]
    scrobble_albums = clean.groupby("artist_name")["album_title"].apply(lambda x: sorted(set(x.unique()))).to_dict()
    scrobble_tracks = (
        scrobbles.groupby("artist_name")["track_title"]
        .apply(lambda x: sorted({t for t in x.dropna().unique() if t.strip()}))
        .to_dict()
    )
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
    log.info(
        "Unified lookups: %d MBDB, %d scrobble fallback, %d empty.",
        n_mbdb,
        n_scrobble,
        n_empty,
    )
    return name_to_albums, name_to_tracks


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Feature engineering
# ═════════════════════════════════════════════════════════════════════════════
def _collect_best_scores(list_a: list[str], list_b: list[str]) -> list[float]:
    """Collects the best fuzzy-match score for each item in *list_a* against *list_b*.

    Uses ``rapidfuzz.process.extractOne`` with ``token_sort_ratio``.
    Returns a list of floats (one per item in *list_a*), 0.0 for no match.
    """
    if not list_a or not list_b:
        return []
    scores: list[float] = []
    for a in list_a:
        result = process.extractOne(
            a,
            list_b,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=0,
        )
        scores.append(float(result[1]) if result is not None else 0.0)
    return scores


def _presence_features(
    items_a: list[str],
    items_b: list[str],
    prefix: str,
) -> dict[str, float]:
    """Computes 9 presence-and-quality features for a single catalogue domain.

    Designed to be coverage-invariant: whether one side has 3 items or
    300, the features capture whether real overlap exists and how strong
    the best evidence is.
    """
    items_a = items_a[: MAX_ALBUMS if "disco" in prefix else MAX_TRACKS]
    items_b = items_b[: MAX_ALBUMS if "disco" in prefix else MAX_TRACKS]
    set_a, set_b = set(items_a), set(items_b)
    exact_matches = set_a & set_b
    n_exact = len(exact_matches)
    # Collecting per-item best fuzzy scores (both directions)
    scores_a = _collect_best_scores(items_a, items_b)
    scores_b = _collect_best_scores(items_b, items_a)
    all_scores = scores_a + scores_b
    best_score = max(all_scores) if all_scores else 0.0
    top_sorted = sorted(all_scores, reverse=True)
    top3 = top_sorted[:3]
    top3_avg = sum(top3) / len(top3) if top3 else 0.0
    # Counting matches at multiple thresholds
    n_70 = sum(1 for s in all_scores if s >= 70)
    n_80 = sum(1 for s in all_scores if s >= 80)
    n_90 = sum(1 for s in all_scores if s >= 90)
    return {
        f"{prefix}_best_match_score": best_score,
        f"{prefix}_top3_avg_score": top3_avg,
        f"{prefix}_any_exact_match": float(n_exact > 0),
        f"{prefix}_n_exact_matches": n_exact,
        f"{prefix}_n_matches_70": n_70,
        f"{prefix}_n_matches_80": n_80,
        f"{prefix}_n_matches_90": n_90,
        f"{prefix}_log_fuzzy_matches": math.log1p(n_80),
        f"{prefix}_min_count": min(len(set_a), len(set_b)),
    }


def _add_catalogue_features(
    df: pd.DataFrame,
    name_to_albums: dict[str, list[str]],
    name_to_tracks: dict[str, list[str]],
) -> pd.DataFrame:
    """Adds presence-and-quality catalogue features using unified lookups."""
    n = len(df)
    log.info("Computing catalogue features for %d pairs...", n)
    feat_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        va, vb = str(row["variant_a"]), str(row["variant_b"])
        albums_a = name_to_albums.get(va, [])
        albums_b = name_to_albums.get(vb, [])
        tracks_a = name_to_tracks.get(va, [])
        tracks_b = name_to_tracks.get(vb, [])
        feats: dict[str, float] = {}
        feats.update(_presence_features(albums_a, albums_b, "disco"))
        feats.update(_presence_features(tracks_a, tracks_b, "melo"))
        feat_rows.append(feats)
        if (i + 1) % 200 == 0:
            log.info("  Catalogue features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    cat_df = pd.DataFrame(feat_rows, index=df.index)
    for col in cat_df.columns:
        df[col] = cat_df[col]
    return df


def _compute_interaction_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise diffs and products among 6 similarity scores (vectorised)."""
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


def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base (23) + interaction (30) features for each pair."""
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
    log.info("Computing interaction features...")
    interaction_df = _compute_interaction_features(feat_df)
    for col in interaction_df.columns:
        if col not in df.columns:
            df[col] = interaction_df[col]
    return df


def compute_all_features(
    df: pd.DataFrame,
    *,
    catalogue: bool = True,
    cat_design: str = "proportional",
    name_to_albums: dict[str, list[str]] | None = None,
    name_to_tracks: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Computes base 23 + interaction 30 + optional catalogue features.

    When *catalogue* is True, *name_to_albums* and *name_to_tracks* must
    be provided (from ``_load_catalogue_lookups``).
    Uses proportional (10 features) or presence (18 features) catalogue
    design based on *cat_design*.
    """
    df = _add_base_features(df)
    if catalogue:
        if name_to_albums is None or name_to_tracks is None:
            raise ValueError("Catalogue lookups required when catalogue=True.")
        if cat_design == "proportional":
            df = _add_proportional_catalogue_features(df, name_to_albums, name_to_tracks)
        else:
            df = _add_catalogue_features(df, name_to_albums, name_to_tracks)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Feature pruning
# ═════════════════════════════════════════════════════════════════════════════
def prune_feature_columns(X: pd.DataFrame) -> list[str]:
    """Applies variance (0.001) and Spearman correlation (0.95) pruning.

    Returns the list of surviving feature column names.
    """
    var_df, selected = stats.variance_testing(X, 0.001)
    X_pruned = X[selected]
    log.info("After variance pruning: %d → %d features.", len(X.columns), len(X_pruned.columns))
    X_pruned = stats.iterative_correlation_dropper(X_pruned, 0.95, var_df, 20)
    log.info("After correlation pruning: → %d features.", len(X_pruned.columns))
    return list(X_pruned.columns)


# ═════════════════════════════════════════════════════════════════════════════
# Step 2b: Scrobble-only catalogue lookups (Exp 13)
# ═════════════════════════════════════════════════════════════════════════════
def _load_scrobble_only_lookups() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Builds name→albums and name→tracks dicts from scrobble.parquet only.

    Used by Exp 13 where catalogue features come exclusively from scrobbles.
    """
    scrobbles = read_scrobble_df()
    if scrobbles is None or scrobbles.empty:
        log.warning("No scrobble data — catalogue features will be zeros.")
        return {}, {}
    clean = scrobbles[scrobbles["album_title"].notna() & (scrobbles["album_title"].str.strip() != "")]
    name_to_albums = clean.groupby("artist_name")["album_title"].apply(lambda x: sorted(set(x.unique()))).to_dict()
    name_to_tracks = (
        scrobbles.groupby("artist_name")["track_title"]
        .apply(lambda x: sorted({t for t in x.dropna().unique() if t.strip()}))
        .to_dict()
    )
    log.info(
        "Scrobble-only lookups: %d album entries, %d track entries.",
        len(name_to_albums),
        len(name_to_tracks),
    )
    return name_to_albums, name_to_tracks


# ═════════════════════════════════════════════════════════════════════════════
# Step 3b: Proportional catalogue features (Exp 12–13 design)
# ═════════════════════════════════════════════════════════════════════════════
def _jaccard_set(a: set, b: set) -> float:
    """Returns Jaccard index, 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _fuzzy_overlap(
    list_a: list[str],
    list_b: list[str],
    threshold: int = 80,
) -> tuple[int, float]:
    """Counts items in list_a that fuzzy-match any item in list_b.

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


def _proportional_disco_features(albums_a: list[str], albums_b: list[str]) -> dict[str, float]:
    """Computes 5 proportional discography features (Exp 12–13 style)."""
    albums_a = albums_a[:MAX_ALBUMS]
    albums_b = albums_b[:MAX_ALBUMS]
    set_aa, set_ab = set(albums_a), set(albums_b)
    n_fuzzy, fuzzy_ratio = _fuzzy_overlap(albums_a, albums_b, 80)
    return {
        "disco_fuzzy_album_ratio": fuzzy_ratio,
        "disco_has_fuzzy_album_match": float(n_fuzzy > 0),
        "disco_n_fuzzy_album_matches": n_fuzzy,
        "disco_exact_album_jaccard": _jaccard_set(set_aa, set_ab),
        "disco_min_album_count": min(len(set_aa), len(set_ab)),
    }


def _proportional_melo_features(tracks_a: list[str], tracks_b: list[str]) -> dict[str, float]:
    """Computes 5 proportional melography features (Exp 12–13 style)."""
    tracks_a = tracks_a[:MAX_TRACKS]
    tracks_b = tracks_b[:MAX_TRACKS]
    set_ta, set_tb = set(tracks_a), set(tracks_b)
    n_fuzzy, fuzzy_ratio = _fuzzy_overlap(tracks_a, tracks_b, 80)
    return {
        "melo_fuzzy_track_ratio": fuzzy_ratio,
        "melo_has_fuzzy_track_match": float(n_fuzzy > 0),
        "melo_n_fuzzy_track_matches": n_fuzzy,
        "melo_exact_track_jaccard": _jaccard_set(set_ta, set_tb),
        "melo_min_track_count": min(len(set_ta), len(set_tb)),
    }


def _add_proportional_catalogue_features(
    df: pd.DataFrame,
    name_to_albums: dict[str, list[str]],
    name_to_tracks: dict[str, list[str]],
) -> pd.DataFrame:
    """Adds proportional catalogue features (Exp 12–13 Jaccard/fuzzy_ratio design)."""
    n = len(df)
    log.info("Computing proportional catalogue features for %d pairs...", n)
    feat_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        va, vb = str(row["variant_a"]), str(row["variant_b"])
        feats: dict[str, float] = {}
        feats.update(
            _proportional_disco_features(
                name_to_albums.get(va, []),
                name_to_albums.get(vb, []),
            )
        )
        feats.update(
            _proportional_melo_features(
                name_to_tracks.get(va, []),
                name_to_tracks.get(vb, []),
            )
        )
        feat_rows.append(feats)
        if (i + 1) % 200 == 0:
            log.info("  Proportional catalogue: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    cat_df = pd.DataFrame(feat_rows, index=df.index)
    for col in cat_df.columns:
        df[col] = cat_df[col]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Alternate data builders (Exp 1–12)
# ═════════════════════════════════════════════════════════════════════════════
def _build_avc_full_test(
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> pd.DataFrame:
    """Expands all decided AVC rows to pairs and filters by WRatio.

    Used as the cross-domain held-out test set for Exp 4–8, 11–12.
    """
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        raise RuntimeError("avc.parquet is empty or missing.")
    decided = avc[avc["to_link"].notna()].reset_index(drop=True)
    all_rows: list[tuple] = []
    for _, row in decided.iterrows():
        all_rows.extend(cluster.expand_pairs(row))
    test_df = pd.DataFrame(
        all_rows,
        columns=["variants", "variant_a", "variant_b", "to_link"],
    )
    test_df["_wr"] = test_df.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    test_df = (
        test_df[(test_df["_wr"] >= wratio_lower) & (test_df["_wr"] < wratio_upper)]
        .drop(columns=["_wr", "variants"])
        .reset_index(drop=True)
    )
    log.info(
        "AVC-full test: %d pairs (pos=%d, neg=%d).",
        len(test_df),
        test_df["to_link"].sum(),
        (~test_df["to_link"]).sum(),
    )
    return test_df


def _build_mbdb_training_data(
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> pd.DataFrame:
    """Loads gs_mb.parquet + gs_mb_backup.parquet, filters by WRatio.

    Used for Exp 4–5 (MBDB-trained, AVC-tested).
    """
    GS_MB_BACKUP_PQ = PQ_DIR / "gs_mb_backup.parquet"
    frames = []
    for pq_path in [GS_MB_PQ, GS_MB_BACKUP_PQ]:
        df = read_parquet(pq_path)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No gs_mb parquet files found — run 'c9r canon avc augment' first.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["variant_a", "variant_b"])
    # Deduplicating (order-insensitive pair keys)
    combined["_key"] = combined.apply(
        lambda r: tuple(sorted([str(r["variant_a"]), str(r["variant_b"])])),
        axis=1,
    )
    combined = combined.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    # Applying WRatio filter
    combined["_wr"] = combined.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    combined = (
        combined[(combined["_wr"] >= wratio_lower) & (combined["_wr"] < wratio_upper)]
        .drop(columns=["_wr"])
        .reset_index(drop=True)
    )
    log.info(
        "MBDB train: %d pairs (pos=%d, neg=%d).",
        len(combined),
        combined["to_link"].sum(),
        (~combined["to_link"]).sum(),
    )
    return combined


def _build_dbscan_training_data(
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> pd.DataFrame:
    """Loads gs_mb_dbscan.parquet, filters by WRatio.

    Used for Exp 6 (DBSCAN-seeded, MBDB-verified).
    """
    GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
    dbscan = read_parquet(GS_DBSCAN_PQ)
    if dbscan is None or dbscan.empty:
        raise RuntimeError("gs_mb_dbscan.parquet not found — run the DBSCAN seeding step first.")
    dbscan["_wr"] = dbscan.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    dbscan = (
        dbscan[(dbscan["_wr"] >= wratio_lower) & (dbscan["_wr"] < wratio_upper)]
        .drop(columns=["_wr"])
        .reset_index(drop=True)
    )
    log.info(
        "DBSCAN train: %d pairs (pos=%d, neg=%d).",
        len(dbscan),
        dbscan["to_link"].sum(),
        (~dbscan["to_link"]).sum(),
    )
    return dbscan


def _build_dbscan_capped_training_data(
    *,
    cluster_cap: int = 30,
    neg_ratio: int = 10,
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> pd.DataFrame:
    """Builds training data from capped DBSCAN clusters with negative subsampling.

    Used for Exp 7.  Caps oversized clusters at *cluster_cap*, preserving
    anchor names (known positive AVC pairs).  Then subsamples negatives
    to *neg_ratio*:1 stratified by WRatio bands.
    """
    GS_DBSCAN_CAPPED_PQ = PQ_DIR / "gs_mb_dbscan_capped.parquet"
    if GS_DBSCAN_CAPPED_PQ.exists():
        cached = read_parquet(GS_DBSCAN_CAPPED_PQ)
        if cached is not None and not cached.empty:
            log.info("Loaded gs_mb_dbscan_capped.parquet: %d pairs.", len(cached))
            return cached
    # ── Running capped DBSCAN pipeline ─────────────────────────────────────
    from corefunc.mb_local import _psql_csv, _escape_pg, check_local_mb

    scrobbles = read_scrobble_df()
    artist_names = sorted(scrobbles["artist_name"].dropna().unique().tolist())
    n = len(artist_names)
    log.info("Capped DBSCAN: %d unique artist names, cluster_cap=%d.", n, cluster_cap)
    # Building anchor sets from positive AVC groups
    avc = read_parquet(AVC_PQ)
    pos_groups = avc[(avc["to_link"].notna()) & (avc["to_link"].eq(True))]
    name2idx = {name: i for i, name in enumerate(artist_names)}
    anchor_idx_sets: list[list[int]] = []
    anchor_names: set[str] = set()
    for _, row in pos_groups.iterrows():
        variants = [v.strip() for v in row["artist_variants_text"].split("{") if v.strip()]
        idxs = [name2idx[v] for v in variants if v in name2idx]
        if len(idxs) >= 2:
            anchor_idx_sets.append(idxs)
            anchor_names.update(v for v in variants if v in name2idx)
    log.info("Anchor sets: %d (protecting %d anchor names).", len(anchor_idx_sets), len(anchor_names))
    # Computing distance matrix
    log.info("Computing %d×%d distance matrix...", n, n)
    sim_matrix = process.cdist(artist_names, artist_names, scorer=fuzz.WRatio, score_cutoff=0, workers=-1) / 100.0
    dist = 1.0 - sim_matrix
    # Grid search for epsilon
    log.info("Running ε grid search [0.05, 1.0)...")
    best_eps = None
    for eps in np.arange(0.05, 1.0, 0.01):
        labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
        if cluster.anchors_ok(labels, anchor_idx_sets):
            best_eps = eps
            break
    if best_eps is None:
        raise RuntimeError("No ε satisfies all anchor constraints.")
    labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
    log.info("Best ε = %.2f", best_eps)
    # Building groups and capping oversized clusters
    df_labels = pd.DataFrame({"artist": artist_names, "label": labels})
    clustered = df_labels[df_labels["label"] != -1]
    raw_groups = clustered.groupby("label")["artist"].apply(list).tolist()
    rng = np.random.default_rng(RANDOM_STATE)
    capped_groups: list[list[str]] = []
    for group in raw_groups:
        if len(group) <= cluster_cap:
            capped_groups.append(group)
        else:
            anchors_in = [nm for nm in group if nm in anchor_names]
            others = [nm for nm in group if nm not in anchor_names]
            budget = cluster_cap - len(anchors_in)
            sampled = (
                rng.choice(others, size=min(budget, len(others)), replace=False).tolist()
                if budget > 0 and others
                else []
            )
            capped_groups.append(anchors_in + sampled)
    log.info("Capped clusters: %d groups, %d total names.", len(capped_groups), sum(len(g) for g in capped_groups))
    # Extracting pairs
    rows: list[dict] = []
    for group in capped_groups:
        for a, b in itertools.combinations(sorted(set(group)), 2):
            rows.append({"variant_a": a, "variant_b": b})
    pairs_df = pd.DataFrame(rows)
    log.info("Candidate pairs from capped clusters: %d", len(pairs_df))
    # MBDB verification
    if not check_local_mb():
        raise RuntimeError("Local MBDB mirror not reachable.")
    all_names_unique = sorted(set(pairs_df["variant_a"]) | set(pairs_df["variant_b"]))
    log.info("Querying MBDB for %d unique names...", len(all_names_unique))
    name_to_mbids: dict[str, set[str]] = {}
    batch_size = 150
    for i in range(0, len(all_names_unique), batch_size):
        batch = all_names_unique[i : i + batch_size]
        values = ",".join(f"'{_escape_pg(nm)}'" for nm in batch)
        sql = f"""SELECT DISTINCT q.lookup_name, a.gid::text AS mbid
FROM (
    SELECT name AS lookup_name, id AS artist_id FROM musicbrainz.artist WHERE name IN ({values})
    UNION
    SELECT aa.name AS lookup_name, aa.artist AS artist_id FROM musicbrainz.artist_alias aa WHERE aa.name IN ({values})
) q
JOIN musicbrainz.artist a ON a.id = q.artist_id"""
        result = _psql_csv(sql)
        if not result.empty:
            for _, r in result.iterrows():
                name_to_mbids.setdefault(r["lookup_name"], set()).add(r["mbid"])
    verified_rows: list[dict] = []
    for _, pair in pairs_df.iterrows():
        a, b = pair["variant_a"], pair["variant_b"]
        mbids_a = name_to_mbids.get(a, set())
        mbids_b = name_to_mbids.get(b, set())
        if not mbids_a or not mbids_b:
            continue
        if mbids_a & mbids_b:
            verified_rows.append({"variant_a": a, "variant_b": b, "to_link": True, "source": "dbscan_mbdb_pos"})
        else:
            verified_rows.append({"variant_a": a, "variant_b": b, "to_link": False, "source": "dbscan_mbdb_neg"})
    verified_df = pd.DataFrame(verified_rows)
    # Applying WRatio filter
    verified_df["_wr"] = verified_df.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    filtered = verified_df[(verified_df["_wr"] >= wratio_lower) & (verified_df["_wr"] < wratio_upper)].copy()
    # Subsampling negatives to target ratio
    pos_df = filtered[filtered["to_link"].eq(True)]
    neg_df = filtered[filtered["to_link"].eq(False)]
    n_pos = len(pos_df)
    target_neg = min(n_pos * neg_ratio, len(neg_df)) if neg_ratio > 0 else len(neg_df)
    if len(neg_df) > target_neg:
        neg_df = neg_df.copy()
        neg_df["_band"] = pd.cut(neg_df["_wr"], bins=[60, 70, 80, 90, 95, 100], right=False)
        sampled_parts: list[pd.DataFrame] = []
        band_counts = neg_df["_band"].value_counts()
        total_in_bands = band_counts.sum()
        for band, count in band_counts.items():
            band_df = neg_df[neg_df["_band"] == band]
            n_sample = max(1, int(round(target_neg * count / total_in_bands)))
            n_sample = min(n_sample, len(band_df))
            sampled_parts.append(band_df.sample(n=n_sample, random_state=RANDOM_STATE))
        neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_band"])
        log.info("Subsampled negatives: %d → %d (target ratio %d:1).", len(neg_df), len(neg_sampled), neg_ratio)
    else:
        neg_sampled = neg_df
    combined = pd.concat([pos_df, neg_sampled], ignore_index=True).drop(columns=["_wr"]).reset_index(drop=True)
    log.info(
        "DBSCAN-capped train: %d pairs (pos=%d, neg=%d).",
        len(combined),
        combined["to_link"].sum(),
        (~combined["to_link"]).sum(),
    )
    dump_parquet(combined, GS_DBSCAN_CAPPED_PQ)
    return combined


def _build_feature_sep_training_data(
    *,
    neg_count: int = 5000,
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> pd.DataFrame:
    """Builds training data with distribution-matched negatives.

    Used for Exp 8.  Samples *neg_count* positives from gs_mb.parquet
    and distribution-matches the same number of negatives from
    gs_mb_dbscan.parquet using fine-grained WRatio histogram binning.
    """
    GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
    gs = read_parquet(GS_MB_PQ)
    if gs is None or gs.empty:
        raise RuntimeError("gs_mb.parquet not found — run 'c9r canon avc augment' first.")
    positives = gs[gs["to_link"].eq(True)].sample(
        n=min(neg_count, gs["to_link"].sum()),
        random_state=RANDOM_STATE,
    )
    log.info("Sampled %d positives from gs_mb.parquet.", len(positives))
    # Loading DBSCAN negative pool
    dbscan = read_parquet(GS_DBSCAN_PQ)
    if dbscan is None or dbscan.empty:
        raise RuntimeError("gs_mb_dbscan.parquet not found.")
    neg_pool = dbscan[dbscan["to_link"].eq(False)].reset_index(drop=True)
    log.info("DBSCAN negative pool: %d pairs.", len(neg_pool))
    # Computing WRatio for both sides
    pos_wr = positives.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    neg_wr = neg_pool.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    neg_pool = neg_pool.copy()
    neg_pool["_wr"] = neg_wr
    # Distribution matching using 8 bins in [60, 100)
    n_bins = 8
    bin_edges = np.linspace(60, 100, n_bins + 1)
    pos_in_range = pos_wr[(pos_wr >= 60) & (pos_wr < 100)]
    pos_hist, _ = np.histogram(pos_in_range, bins=bin_edges)
    pos_fracs = pos_hist / max(pos_hist.sum(), 1)
    n_target = min(neg_count, len(neg_pool))
    neg_pool["_bin"] = pd.cut(neg_pool["_wr"], bins=bin_edges, right=False, labels=False)
    neg_pool = neg_pool.dropna(subset=["_bin"])
    neg_pool["_bin"] = neg_pool["_bin"].astype(int)
    targets = (pos_fracs * n_target).astype(int)
    targets[np.argmax(pos_fracs)] += n_target - targets.sum()
    sampled_parts: list[pd.DataFrame] = []
    shortfall = 0
    available_bins: list[tuple] = []
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
    train = pd.concat(
        [
            positives[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
            neg_sampled[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    log.info(
        "Feature-sep train: %d pairs (pos=%d, neg=%d).",
        len(train),
        train["to_link"].sum(),
        (~train["to_link"]).sum(),
    )
    return train


def _build_mbdb_max_training_data(
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> pd.DataFrame:
    """Loads gs_mb_max.parquet positives + DBSCAN negatives, distribution-matched.

    Used for Exp 11–12 (MBDB-enlarged training).
    """
    GS_MB_MAX_PQ = PQ_DIR / "gs_mb_max.parquet"
    GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
    positives_all = read_parquet(GS_MB_MAX_PQ)
    if positives_all is None or positives_all.empty:
        raise RuntimeError("gs_mb_max.parquet not found — run Exp 10 first.")
    pos_wr = positives_all.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    mask = (pos_wr >= wratio_lower) & (pos_wr < wratio_upper)
    positives = positives_all[mask].reset_index(drop=True)
    # Loading DBSCAN negatives
    dbscan = read_parquet(GS_DBSCAN_PQ)
    if dbscan is None or dbscan.empty:
        raise RuntimeError("gs_mb_dbscan.parquet not found.")
    neg_pool = dbscan[dbscan["to_link"].eq(False)].reset_index(drop=True)
    n_target = min(len(positives), len(neg_pool))
    neg_sampled = neg_pool.sample(n=n_target, random_state=RANDOM_STATE)
    train = pd.concat(
        [
            positives[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
            neg_sampled[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    log.info(
        "MBDB-max train: %d pairs (pos=%d, neg=%d).",
        len(train),
        train["to_link"].sum(),
        (~train["to_link"]).sum(),
    )
    return train


def _build_mixed_training_data(
    test_size: float = 0.20,
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merges AVC expanded pairs + gs_mb pairs, random split.

    Used for Exp 1–3 (same-domain evaluation).
    """
    # Expanding AVC
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        raise RuntimeError("avc.parquet is empty or missing.")
    decided = avc[avc["to_link"].notna()].reset_index(drop=True)
    avc_rows: list[tuple] = []
    for _, row in decided.iterrows():
        avc_rows.extend(cluster.expand_pairs(row))
    avc_pairs = pd.DataFrame(
        avc_rows,
        columns=["variants", "variant_a", "variant_b", "to_link"],
    ).drop(columns=["variants"])
    # Loading MBDB
    mb = read_parquet(GS_MB_PQ)
    frames = [avc_pairs]
    if mb is not None and not mb.empty:
        frames.append(mb[["variant_a", "variant_b", "to_link"]])
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["variant_a", "variant_b"])
    combined = combined.drop_duplicates(
        subset=["variant_a", "variant_b"],
    ).reset_index(drop=True)
    # Applying WRatio filter
    combined["_wr"] = combined.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    combined = (
        combined[(combined["_wr"] >= wratio_lower) & (combined["_wr"] < wratio_upper)]
        .drop(columns=["_wr"])
        .reset_index(drop=True)
    )
    train, test = train_test_split(
        combined,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=combined["to_link"],
    )
    log.info(
        "Mixed train: %d | test: %d (pos=%d/%d, neg=%d/%d).",
        len(train),
        len(test),
        train["to_link"].sum(),
        test["to_link"].sum(),
        (~train["to_link"]).sum(),
        (~test["to_link"]).sum(),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _build_avc_group_split(
    test_size: float = 0.20,
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits AVC at the group level, then expands to pairs.

    Used for Exp 13 (group-level stratification).
    """
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        raise RuntimeError("avc.parquet is empty or missing.")
    decided = avc[avc["to_link"].notna()].reset_index(drop=True)
    train_groups, test_groups = train_test_split(
        decided,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=decided["to_link"],
    )

    def _expand(groups_df: pd.DataFrame) -> pd.DataFrame:
        """Expands group-level rows into all pairwise variant combinations."""
        rows: list[tuple] = []
        for _, row in groups_df.iterrows():
            rows.extend(cluster.expand_pairs(row))
        return pd.DataFrame(rows, columns=["variants", "variant_a", "variant_b", "to_link"])

    train_pairs = _expand(train_groups)
    test_pairs = _expand(test_groups)
    # Filtering to WRatio band
    for df in [train_pairs, test_pairs]:
        df["_wr"] = df.apply(
            lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
            axis=1,
        )
    train_pairs = (
        train_pairs[(train_pairs["_wr"] >= wratio_lower) & (train_pairs["_wr"] < wratio_upper)]
        .drop(columns=["_wr"])
        .reset_index(drop=True)
    )
    test_pairs = (
        test_pairs[(test_pairs["_wr"] >= wratio_lower) & (test_pairs["_wr"] < wratio_upper)]
        .drop(columns=["_wr"])
        .reset_index(drop=True)
    )
    log.info(
        "AVC group split: train=%d (pos=%d), test=%d (pos=%d).",
        len(train_pairs),
        train_pairs["to_link"].sum(),
        len(test_pairs),
        test_pairs["to_link"].sum(),
    )
    return train_pairs, test_pairs


def _dispatch_data_build(
    *,
    data_source: str,
    split_strategy: str,
    test_source: str,
    test_size: float,
    wratio_lower: int,
    wratio_upper: int,
    cluster_cap: int = 0,
    neg_ratio: int = 0,
    neg_matching: str = "none",
    neg_count: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Routes to the correct data-build function based on configuration."""
    if data_source == "mixed":
        return _build_mixed_training_data(
            test_size=test_size,
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    if data_source == "avc":
        if split_strategy == "group":
            return _build_avc_group_split(
                test_size=test_size,
                wratio_lower=wratio_lower,
                wratio_upper=wratio_upper,
            )
        return build_training_data(
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
    # Cross-domain experiments: train on external, test on AVC
    if test_source == "avc-full":
        test_pairs = _build_avc_full_test(
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    else:
        raise RuntimeError(
            f"test_source='{test_source}' with data_source='{data_source}' — "
            "cross-domain experiments require --test-source=avc-full."
        )
    if data_source == "mbdb":
        train_pairs = _build_mbdb_training_data(
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    elif data_source == "mbdb-max":
        train_pairs = _build_mbdb_max_training_data(
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    elif data_source == "dbscan-capped":
        train_pairs = _build_dbscan_capped_training_data(
            cluster_cap=cluster_cap or 30,
            neg_ratio=neg_ratio or 10,
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    elif data_source == "dbscan" and neg_matching == "distribution":
        train_pairs = _build_feature_sep_training_data(
            neg_count=neg_count,
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    elif data_source == "dbscan":
        train_pairs = _build_dbscan_training_data(
            wratio_lower=wratio_lower,
            wratio_upper=wratio_upper,
        )
    else:
        raise RuntimeError(f"Unknown data_source: {data_source}")
    return train_pairs, test_pairs


# ═════════════════════════════════════════════════════════════════════════════
# Feature separation constants (Exp 8)
# ═════════════════════════════════════════════════════════════════════════════
_WHOLE_STRING_FEATURES = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "WRatio",
    "QRatio",
    "norm_levenshtein",
    "jaro_winkler",
    "length_ratio",
    "abs_len_diff",
]
_NON_WS_FEATURES = [
    "token_count_diff",
    "token_jaccard",
    "shared_token_ratio",
    "lcs_token_len",
    "token_order_displacement",
    "bigram_jaccard",
    "trigram_jaccard",
    "edit_inserts",
    "edit_deletes",
    "edit_replaces",
    "shared_prefix_len",
    "shared_suffix_len",
    "script_mismatch",
]
_WS_SIM_SCORES = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "WRatio",
    "QRatio",
]


def _compute_cross_tier_interactions(base_feats: dict[str, float]) -> dict[str, float]:
    """Computes interaction features with at most 1 whole-string factor.

    Two types:
    1. Cross-tier: 1 whole-string score × 1 non-WS feature.
    2. Non-WS only: all pairwise products among non-WS features.
    """
    seen: Counter = Counter()
    interactions: dict[str, float] = {}
    for ws in _WS_SIM_SCORES:
        ws_val = base_feats.get(ws, 0.0)
        for nws in _NON_WS_FEATURES:
            nws_val = base_feats.get(nws, 0.0)
            col = sanitize(f"{ws} * {nws}", seen)
            interactions[col] = ws_val * nws_val
    for a_name, b_name in itertools.combinations(_NON_WS_FEATURES, 2):
        a_val = base_feats.get(a_name, 0.0)
        b_val = base_feats.get(b_name, 0.0)
        col = sanitize(f"{a_name} * {b_name}", seen)
        interactions[col] = a_val * b_val
    return interactions


def _add_separated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base features + cross-tier interactions (Exp 8 design).

    Whole-string base features are computed (needed for interaction
    products) but will be excluded from the final training columns.
    """
    n = len(df)
    log.info("Computing separated features for %d pairs...", n)
    # Computing base features (all 3 tiers)
    base_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        feats = compute_pair_features(str(row["variant_a"]), str(row["variant_b"]))
        base_rows.append(feats)
        if (i + 1) % 200 == 0:
            log.info("  Separated base features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    base_df = pd.DataFrame(base_rows, index=df.index)
    for col in base_df.columns:
        if col not in df.columns:
            df[col] = base_df[col]
    # Computing cross-tier + non-WS interactions
    log.info("Computing cross-tier and non-WS interaction features...")
    interaction_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        feats = compute_pair_features(str(row["variant_a"]), str(row["variant_b"]))
        interaction_rows.append(_compute_cross_tier_interactions(feats))
    interaction_df = pd.DataFrame(interaction_rows, index=df.index)
    for col in interaction_df.columns:
        if col not in df.columns:
            df[col] = interaction_df[col]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Feature dispatch helper
# ═════════════════════════════════════════════════════════════════════════════
def _compute_features_for_split(
    df: pd.DataFrame,
    *,
    features: str,
    feature_strategy: str = "standard",
    catalogue: bool,
    cat_design: str,
    name_to_albums: dict[str, list[str]] | None,
    name_to_tracks: dict[str, list[str]] | None,
    group_features: bool,
) -> pd.DataFrame:
    """Computes the appropriate features based on the feature tier setting."""
    if feature_strategy == "separated":
        df = _add_separated_features(df)
    elif features == "base":
        # Base 23 only — no interaction, no catalogue
        df = _add_base_features_only(df)
    else:
        # Base 23 + interaction 30
        df = _add_base_features(df)
    # Catalogue features (only when features == "full" and catalogue is True)
    if catalogue and name_to_albums is not None and name_to_tracks is not None:
        if cat_design == "proportional":
            df = _add_proportional_catalogue_features(df, name_to_albums, name_to_tracks)
        else:
            df = _add_catalogue_features(df, name_to_albums, name_to_tracks)
    # Group-level features (Exp 5–13)
    if group_features:
        if "variants" not in df.columns:
            df["variants"] = df["variant_a"].astype(str) + "{" + df["variant_b"].astype(str)
        df = pd.concat([df, df["variants"].apply(stats.length_stats)], axis=1)
    return df


def _add_base_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Computes base 23 features without interaction features."""
    n = len(df)
    log.info("Computing base features (no interaction) for %d pairs...", n)
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
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
def _optimal_threshold(y_true, y_prob):
    """Finds the probability threshold that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1s)]), float(f1s[np.argmax(f1s)])


def _eval_at(y_true, y_prob, thr):
    """Computes precision, recall, and F1 at a given threshold."""
    y_pred = (y_prob >= thr).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": thr,
    }


def _high_precision_threshold(y_true, y_prob, min_precision: float = 0.80):
    """Finds the best-F1 threshold where precision ≥ *min_precision*."""
    best = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in np.arange(0.50, 0.99, 0.01):
        m = _eval_at(y_true, y_prob, t)
        if m["precision"] >= min_precision and m["f1"] > best["f1"]:
            best = m
    return best


# ═════════════════════════════════════════════════════════════════════════════
# Cross-validation loop
# ═════════════════════════════════════════════════════════════════════════════
def _cv_evaluate(
    clf,
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    *,
    n_folds: int = 5,
    random_state: int = RANDOM_STATE,
    model_name: str = "",
) -> dict[str, float]:
    """Runs stratified k-fold CV with per-fold MLflow nested runs.

    Returns mean and std metrics across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        # Building a fresh pipeline per fold
        pre = ColumnTransformer(
            [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        pre.set_output(transform="pandas")
        fold_pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_pipeline.fit(X_tr, y_tr)
        y_prob = fold_pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        y_pred = fold_pipeline.predict(X_val)
        metrics = {
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "auc": auc,
        }
        fold_metrics.append(metrics)
        experiment.log_cv_fold(fold_idx, metrics, run_name_prefix=f"{model_name}_fold")
    # Aggregating fold results
    mean_metrics: dict[str, float] = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        mean_metrics[f"cv_mean_{key}"] = float(np.mean(vals))
        mean_metrics[f"cv_std_{key}"] = float(np.std(vals))
    return mean_metrics


# ═════════════════════════════════════════════════════════════════════════════
# GPU-safe fitting wrapper
# ═════════════════════════════════════════════════════════════════════════════
def _fit_with_gpu_fallback(
    pipeline: Pipeline,
    X,
    y,
    device: str,
) -> tuple[Pipeline, str]:
    """Fits the pipeline; retries on CPU if CUDA fails at runtime."""
    try:
        pipeline.fit(X, y)
        return pipeline, device
    except Exception as exc:
        if device != "cuda":
            raise
        log.warning("CUDA training failed (%s) — retrying on CPU.", exc)
        clf = pipeline.named_steps.get("clf")
        if clf is not None and hasattr(clf, "device"):
            clf.set_params(device="cpu")
        pipeline.fit(X, y)
        return pipeline, "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# Next experiment number
# ═════════════════════════════════════════════════════════════════════════════
def _next_experiment_number() -> int:
    """Queries MLflow for the highest logged experiment number and returns n+1."""
    import mlflow

    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment.DEFAULT_EXPERIMENT)
        if exp is None:
            return 16
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="",
            max_results=1000,
        )
        max_num = 15  # baseline from existing experiments
        for run in runs:
            val = run.data.params.get("experiment")
            if val is not None:
                try:
                    max_num = max(max_num, int(val))
                except (ValueError, TypeError):
                    pass
        return max_num + 1
    except Exception:
        return 16


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════
def run_training(
    *,
    run_name: str | None = None,
    n_folds: int = 5,
    test_size: float = 0.20,
    models: list[str] | None = None,
    catalogue: bool = True,
    data_source: str = "avc",
    split_strategy: str = "pair",
    test_source: str = "holdout",
    features: str = "full",
    catalogue_source: str = "unified",
    catalogue_design: str = "proportional",
    group_features: bool = False,
    wratio_lower: int = WRATIO_LOWER,
    wratio_upper: int = WRATIO_UPPER,
    experiment_num: int | None = None,
    include_composites: bool = False,
    cluster_cap: int = 0,
    neg_ratio: int = 0,
    feature_strategy: str = "standard",
    neg_matching: str = "none",
    neg_count: int = 5000,
) -> dict[str, dict[str, float]]:
    """Runs the unified training pipeline: data → features → CV → evaluation.

    Parameters
    ----------
    run_name : optional MLflow parent run name (auto-generated if None).
    n_folds : number of stratified CV folds.
    test_size : fraction held out for final evaluation.
    models : subset of model names to train (default: all 5 tree-based).
    catalogue : whether to include catalogue features.
    data_source : training data origin (avc, mbdb, dbscan, dbscan-capped, mixed).
    split_strategy : how to split (pair, group).
    test_source : test data origin (holdout, avc-full).
    features : feature tiers (base, interaction, full).
    catalogue_source : catalogue data origin (none, scrobble, mbdb, unified).
    catalogue_design : catalogue feature style (proportional, presence).
    group_features : whether to include group-level length_stats.
    wratio_lower : WRatio band lower bound.
    wratio_upper : WRatio band upper bound.
    experiment_num : explicit experiment number for backfill labelling.
    include_composites : whether to include composite models.
    cluster_cap : max cluster size for dbscan-capped (Exp 7).
    neg_ratio : target negative:positive ratio for dbscan-capped (Exp 7).
    feature_strategy : 'standard' or 'separated' (Exp 8).
    neg_matching : 'none' or 'distribution' (Exp 8).
    neg_count : target count for distribution-matched negatives (Exp 8).

    Returns a dict mapping model_name → held-out test metrics.
    """
    # ── Step 0: Pre-verifying MLflow ───────────────────────────────────────
    verify_mlflow()
    exp_num = experiment_num or _next_experiment_number()
    log.info("Training as Experiment %d.", exp_num)
    # ── Resolving effective catalogue flag ──────────────────────────────────
    effective_catalogue = catalogue and features == "full"
    effective_cat_source = catalogue_source if effective_catalogue else "none"
    effective_cat_design = catalogue_design if effective_catalogue else "none"
    # ── Step 1: Building training data ─────────────────────────────────────
    train_pairs, test_pairs = _dispatch_data_build(
        data_source=data_source,
        split_strategy=split_strategy,
        test_source=test_source,
        test_size=test_size,
        wratio_lower=wratio_lower,
        wratio_upper=wratio_upper,
        cluster_cap=cluster_cap,
        neg_ratio=neg_ratio,
        neg_matching=neg_matching,
        neg_count=neg_count,
    )
    # ── Step 2: Loading catalogue lookups (if needed) ──────────────────────
    name_to_albums: dict[str, list[str]] | None = None
    name_to_tracks: dict[str, list[str]] | None = None
    if effective_catalogue:
        if effective_cat_source == "scrobble":
            name_to_albums, name_to_tracks = _load_scrobble_only_lookups()
        else:
            name_to_albums, name_to_tracks = _load_catalogue_lookups()
    # ── Step 3: Computing features ─────────────────────────────────────────
    log.info("Computing features for training set...")
    train_df = _compute_features_for_split(
        train_pairs,
        features=features,
        feature_strategy=feature_strategy,
        catalogue=effective_catalogue,
        cat_design=effective_cat_design,
        name_to_albums=name_to_albums,
        name_to_tracks=name_to_tracks,
        group_features=group_features,
    )
    log.info("Computing features for test set...")
    test_df = _compute_features_for_split(
        test_pairs,
        features=features,
        feature_strategy=feature_strategy,
        catalogue=effective_catalogue,
        cat_design=effective_cat_design,
        name_to_albums=name_to_albums,
        name_to_tracks=name_to_tracks,
        group_features=group_features,
    )
    # ── Step 4: Pruning ────────────────────────────────────────────────────
    target = "to_link"
    exclude = {target, "variant_a", "variant_b", "source", "_key", "variants"}
    # Excluding whole-string features from training when using separated strategy
    if feature_strategy == "separated":
        exclude.update(_WHOLE_STRING_FEATURES)
    all_num = [
        c
        for c in train_df.columns
        if c not in exclude and train_df[c].dtype in ("float64", "int64", "float32", "int32")
    ]
    log.info("Pre-pruning features: %d", len(all_num))
    num_cols = prune_feature_columns(train_df[all_num])
    # Ensuring test_df has all surviving columns
    missing = [c for c in num_cols if c not in test_df.columns]
    if missing:
        log.warning("Columns missing in test_df (filling with 0): %s", missing)
        for c in missing:
            test_df[c] = 0.0
    disco_survived = [c for c in num_cols if c.startswith("disco_")]
    melo_survived = [c for c in num_cols if c.startswith("melo_")]
    log.info(
        "Post-pruning: %d features (%d disco, %d melo survived).",
        len(num_cols),
        len(disco_survived),
        len(melo_survived),
    )
    # ── Step 5: Training and evaluation ────────────────────────────────────
    X_train = train_df[num_cols]
    y_train = train_df[target].astype(int).values
    X_test = test_df[num_cols]
    y_test = test_df[target].astype(int).values
    device = get_device()
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    log.info(
        "Train: %d | Test: %d | Features: %d | spw: %.2f | device: %s",
        len(X_train),
        len(X_test),
        len(num_cols),
        spw,
        device,
    )
    model_catalogue = _build_model_catalogue(spw, device, random_state=RANDOM_STATE)
    if not include_composites:
        model_catalogue = {k: v for k, v in model_catalogue.items() if k in _TREE_MODELS}
    # Defaulting to LightGBM when no explicit model list is provided
    effective_models = models or DEFAULT_MODELS
    model_catalogue = {k: v for k, v in model_catalogue.items() if k in effective_models}
    if not model_catalogue:
        raise RuntimeError("No models selected — check --models argument.")
    parent_name = run_name or f"exp{exp_num}_unified"
    experiment.init_experiment()
    results: dict[str, dict[str, float]] = {}
    with experiment.start_run(run_name=parent_name):
        experiment.log_params(
            {
                "experiment": exp_num,
                "experiment_type": "unified_pipeline",
                "random_state": RANDOM_STATE,
                "test_size": test_size,
                "n_folds": n_folds,
                "wratio_lower": wratio_lower,
                "wratio_upper": wratio_upper,
                "catalogue_features": effective_catalogue,
                "n_features": len(num_cols),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_pos": int(y_train.sum()),
                "train_neg": int((y_train == 0).sum()),
                "test_pos": int(y_test.sum()),
                "test_neg": int((y_test == 0).sum()),
                "spw": round(spw, 2),
                "device_probed": device,
                "model_count": len(model_catalogue),
                "data_source": data_source,
                "split_strategy": split_strategy,
                "test_source": test_source,
                "feature_tiers": features,
                "catalogue_source": effective_cat_source,
                "group_features": "length_stats" if group_features else "none",
                "catalogue_feature_design": effective_cat_design,
                "include_composites": include_composites,
                "feature_strategy": feature_strategy,
                "neg_matching": neg_matching,
                "neg_count": neg_count if neg_matching == "distribution" else 0,
                "cluster_cap": cluster_cap if data_source == "dbscan-capped" else 0,
                "neg_ratio": neg_ratio if data_source == "dbscan-capped" else 0,
            }
        )
        # Logging train/test splits as artefacts
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / f"exp{exp_num}_train.parquet"
            test_path = Path(tmpdir) / f"exp{exp_num}_test.parquet"
            dump_parquet(train_df, train_path)
            dump_parquet(test_df, test_path)
            experiment.log_artifact(train_path)
            experiment.log_artifact(test_path)
        for model_name, clf in model_catalogue.items():
            log.info("─── Training %s ───", model_name)
            with experiment.start_run(run_name=model_name, nested=True):
                import mlflow

                mlflow.set_tag("model_type", model_name)
                safe_params = _safe_get_params(clf)
                safe_params["device_used"] = device
                experiment.log_params(safe_params)
                # Running cross-validation with per-fold tracking
                cv_metrics = _cv_evaluate(
                    clf,
                    X_train,
                    y_train,
                    num_cols,
                    n_folds=n_folds,
                    random_state=RANDOM_STATE,
                    model_name=model_name,
                )
                experiment.log_metrics(cv_metrics)
                # Training final model on full training set
                pre = ColumnTransformer(
                    [("num", Pipeline([("scaler", RobustScaler())]), num_cols)],
                    remainder="drop",
                    verbose_feature_names_out=False,
                )
                pre.set_output(transform="pandas")
                final_pipeline = Pipeline([("prep", pre), ("clf", clone(clf))])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    final_pipeline, actual_device = _fit_with_gpu_fallback(
                        final_pipeline,
                        X_train,
                        y_train,
                        device,
                    )
                if actual_device != device:
                    experiment.log_params({"device_fallback": actual_device})
                # Evaluating on held-out test at 3 operating points
                y_prob = final_pipeline.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                default_m = _eval_at(y_test, y_prob, 0.5)
                opt_thr, _ = _optimal_threshold(y_test, y_prob)
                optimal_m = _eval_at(y_test, y_prob, opt_thr)
                hiprec_m = _high_precision_threshold(y_test, y_prob)
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
                        "hiprec_threshold": hiprec_m["threshold"],
                        "hiprec_f1": hiprec_m["f1"],
                        "hiprec_precision": hiprec_m["precision"],
                        "hiprec_recall": hiprec_m["recall"],
                    }
                )
                results[model_name] = {
                    "auc": auc,
                    "default_f1": default_m["f1"],
                    "default_prec": default_m["precision"],
                    "default_rec": default_m["recall"],
                    "opt_thr": optimal_m["threshold"],
                    "opt_f1": optimal_m["f1"],
                    "opt_prec": optimal_m["precision"],
                    "opt_rec": optimal_m["recall"],
                    "hiprec_thr": hiprec_m["threshold"],
                    "hiprec_f1": hiprec_m["f1"],
                    "hiprec_prec": hiprec_m["precision"],
                    "hiprec_rec": hiprec_m["recall"],
                    **cv_metrics,
                }
                log.info(
                    "%s → AUC=%.4f | def F1=%.4f | opt F1=%.4f (thr=%.3f) | hi-P F1=%.4f (thr=%.3f)",
                    model_name,
                    auc,
                    default_m["f1"],
                    optimal_m["f1"],
                    opt_thr,
                    hiprec_m["f1"],
                    hiprec_m["threshold"],
                )
                # Printing classification report at optimal threshold
                y_pred_opt = (y_prob >= opt_thr).astype(int)
                print(f"\n=== {model_name} (optimal thr={opt_thr:.3f}) ===")
                print(
                    classification_report(
                        y_test,
                        y_pred_opt,
                        target_names=["no link", "link"],
                    )
                )
                # Logging artefacts
                experiment.log_confusion_matrix(y_test, y_pred_opt)
                experiment.log_feature_importance(final_pipeline, num_cols)
                X_test_transformed = final_pipeline.named_steps["prep"].transform(X_test)
                experiment.log_shap_summary(
                    final_pipeline,
                    X_test_transformed,
                    num_cols,
                )
                experiment.log_model(final_pipeline)
    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print(
        f"{'Model':<22} {'AUC':>6} | {'Def P':>6} {'Def R':>6} {'Def F1':>6} | "
        f"{'Opt thr':>7} {'Opt P':>6} {'Opt R':>6} {'Opt F1':>6} | "
        f"{'HiP thr':>7} {'HiP P':>6} {'HiP R':>6} {'HiP F1':>6}"
    )
    print("-" * 130)
    for name in sorted(results, key=lambda k: results[k]["opt_f1"], reverse=True):
        r = results[name]
        print(
            f"{name:<22} {r['auc']:>6.4f} | "
            f"{r['default_prec']:>6.4f} {r['default_rec']:>6.4f} "
            f"{r['default_f1']:>6.4f} | "
            f"{r['opt_thr']:>7.3f} {r['opt_prec']:>6.4f} "
            f"{r['opt_rec']:>6.4f} {r['opt_f1']:>6.4f} | "
            f"{r['hiprec_thr']:>7.3f} {r['hiprec_prec']:>6.4f} "
            f"{r['hiprec_rec']:>6.4f} {r['hiprec_f1']:>6.4f}"
        )
    print("=" * 130)
    # Selecting best model by c9r composite score (0.4×HiP_P + 0.3×HiP_F1 + 0.3×AUC)
    best = max(
        results, key=lambda k: 0.4 * results[k]["hiprec_prec"] + 0.3 * results[k]["hiprec_f1"] + 0.3 * results[k]["auc"]
    )
    score = 0.4 * results[best]["hiprec_prec"] + 0.3 * results[best]["hiprec_f1"] + 0.3 * results[best]["auc"]
    print(f"\nBest model by c9r score: {best} (score={score:.4f})")
    return results
