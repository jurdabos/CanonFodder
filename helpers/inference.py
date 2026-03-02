"""
Provides single-pair feature engineering for model inference.

Computes the same feature set used at training time (base 23 +
interaction 30 + catalogue 10 = 63 raw features) so any pruned-model
pickle can select the subset it needs via ``pipeline.feature_names_in_``.

The single entry point is ``compute_inference_features(a, b)``.
A convenience ``load_model()`` returns the persisted sklearn Pipeline.
"""

from __future__ import annotations
import itertools
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from rapidfuzz import fuzz, process
from helpers.features import compute_pair_features
from helpers.io import PQ_DIR, read_parquet, read_scrobble_df, sanitize

log = logging.getLogger(__name__)

# ── Constants (must match Exp 14/15 training recipe) ──────────────────────────
FUZZY_ALBUM_THR = 80
FUZZY_TRACK_THR = 80
MAX_ALBUMS = 1_000
MAX_TRACKS = 10_000
SOLO_DISCO_PQ = PQ_DIR / "mbdb_discography_solo.parquet"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ML" / "lightgbm_best.pkl"
_SIM_SCORES = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "WRatio",
    "QRatio",
]

# ── Session-level catalogue cache ─────────────────────────────────────────────
_catalogue_cache: dict[str, Any] | None = None


def _parse_delimited(s: str) -> list[str]:
    """Splits a ``{``-delimited string into a list, filtering blanks."""
    if not s or pd.isna(s):
        return []
    return [x for x in s.split("{") if x.strip()]


def _load_catalogue_cache() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Builds and caches name→albums and name→tracks dicts.

    Uses MBDB solo-credit discography where available,
    scrobble-side data as fallback.
    """
    global _catalogue_cache
    if _catalogue_cache is not None:
        return _catalogue_cache["albums"], _catalogue_cache["tracks"]
    # Loading MBDB solo-credit disco cache
    mbid_to_disco: dict[str, tuple[list[str], list[str]]] = {}
    if SOLO_DISCO_PQ.exists():
        disco_df = read_parquet(SOLO_DISCO_PQ)
        if disco_df is not None:
            for _, row in disco_df.iterrows():
                mbid_to_disco[row["mbid"]] = (
                    _parse_delimited(row["albums_str"]),
                    _parse_delimited(row["tracks_str"]),
                )
            log.info("Solo-credit disco lookup: %d MBIDs.", len(mbid_to_disco))
    # Loading scrobble data
    scrobbles = read_scrobble_df()
    if scrobbles is None or scrobbles.empty:
        log.warning("No scrobble data — catalogue features will be zeros.")
        _catalogue_cache = {"albums": {}, "tracks": {}}
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
    # Building unified lookups
    all_names = set(scrobbles["artist_name"].unique())
    name_to_albums: dict[str, list[str]] = {}
    name_to_tracks: dict[str, list[str]] = {}
    for name in all_names:
        mbid = name_to_mbid.get(name)
        if mbid and mbid in mbid_to_disco:
            albums, tracks = mbid_to_disco[mbid]
        else:
            albums = scrobble_albums.get(name, [])
            tracks = scrobble_tracks.get(name, [])
        name_to_albums[name] = albums
        name_to_tracks[name] = tracks
    _catalogue_cache = {"albums": name_to_albums, "tracks": name_to_tracks}
    log.info(
        "Catalogue cache loaded: %d album entries, %d track entries.",
        len(name_to_albums),
        len(name_to_tracks),
    )
    return name_to_albums, name_to_tracks


def invalidate_catalogue_cache() -> None:
    """Clears the cached catalogue lookups (e.g. after new scrobble ingestion)."""
    global _catalogue_cache
    _catalogue_cache = None


# ── Catalogue feature helpers (Exp 14 recipe) ────────────────────────────────
def _jaccard(a: set, b: set) -> float:
    """Returns Jaccard index, 0.0 when both sets are empty."""
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


def _compute_disco_features(albums_a: list[str], albums_b: list[str]) -> dict[str, float]:
    """Computes 5 discography (album) overlap features for a single pair."""
    albums_a = albums_a[:MAX_ALBUMS]
    albums_b = albums_b[:MAX_ALBUMS]
    set_aa, set_ab = set(albums_a), set(albums_b)
    n_fuzzy, fuzzy_ratio = _fuzzy_overlap(albums_a, albums_b, FUZZY_ALBUM_THR)
    return {
        "disco_fuzzy_album_ratio": fuzzy_ratio,
        "disco_has_fuzzy_album_match": float(n_fuzzy > 0),
        "disco_n_fuzzy_album_matches": n_fuzzy,
        "disco_exact_album_jaccard": _jaccard(set_aa, set_ab),
        "disco_min_album_count": min(len(set_aa), len(set_ab)),
    }


def _compute_melo_features(tracks_a: list[str], tracks_b: list[str]) -> dict[str, float]:
    """Computes 5 melography (track) overlap features for a single pair."""
    tracks_a = tracks_a[:MAX_TRACKS]
    tracks_b = tracks_b[:MAX_TRACKS]
    set_ta, set_tb = set(tracks_a), set(tracks_b)
    n_fuzzy, fuzzy_ratio = _fuzzy_overlap(tracks_a, tracks_b, FUZZY_TRACK_THR)
    return {
        "melo_fuzzy_track_ratio": fuzzy_ratio,
        "melo_has_fuzzy_track_match": float(n_fuzzy > 0),
        "melo_n_fuzzy_track_matches": n_fuzzy,
        "melo_exact_track_jaccard": _jaccard(set_ta, set_tb),
        "melo_min_track_count": min(len(set_ta), len(set_tb)),
    }


def _compute_catalogue_features(name_a: str, name_b: str) -> dict[str, float]:
    """Computes all 10 catalogue features (disco + melo) for a pair."""
    name_to_albums, name_to_tracks = _load_catalogue_cache()
    albums_a = name_to_albums.get(name_a, [])
    albums_b = name_to_albums.get(name_b, [])
    tracks_a = name_to_tracks.get(name_a, [])
    tracks_b = name_to_tracks.get(name_b, [])
    feats: dict[str, float] = {}
    feats.update(_compute_disco_features(albums_a, albums_b))
    feats.update(_compute_melo_features(tracks_a, tracks_b))
    return feats


# ── Interaction features ──────────────────────────────────────────────────────
def _compute_interaction_features(base: dict[str, float]) -> dict[str, float]:
    """Computes pairwise diffs and products among the 6 similarity scores."""
    seen: Counter = Counter()
    result: dict[str, float] = {}
    score_names = [s for s in _SIM_SCORES if s in base]
    for i, j in itertools.combinations(range(len(score_names)), 2):
        a_name, b_name = score_names[i], score_names[j]
        diff_col = sanitize(f"{a_name} - {b_name}", seen)
        result[diff_col] = base[a_name] - base[b_name]
        prod_col = sanitize(f"{a_name} * {b_name}", seen)
        result[prod_col] = base[a_name] * base[b_name]
    return result


# ── Public entry points ───────────────────────────────────────────────────────
def compute_inference_features(a: str, b: str) -> dict[str, float]:
    """Computes all features for a single (name_a, name_b) pair.

    Returns a flat dict with ~63 features (base 23 + interaction 30 +
    catalogue 10).  The model pipeline selects the subset it needs.
    """
    a = a or ""
    b = b or ""
    base = compute_pair_features(a, b)
    interaction = _compute_interaction_features(base)
    catalogue = _compute_catalogue_features(a, b)
    feats: dict[str, float] = {}
    feats.update(base)
    feats.update(interaction)
    feats.update(catalogue)
    return feats


def load_model(path: Path | None = None):
    """Loads the persisted sklearn Pipeline from a pickle file.

    Returns the fitted Pipeline.  Raises FileNotFoundError when the
    pickle does not exist.
    """
    model_path = path or MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {model_path}")
    with open(model_path, "rb") as fh:
        pipeline = pickle.load(fh)  # noqa: S301
    log.info("Model loaded from %s (%d features).", model_path, len(pipeline.feature_names_in_))
    return pipeline
