"""
Provides business logic for the ``c9r canon`` command group.

Functions
---------
avc_summary       – returns the current avc table for display.
propagate_avc     – applies canonisation results to artist_info.
undecided_rows    – returns avc rows with to_link IS NULL.
list_mlflow_runs  – queries MLflow for trained model runs.
load_run_model    – loads a sklearn Pipeline from an MLflow run.
discover_candidates – scans scrobble data for new variant candidates.
"""

from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pandas as pd
from helpers.io import (
    ALIAS_SEP,
    AVC_PQ,
    PQ_DIR,
    ARTIST_INFO_PQ,
    read_parquet,
    dump_parquet,
    append_to_parquet,
    scrobble_data_exists,
    scrobble_duckdb_from,
)

log = logging.getLogger(__name__)
SEPARATOR = "{"


def _pq(path: Path) -> str:
    """Returns a quoted Parquet path string safe for SQL embedding."""
    return f"'{path.as_posix()}'"


# ── (i) avc show ──────────────────────────────────────────────────────────────
def avc_summary(
    *,
    decided_only: bool = False,
    undecided_only: bool = False,
    last_n: int | None = None,
) -> list[dict]:
    """
    Returns the current avc table rows for CLI display.

    Each dict contains: idx, to_link_display, canonical_name, stamp, artist_variants_text.
    """
    if not AVC_PQ.exists():
        return []
    con = duckdb.connect()
    try:
        where_parts: list[str] = []
        if decided_only:
            where_parts.append("to_link IS NOT NULL")
        if undecided_only:
            where_parts.append("to_link IS NULL")
        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        limit = f"LIMIT {last_n}" if last_n else ""
        df = con.execute(f"""
            SELECT to_link, canonical_name, stamp, artist_variants_text
            FROM {_pq(AVC_PQ)}
            {where}
            ORDER BY stamp DESC
            {limit}
        """).df()
    finally:
        con.close()
    rows: list[dict] = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        tl = r["to_link"]
        if tl is None or pd.isna(tl):
            link_display = "?"
        elif tl:
            link_display = "✓"
        else:
            link_display = "✗"
        stamp_str = str(r["stamp"])[:10] if r["stamp"] is not None else ""
        rows.append(
            {
                "idx": i,
                "to_link_display": link_display,
                "canonical_name": r["canonical_name"] or "",
                "stamp": stamp_str,
                "artist_variants_text": r["artist_variants_text"] or "",
            }
        )
    return rows


# ── (ii) avc propagate ───────────────────────────────────────────────────────
def propagate_avc() -> dict:
    """
    Applies canonisation results from avc.parquet to artist_info.parquet.

    For each avc row where to_link is True:
    - Renames variant artist_info rows to the canonical_name.
    - Appends non-canonical variants to the canonical artist's aliases field.

    Returns a summary dict with keys 'updated' and 'aliases_added'.
    """
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        return {"updated": 0, "aliases_added": 0}
    ai = read_parquet(ARTIST_INFO_PQ)
    if ai is None or ai.empty:
        return {"updated": 0, "aliases_added": 0}
    # Filtering to decided-link rows
    linked = avc[avc["to_link"] == True]  # noqa: E712
    if linked.empty:
        return {"updated": 0, "aliases_added": 0}
    updated = 0
    aliases_added = 0
    ai_names = set(ai["artist_name"].tolist())
    for _, row in linked.iterrows():
        canonical = row["canonical_name"]
        if not canonical or canonical == "__SKIP__":
            continue
        variants = [v.strip() for v in str(row["artist_variants_text"]).split(SEPARATOR) if v.strip()]
        non_canonical = [v for v in variants if v != canonical]
        if not non_canonical:
            continue
        # Ensuring canonical row exists in artist_info
        if canonical not in ai_names:
            # Checking if any variant has a row we can rename
            for v in non_canonical:
                if v in ai_names:
                    ai.loc[ai["artist_name"] == v, "artist_name"] = canonical
                    ai_names.discard(v)
                    ai_names.add(canonical)
                    updated += 1
                    break
        # Building current alias set for the canonical artist
        canon_mask = ai["artist_name"] == canonical
        if not canon_mask.any():
            continue
        canon_idx = canon_mask.idxmax()
        existing_aliases_raw = ai.at[canon_idx, "aliases"] or ""
        existing_aliases = {
            a.strip() for a in str(existing_aliases_raw).split(ALIAS_SEP) if a.strip() and a.strip() != "None"
        }
        new_aliases = set()
        for v in non_canonical:
            if v not in existing_aliases and v != canonical:
                new_aliases.add(v)
            # Renaming variant rows to canonical (merge)
            if v in ai_names and v != canonical:
                ai = ai[ai["artist_name"] != v]
                ai_names.discard(v)
                updated += 1
        if new_aliases:
            merged = existing_aliases | new_aliases
            ai.at[canon_idx, "aliases"] = ALIAS_SEP.join(sorted(merged))
            aliases_added += len(new_aliases)
    # Deduplicating on artist_name (keep first)
    ai = ai.drop_duplicates(subset=["artist_name"], keep="first")
    dump_parquet(ai, ARTIST_INFO_PQ)
    log.info("Propagated AVC: %d rows updated, %d aliases added.", updated, aliases_added)
    return {"updated": updated, "aliases_added": aliases_added}


# ── (iii) canon human ─────────────────────────────────────────────────────────
def undecided_rows() -> pd.DataFrame:
    """
    Returns avc rows where to_link IS NULL, sorted oldest first.

    These are candidate groups awaiting human review.
    """
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        return pd.DataFrame()
    mask = avc["to_link"].isna()
    return avc[mask].sort_values("stamp").reset_index(drop=True)


def update_avc_decision(
    variant_hash: str,
    to_link: bool,
    canonical_name: str,
    comment: str = "",
) -> None:
    """
    Updates a single avc row with the human decision.

    Reads the full table, updates the matching row in-place, and rewrites.
    """
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        log.warning("avc.parquet is empty, nothing to update.")
        return
    mask = avc["artist_variants_hash"] == variant_hash
    if not mask.any():
        log.warning("Hash %s not found in avc.parquet.", variant_hash)
        return
    idx = mask.idxmax()
    avc.at[idx, "to_link"] = to_link
    avc.at[idx, "canonical_name"] = canonical_name
    avc.at[idx, "comment"] = comment
    avc.at[idx, "stamp"] = pd.Timestamp(datetime.now(timezone.utc))
    # Enforcing dtype after in-place update
    avc["to_link"] = avc["to_link"].astype("boolean")
    dump_parquet(avc, AVC_PQ)


# ── (iv) canon machine ───────────────────────────────────────────────────────
def list_mlflow_runs(experiment_name: str = "c9r-record-linkage") -> list[dict]:
    """
    Queries the MLflow tracking store for finished runs.

    Returns a list of dicts with keys: run_id, run_name, start_time,
    precision, recall, f1, auc.
    """
    import mlflow
    from helpers.experiment import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return []
    result: list[dict] = []
    for _, r in runs.iterrows():
        # Skipping nested fold runs
        if str(r.get("tags.mlflow.runName", "")).startswith("fold"):
            continue
        result.append(
            {
                "run_id": r["run_id"],
                "run_name": r.get("tags.mlflow.runName", ""),
                "start_time": str(r["start_time"])[:19],
                "precision": round(r.get("metrics.precision", 0), 4),
                "recall": round(r.get("metrics.recall", 0), 4),
                "f1": round(r.get("metrics.f1", 0), 4),
                "auc": round(r.get("metrics.auc", 0), 4),
            }
        )
    return result


def load_run_model(run_id: str):
    """Loads the sklearn Pipeline logged in the given MLflow run."""
    import mlflow
    from helpers.experiment import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


def _build_exclusion_set() -> set[str]:
    """
    Builds the set of artist names already covered by avc or artist_info.

    Includes:
    - Every name inside any artist_variants_text (split on '{')
    - Every canonical_name in avc
    - Every artist_name and alias in artist_info
    """
    covered: set[str] = set()
    # Gathering from avc.parquet
    avc = read_parquet(AVC_PQ)
    if avc is not None and not avc.empty:
        for _, row in avc.iterrows():
            text = str(row.get("artist_variants_text", ""))
            for v in text.split(SEPARATOR):
                v = v.strip()
                if v:
                    covered.add(v)
            cn = row.get("canonical_name", "")
            if cn and cn != "__SKIP__":
                covered.add(cn)
    # Gathering from artist_info.parquet
    ai = read_parquet(ARTIST_INFO_PQ)
    if ai is not None and not ai.empty:
        for _, row in ai.iterrows():
            name = row.get("artist_name", "")
            if name:
                covered.add(name)
            aliases_raw = row.get("aliases", "")
            if aliases_raw and str(aliases_raw) not in ("None", "nan", ""):
                for a in str(aliases_raw).split(ALIAS_SEP):
                    a = a.strip()
                    if a:
                        covered.add(a)
    return covered


def _make_signature(variants: list[str]) -> str:
    """Canonical, DB-compatible signature string."""
    return SEPARATOR.join(sorted(v.strip() for v in variants if v.strip()))


def _make_hash(signature: str) -> str:
    """Returns sha256 hex digest of the signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


PREDICTIONS_LOG_PQ = PQ_DIR / "predictions_log.parquet"


def _log_predictions(prediction_rows: list[dict]) -> None:
    """Appends prediction records to predictions_log.parquet for drift detection."""
    if not prediction_rows:
        return
    df = pd.DataFrame(prediction_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    append_to_parquet(df, PREDICTIONS_LOG_PQ)
    log.info("Logged %d predictions to %s.", len(df), PREDICTIONS_LOG_PQ.name)


def discover_candidates(
    model=None,
    *,
    wratio_cutoff: int = 75,
    proba_threshold: float = 0.5,
    min_plays: int = 2,
    limit: int = 2000,
) -> list[dict]:
    """
    Scans scrobble.parquet for new artist name variant candidates.

    Uses RapidFuzz for fast pre-filtering and the full 41-feature pipeline
    for classification.  Loads ``ML/lightgbm_best.pkl`` when *model* is
    None.  Logs every prediction to ``PQ/predictions_log.parquet``.
    Returns a list of dicts with keys: signature, variants, hash, max_prob.
    """
    from rapidfuzz import fuzz, process
    from helpers.inference import compute_inference_features, load_model

    if model is None:
        model = load_model()
    if not scrobble_data_exists():
        return []
    # Getting unique artist names with play counts
    con = duckdb.connect()
    try:
        artists_df = con.execute(f"""
            SELECT artist_name, COUNT(*) AS plays
            FROM {scrobble_duckdb_from()}
            WHERE artist_name IS NOT NULL AND artist_name != ''
            GROUP BY artist_name
            HAVING plays >= {min_plays}
            ORDER BY plays DESC
            LIMIT {limit}
        """).df()
    finally:
        con.close()
    if artists_df.empty:
        return []
    exclusion = _build_exclusion_set()
    # Filtering to names not yet fully covered
    candidates = [n for n in artists_df["artist_name"].tolist() if n not in exclusion]
    if not candidates:
        log.info("All artist names already covered by avc/artist_info.")
        return []
    log.info("Scanning %d uncovered artist names for variant candidates.", len(candidates))
    # Pre-filtering with RapidFuzz, then classifying with the full feature pipeline
    all_names = artists_df["artist_name"].tolist()
    pairs_seen: set[frozenset] = set()
    positive_pairs: list[tuple[str, str, float]] = []
    prediction_log: list[dict] = []
    now_ts = datetime.now(timezone.utc)
    n_scored = 0
    for name in candidates:
        matches = process.extract(
            name,
            all_names,
            scorer=fuzz.WRatio,
            score_cutoff=wratio_cutoff,
            limit=10,
        )
        for match_name, _score, _ in matches:
            if match_name == name:
                continue
            pair = frozenset((name, match_name))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)
            # Computing full feature vector and classifying
            try:
                feats = compute_inference_features(name, match_name)
                vec = pd.DataFrame([feats])
                prob = float(model.predict_proba(vec)[0, 1])
                n_scored += 1
                # Logging every prediction for drift detection
                prediction_log.append(
                    {
                        "timestamp": now_ts,
                        "variant_a": name,
                        "variant_b": match_name,
                        "probability": prob,
                        "features_json": json.dumps(feats, allow_nan=False),
                    }
                )
                if prob >= proba_threshold:
                    positive_pairs.append((name, match_name, prob))
            except Exception:
                log.debug("Skipping pair (%s, %s) — model prediction failed.", name, match_name)
    log.info("Scored %d pairs, %d above threshold %.2f.", n_scored, len(positive_pairs), proba_threshold)
    # Persisting prediction log
    _log_predictions(prediction_log)
    if not positive_pairs:
        log.info("No new variant candidates found.")
        return []
    # Building pair→probability lookup for group-level max
    pair_probs: dict[frozenset, float] = {}
    for a, b, prob in positive_pairs:
        pair_probs[frozenset((a, b))] = prob
    # Grouping pairwise links via union-find
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        """Finds the root of x in the union-find structure."""
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        """Merges the sets containing a and b."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in positive_pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)
    # Collecting groups
    groups: dict[str, list[str]] = {}
    for name in parent:
        root = find(name)
        groups.setdefault(root, []).append(name)
    # Filtering to multi-member groups and checking against existing avc hashes
    existing_hashes: set[str] = set()
    avc = read_parquet(AVC_PQ)
    if avc is not None and not avc.empty:
        existing_hashes = set(avc["artist_variants_hash"].tolist())
    new_candidates: list[dict] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        sig = _make_signature(members)
        h = _make_hash(sig)
        if h in existing_hashes:
            continue
        # Computing the maximum pairwise probability for this group
        max_prob = 0.0
        from itertools import combinations

        for ma, mb in combinations(sorted(members), 2):
            p = pair_probs.get(frozenset((ma, mb)), 0.0)
            max_prob = max(max_prob, p)
        new_candidates.append(
            {
                "signature": sig,
                "variants": sorted(members),
                "hash": h,
                "max_prob": round(max_prob, 4),
            }
        )
    # Sorting by descending probability
    new_candidates.sort(key=lambda c: c["max_prob"], reverse=True)
    return new_candidates


def write_new_candidates(candidates: list[dict]) -> int:
    """
    Appends newly discovered candidate groups to avc.parquet with to_link=NULL.

    Stores the model probability in the comment field for human review.
    Returns the number of rows written.
    """
    if not candidates:
        return 0
    rows = []
    now = pd.Timestamp(datetime.now(timezone.utc))
    for c in candidates:
        prob_comment = f"p={c['max_prob']:.4f}" if "max_prob" in c else ""
        rows.append(
            {
                "artist_variants_hash": c["hash"],
                "artist_variants_text": c["signature"],
                "canonical_name": "",
                "to_link": pd.NA,
                "comment": prob_comment,
                "stamp": now,
            }
        )
    df = pd.DataFrame(rows)
    df["to_link"] = df["to_link"].astype("boolean")
    df["stamp"] = pd.to_datetime(df["stamp"], utc=True)
    append_to_parquet(df, AVC_PQ, dedup_cols=["artist_variants_hash"])
    log.info("Wrote %d new candidate groups to avc.parquet.", len(df))
    return len(df)
