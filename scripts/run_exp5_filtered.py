"""
Runs Experiment 5: WRatio band-filtered holdout experiment.

Filters both MBDB training data and AVC test data to WRatio ∈ [60, 100),
then runs the full model catalogue via run_holdout_experiment().
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

# Ensuring project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.io import AVC_PQ, GS_MB_PQ, PQ_DIR, read_parquet
from helpers.features import compute_pair_features
from helpers import cluster, stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

# Operating range: WRatio ∈ [LOWER, UPPER)
WRATIO_LOWER = 60
WRATIO_UPPER = 100


def _compute_wratio(row: pd.Series) -> float:
    """Computes raw WRatio (0–100 scale) for a variant pair."""
    a = str(row["variant_a"]) if pd.notna(row["variant_a"]) else ""
    b = str(row["variant_b"]) if pd.notna(row["variant_b"]) else ""
    return fuzz.WRatio(a, b)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes three-tier pair features and length stats on a pairs DataFrame."""
    feat_df = df.apply(
        lambda r: pd.Series(compute_pair_features(str(r["variant_a"]), str(r["variant_b"]))),
        axis=1,
    )
    for col in feat_df.columns:
        if col not in df.columns:
            df[col] = feat_df[col]
    # Synthesising variants column for length_stats
    if "variants" not in df.columns:
        df["variants"] = df["variant_a"].astype(str) + "{" + df["variant_b"].astype(str)
    df = pd.concat([df, df["variants"].apply(stats.length_stats)], axis=1)
    return df


def build_filtered_train() -> pd.DataFrame:
    """Loads combined MBDB training data and filters to the operating range."""
    # Loading both gs_mb files
    gs_mb = read_parquet(GS_MB_PQ)
    gs_mb_backup = read_parquet(PQ_DIR / "gs_mb_backup.parquet")
    frames = [df for df in [gs_mb, gs_mb_backup] if df is not None and not df.empty]
    if not frames:
        raise FileNotFoundError("No gs_mb parquet files found.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["variant_a", "variant_b"])
    # Deduplicating (order-insensitive pair keys)
    combined["_key"] = combined.apply(
        lambda r: tuple(sorted([str(r["variant_a"]), str(r["variant_b"])])), axis=1,
    )
    combined = combined.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    log.info("Combined MBDB: %d pairs (before filter).", len(combined))
    # Computing WRatio and filtering
    combined["_wratio_raw"] = combined.apply(_compute_wratio, axis=1)
    mask = (combined["_wratio_raw"] >= WRATIO_LOWER) & (combined["_wratio_raw"] < WRATIO_UPPER)
    filtered = combined[mask].copy().drop(columns=["_wratio_raw"])
    pos = filtered["to_link"].sum()
    neg = len(filtered) - pos
    log.info(
        "Filtered MBDB: %d pairs (pos=%d, neg=%d) in WRatio [%d, %d).",
        len(filtered), pos, neg, WRATIO_LOWER, WRATIO_UPPER,
    )
    return filtered.reset_index(drop=True)


def build_filtered_test() -> pd.DataFrame:
    """Loads AVC, expands pairs, and filters to the operating range."""
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        raise FileNotFoundError("avc.parquet not found or empty.")
    # Keeping only decided rows
    decided = avc[avc["to_link"].notna()].copy()
    log.info("AVC decided rows: %d", len(decided))
    # Expanding into pairwise rows
    rows = []
    for _, row in decided.iterrows():
        rows.extend(cluster.expand_pairs(row))
    test_df = pd.DataFrame(rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    log.info("AVC expanded: %d pairs.", len(test_df))
    # Computing WRatio and filtering
    test_df["_wratio_raw"] = test_df.apply(_compute_wratio, axis=1)
    mask = (test_df["_wratio_raw"] >= WRATIO_LOWER) & (test_df["_wratio_raw"] < WRATIO_UPPER)
    filtered = test_df[mask].copy().drop(columns=["_wratio_raw"])
    pos = filtered["to_link"].sum()
    neg = len(filtered) - pos
    log.info(
        "Filtered AVC: %d pairs (pos=%d, neg=%d) in WRatio [%d, %d).",
        len(filtered), pos, neg, WRATIO_LOWER, WRATIO_UPPER,
    )
    return filtered.reset_index(drop=True)


def main():
    """Runs the band-filtered holdout experiment."""
    log.info("=== Experiment 5: WRatio band-filtered holdout ===")
    log.info("Operating range: WRatio ∈ [%d, %d)", WRATIO_LOWER, WRATIO_UPPER)
    # Building filtered datasets
    train_raw = build_filtered_train()
    test_raw = build_filtered_test()
    # Computing features
    log.info("Computing features for %d training pairs...", len(train_raw))
    train_df = _add_features(train_raw)
    log.info("Computing features for %d test pairs...", len(test_raw))
    test_df = _add_features(test_raw)
    # Running holdout experiment
    from corefunc.canon.experiment_runner import run_holdout_experiment
    results = run_holdout_experiment(
        train_df=train_df,
        test_df=test_df,
        n_folds=10,
        random_state=47,
        run_name="exp5_wratio_band_filtered",
    )
    return results


if __name__ == "__main__":
    main()
