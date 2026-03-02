"""
Extracts artist name variant training pairs from the local MusicBrainz mirror.

Produces a supplementary gold standard (gs_mb.parquet) with positive pairs
from MB aliases and negative pairs from same-name / fuzzy-similar artists.
"""

from __future__ import annotations
import logging
import pandas as pd
from rapidfuzz import fuzz, process
from corefunc.mb_local import _psql_csv, check_local_mb
from helpers.io import GS_MB_PQ, dump_parquet

log = logging.getLogger(__name__)
# Alias types to include (1 = Artist name, 3 = Search hint; excludes 2 = Legal name)
_ALIAS_TYPES = "(1, 3)"


def _exact_neg_cap(neg_limit: int) -> int:
    """Scales same-name negative cap proportionally to the requested limit."""
    return max(200, min(neg_limit // 15, 1000))


def _name_pool_size(neg_limit: int) -> int:
    """Scales the fuzzy-negative name pool to at least 2x the requested limit."""
    return max(10_000, neg_limit * 2)


def extract_positive_pairs(limit: int = 5000) -> pd.DataFrame:
    """
    Queries the local MBDB for alias→canonical positive pairs.

    Excludes Legal name aliases (type 2) which map real names to stage names
    and would teach the model to link genuinely different-looking strings.
    When limit > 10 000, supplements with cross-alias pairs (alias_a ↔ alias_b
    of the same artist) to improve training diversity.
    Returns a DataFrame with variant_a, variant_b, to_link, source columns.
    """
    # ── Primary source: alias → canonical ─────────────────────────────────
    sql = f"""\
SELECT a.name  AS canonical,
       aa.name AS alias
FROM musicbrainz.artist a
JOIN musicbrainz.artist_alias aa ON aa.artist = a.id
WHERE a.name != aa.name
  AND (aa.type IS NULL OR aa.type IN {_ALIAS_TYPES})
ORDER BY RANDOM()
LIMIT {limit}"""
    df = _psql_csv(sql)
    frames: list[pd.DataFrame] = []
    if df.empty:
        log.warning("No primary positive pairs returned from MBDB.")
    else:
        df = df.rename(columns={"alias": "variant_a", "canonical": "variant_b"})
        df["to_link"] = True
        df["source"] = "mb_alias"
        frames.append(df[["variant_a", "variant_b", "to_link", "source"]])
        log.info("Primary: %d alias→canonical positive pairs.", len(df))
    # ── Secondary source: cross-alias pairs (alias_a ↔ alias_b) ──────────
    primary_count = len(frames[0]) if frames else 0
    cross_budget = max(0, limit - primary_count)
    if cross_budget >= 500:
        cross_limit = min(cross_budget, limit // 3)
        sql_cross = f"""\
SELECT aa1.name AS alias_a,
       aa2.name AS alias_b
FROM musicbrainz.artist_alias aa1
JOIN musicbrainz.artist_alias aa2
     ON aa1.artist = aa2.artist AND aa1.id < aa2.id
WHERE aa1.name != aa2.name
  AND (aa1.type IS NULL OR aa1.type IN {_ALIAS_TYPES})
  AND (aa2.type IS NULL OR aa2.type IN {_ALIAS_TYPES})
ORDER BY RANDOM()
LIMIT {cross_limit}"""
        cross_df = _psql_csv(sql_cross)
        if not cross_df.empty:
            cross_df = cross_df.rename(columns={"alias_a": "variant_a", "alias_b": "variant_b"})
            cross_df["to_link"] = True
            cross_df["source"] = "mb_cross_alias"
            frames.append(cross_df[["variant_a", "variant_b", "to_link", "source"]])
            log.info("Cross-alias: %d positive pairs.", len(cross_df))
    if not frames:
        return pd.DataFrame(columns=["variant_a", "variant_b", "to_link", "source"])
    combined = pd.concat(frames, ignore_index=True)
    # Dropping rows with missing variant names
    combined = combined.dropna(subset=["variant_a", "variant_b"])
    # Deduplicating pairs (order-insensitive)
    combined["_key"] = combined.apply(
        lambda r: tuple(sorted([str(r["variant_a"]), str(r["variant_b"])])),
        axis=1,
    )
    combined = combined.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
    log.info("Extracted %d total positive pairs from MBDB.", len(combined))
    return combined.head(limit).reset_index(drop=True)


def extract_negative_pairs(
    limit: int = 5000,
    similarity_floor: int = 60,
) -> pd.DataFrame:
    """
    Generates negative (to_link=False) pairs from MBDB in two phases.

    Phase A — same-name negatives: pairs of different artists sharing an
    identical name, capped adaptively based on the requested limit.
    Phase B — hard negatives: pulls a pool of distinct artist names and
    uses RapidFuzz to find cross-artist pairs with WRatio >= similarity_floor.
    Returns a DataFrame with variant_a, variant_b, to_link, source columns.
    """
    frames: list[pd.DataFrame] = []
    # ── Phase A: same-name negatives ──────────────────────────────────────
    exact_limit = _exact_neg_cap(limit)
    sql_exact = f"""\
SELECT DISTINCT ON (a1.name)
       a1.name AS name_a,
       a1.name AS name_b
FROM musicbrainz.artist a1
JOIN musicbrainz.artist a2
     ON a1.name = a2.name AND a1.id < a2.id
WHERE length(a1.name) >= 3
ORDER BY a1.name, RANDOM()
LIMIT {exact_limit}"""
    exact_df = _psql_csv(sql_exact)
    if not exact_df.empty:
        exact_df = exact_df.rename(columns={"name_a": "variant_a", "name_b": "variant_b"})
        exact_df["to_link"] = False
        exact_df["source"] = "mb_neg_exact"
        frames.append(exact_df[["variant_a", "variant_b", "to_link", "source"]])
        log.info("Phase A: %d same-name negative pairs.", len(exact_df))
    phase_a_count = len(exact_df) if not exact_df.empty else 0
    # ── Phase B: hard negatives via RapidFuzz ─────────────────────────────
    remaining = limit - phase_a_count
    if remaining > 0:
        fuzzy_df = _generate_hard_negatives(remaining, similarity_floor, limit)
        if not fuzzy_df.empty:
            frames.append(fuzzy_df)
            log.info("Phase B: %d hard-negative pairs.", len(fuzzy_df))
    if not frames:
        return pd.DataFrame(columns=["variant_a", "variant_b", "to_link", "source"])
    return pd.concat(frames, ignore_index=True)


def _generate_hard_negatives(
    limit: int,
    similarity_floor: int,
    neg_limit: int,
) -> pd.DataFrame:
    """
    Pulls a MBID-keyed pool of artist names from MBDB and uses RapidFuzz
    to find cross-artist pairs whose WRatio >= similarity_floor.

    Only keeps pairs where the two names map to different MBIDs, so every
    hard negative is verified as genuinely different artists.
    Pool size scales with the requested neg_limit for better yield.
    """
    pool_size = _name_pool_size(neg_limit)
    # Pulling a diverse (name, mbid) pool — artists with at least a few credits
    sql_pool = f"""\
SELECT a.name, a.gid::text AS mbid
FROM musicbrainz.artist a
JOIN musicbrainz.artist_credit_name acn ON acn.artist = a.id
WHERE length(a.name) >= 3
  AND a.name !~ '^\\[.+\\]$'
GROUP BY a.name, a.gid
HAVING COUNT(DISTINCT acn.artist_credit) >= 2
ORDER BY RANDOM()
LIMIT {pool_size}"""
    pool_df = _psql_csv(sql_pool)
    if pool_df.empty or len(pool_df) < 10:
        log.warning("Name pool too small for hard-negative generation.")
        return pd.DataFrame(columns=["variant_a", "variant_b", "to_link", "source"])
    names = pool_df["name"].tolist()
    # Building a name → mbid lookup (first MBID wins for duplicate names)
    name_to_mbid: dict[str, str] = {}
    for _, r in pool_df.iterrows():
        name_to_mbid.setdefault(r["name"], r["mbid"])
    # Finding similar-but-different pairs with RapidFuzz, verified by MBID
    pairs_seen: set[frozenset[str]] = set()
    rows: list[dict] = []
    for name in names:
        if len(rows) >= limit:
            break
        matches = process.extract(
            name,
            names,
            scorer=fuzz.WRatio,
            score_cutoff=similarity_floor,
            limit=5,
        )
        for match_name, _score, _ in matches:
            if match_name == name:
                continue
            # Verifying different MBIDs — skipping if same artist
            if name_to_mbid.get(name) == name_to_mbid.get(match_name):
                continue
            pair = frozenset((name, match_name))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)
            rows.append(
                {
                    "variant_a": name,
                    "variant_b": match_name,
                    "to_link": False,
                    "source": "mb_neg_fuzzy",
                }
            )
            if len(rows) >= limit:
                break
    if not rows:
        return pd.DataFrame(columns=["variant_a", "variant_b", "to_link", "source"])
    return pd.DataFrame(rows)


def augment_gold_standard(
    *,
    pos_limit: int = 5000,
    neg_limit: int = 5000,
    similarity_floor: int = 60,
) -> int:
    """
    Extracts positive and negative pairs from MBDB and writes gs_mb.parquet.

    Re-running overwrites the previous file with a fresh random sample.
    Returns the total number of rows written.
    """
    if not check_local_mb():
        raise RuntimeError("Cannot reach local MusicBrainz mirror. Is musicbrainz-docker running?")
    log.info(
        "Augmenting gold standard: pos_limit=%d, neg_limit=%d, similarity_floor=%d",
        pos_limit,
        neg_limit,
        similarity_floor,
    )
    positives = extract_positive_pairs(limit=pos_limit)
    negatives = extract_negative_pairs(limit=neg_limit, similarity_floor=similarity_floor)
    combined = pd.concat([positives, negatives], ignore_index=True)
    if combined.empty:
        log.warning("No pairs extracted — gs_mb.parquet not written.")
        return 0
    dump_parquet(combined, GS_MB_PQ)
    n_pos = int((combined["to_link"] == True).sum())  # noqa: E712
    n_neg = int((combined["to_link"] == False).sum())  # noqa: E712
    log.info("Wrote %d rows to gs_mb.parquet (%d pos, %d neg).", len(combined), n_pos, n_neg)
    return len(combined)
