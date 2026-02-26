"""
Computes three-tier pairwise features for artist name variant classification.

Tier A — Whole-string: global similarity measures over the full name pair.
Tier B — Token-level:  features over space-separated name elements.
Tier C — Character-level: fine-grained character and n-gram features.

The single entry point is ``compute_pair_features(a, b) -> dict``.
"""
from __future__ import annotations
import logging
import unicodedata
from itertools import combinations
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

log = logging.getLogger(__name__)


# ── Tier A: whole-string features ─────────────────────────────────────────────
def _whole_string_features(a: str, b: str) -> dict[str, float]:
    """Computes 10 whole-string similarity features."""
    max_len = max(len(a), len(b), 1)
    min_len = max(min(len(a), len(b)), 1)
    lev_dist = Levenshtein.distance(a, b)
    return {
        "ratio": fuzz.ratio(a, b) / 100,
        "partial_ratio": fuzz.partial_ratio(a, b) / 100,
        "token_sort_ratio": fuzz.token_sort_ratio(a, b) / 100,
        "token_set_ratio": fuzz.token_set_ratio(a, b) / 100,
        "WRatio": fuzz.WRatio(a, b) / 100,
        "QRatio": fuzz.QRatio(a, b) / 100,
        "norm_levenshtein": lev_dist / max_len,
        "jaro_winkler": JaroWinkler.similarity(a, b),
        "length_ratio": min_len / max_len,
        "abs_len_diff": abs(len(a) - len(b)),
    }


# ── Tier B: token-level features ─────────────────────────────────────────────
def _tokenise(s: str) -> list[str]:
    """Splits on whitespace and lowercases."""
    return s.lower().split()


def _jaccard(set_a: set, set_b: set) -> float:
    """Returns Jaccard index, 0.0 when both sets are empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _longest_common_subseq_len(seq_a: list[str], seq_b: list[str]) -> int:
    """Returns the length of the longest common subsequence (tokens)."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _kendall_tau_displacement(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Returns normalised Kendall τ displacement on shared tokens.

    Values range from 0.0 (identical order) to 1.0 (fully reversed).
    Returns 0.0 when fewer than 2 tokens are shared.
    """
    shared = [t for t in tokens_a if t in set(tokens_b)]
    if len(shared) < 2:
        return 0.0
    # Building position map in tokens_b for shared tokens
    pos_b = {t: i for i, t in enumerate(tokens_b) if t in set(shared)}
    order_in_b = [pos_b[t] for t in shared if t in pos_b]
    if len(order_in_b) < 2:
        return 0.0
    # Counting discordant pairs
    n = len(order_in_b)
    discordant = sum(
        1 for i, j in combinations(range(n), 2) if order_in_b[i] > order_in_b[j]
    )
    max_pairs = n * (n - 1) / 2
    return discordant / max_pairs if max_pairs > 0 else 0.0


def _token_features(a: str, b: str) -> dict[str, float]:
    """Computes 5 token-level features."""
    toks_a = _tokenise(a)
    toks_b = _tokenise(b)
    set_a = set(toks_a)
    set_b = set(toks_b)
    total_toks = max(len(set_a | set_b), 1)
    shared = set_a & set_b
    return {
        "token_count_diff": abs(len(toks_a) - len(toks_b)),
        "token_jaccard": _jaccard(set_a, set_b),
        "shared_token_ratio": len(shared) / total_toks,
        "lcs_token_len": _longest_common_subseq_len(toks_a, toks_b),
        "token_order_displacement": _kendall_tau_displacement(toks_a, toks_b),
    }


# ── Tier C: character-level features ─────────────────────────────────────────
def _char_ngrams(s: str, n: int) -> set[str]:
    """Returns the set of character n-grams for a lowercased string."""
    s = s.lower()
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def _shared_prefix_len(a: str, b: str) -> int:
    """Returns the length of the longest shared prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _shared_suffix_len(a: str, b: str) -> int:
    """Returns the length of the longest shared suffix."""
    n = min(len(a), len(b))
    for i in range(1, n + 1):
        if a[-i] != b[-i]:
            return i - 1
    return n


def _unicode_script(ch: str) -> str:
    """Returns the Unicode script block for a character (simplified)."""
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return "UNKNOWN"
    if "CJK" in name:
        return "CJK"
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "LATIN" in name or "DIGIT" in name:
        return "LATIN"
    if "ARABIC" in name:
        return "ARABIC"
    if "HANGUL" in name:
        return "HANGUL"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "JAPANESE"
    return "OTHER"


def _script_mismatch_flag(a: str, b: str) -> int:
    """Returns 1 if the dominant scripts of a and b differ, else 0."""
    def dominant_script(s: str) -> str:
        """Finds the most common script in a string."""
        scripts: dict[str, int] = {}
        for ch in s:
            if ch.isalpha():
                sc = _unicode_script(ch)
                scripts[sc] = scripts.get(sc, 0) + 1
        if not scripts:
            return "LATIN"
        return max(scripts, key=scripts.get)  # type: ignore[arg-type]
    return int(dominant_script(a) != dominant_script(b))


def _character_features(a: str, b: str) -> dict[str, float]:
    """Computes 7 character-level features."""
    # n-gram overlaps
    bi_a, bi_b = _char_ngrams(a, 2), _char_ngrams(b, 2)
    tri_a, tri_b = _char_ngrams(a, 3), _char_ngrams(b, 3)
    # Levenshtein edit operation breakdown
    ops = Levenshtein.editops(a, b)
    n_insert = sum(1 for op in ops if op.tag == "insert")
    n_delete = sum(1 for op in ops if op.tag == "delete")
    n_replace = sum(1 for op in ops if op.tag == "replace")
    return {
        "bigram_jaccard": _jaccard(bi_a, bi_b),
        "trigram_jaccard": _jaccard(tri_a, tri_b),
        "edit_inserts": n_insert,
        "edit_deletes": n_delete,
        "edit_replaces": n_replace,
        "shared_prefix_len": _shared_prefix_len(a, b),
        "shared_suffix_len": _shared_suffix_len(a, b),
        "script_mismatch": _script_mismatch_flag(a, b),
    }


# ── Public entry point ────────────────────────────────────────────────────────
def compute_pair_features(a: str, b: str) -> dict[str, float]:
    """Computes all three-tier features for a single (variant_a, variant_b) pair.

    Returns a flat dict with ~23 numeric features suitable for ML.
    """
    a = a or ""
    b = b or ""
    feats: dict[str, float] = {}
    feats.update(_whole_string_features(a, b))
    feats.update(_token_features(a, b))
    feats.update(_character_features(a, b))
    return feats


# ── Legacy compatibility shim ─────────────────────────────────────────────────
def fuzzy_scores(a: str, b: str) -> dict[str, float]:
    """Returns the original 6-score dict for backward compatibility.

    Delegates to helpers.cluster.fuzzy_scores (canonical location).
    """
    from helpers.cluster import fuzzy_scores as _legacy
    return _legacy(a, b)
