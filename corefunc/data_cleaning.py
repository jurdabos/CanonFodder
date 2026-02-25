"""Provides data-cleaning helpers over Parquet files."""
from __future__ import annotations
import logging
import re
import pandas as pd
from helpers.io import (
    ARTIST_INFO_PQ, SCROBBLE_PQ,
    read_parquet, dump_parquet,
)

log = logging.getLogger(__name__)

# ── Encoding-repair constants ─────────────────────────────────────────────────
# Matching Unicode replacement char or C0/C1 control chars (except \n \r \t)
_BAD_CHAR_RE = re.compile(r"[\uFFFD\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
_SCROBBLE_TEXT_COLS = ["artist_name", "album_title", "track_title"]
_ARTIST_INFO_TEXT_COLS = ["artist_name", "disambiguation_comment", "aliases"]
# CP1252 bytes 0x80–0x9F → proper Unicode (0x81/0x8D/0x8F/0x90/0x9D undefined)
_CP1252_C1: dict[int, str] = {
    0x80: "\u20AC", 0x82: "\u201A", 0x83: "\u0192", 0x84: "\u201E",
    0x85: "\u2026", 0x86: "\u2020", 0x87: "\u2021", 0x88: "\u02C6",
    0x89: "\u2030", 0x8A: "\u0160", 0x8B: "\u2039", 0x8C: "\u0152",
    0x8E: "\u017D", 0x91: "\u2018", 0x92: "\u2019", 0x93: "\u201C",
    0x94: "\u201D", 0x95: "\u2022", 0x96: "\u2013", 0x97: "\u2014",
    0x98: "\u02DC", 0x99: "\u2122", 0x9A: "\u0161", 0x9B: "\u203A",
    0x9C: "\u0153", 0x9E: "\u017E", 0x9F: "\u0178",
}
# Reverse: CP1252-decoded Unicode chars → original byte values
_CP1252_REVERSE: dict[str, int] = {v: k for k, v in _CP1252_C1.items()}
# Accepting Shift-JIS only when the result actually contains Japanese characters
_CJK_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF]")


# ── Encoding-repair helpers ───────────────────────────────────────────────────
def _to_raw_bytes(text: str) -> bytes:
    """
    Recovers raw byte values from a string partially decoded as
    CP1252/Latin-1, reversing both C1 pass-through and CP1252 decoding.
    """
    result = bytearray()
    for ch in text:
        cp = ord(ch)
        if cp < 0x100:
            result.append(cp)
        elif ch in _CP1252_REVERSE:
            result.append(_CP1252_REVERSE[ch])
        else:
            raise ValueError(f"Cannot map U+{cp:04X} to a single byte")
    return bytes(result)


def _repair_text(text: str) -> str:
    """
    Attempts to repair encoding-corrupted text via a cascade of strategies.

    Tries (in order):
      1. Latin-1 → UTF-8      (double-decoded UTF-8 / mojibake)
      2. Latin-2 → UTF-8      (UTF-8 decoded as ISO 8859-2)
      3. Latin-1 → CP1252     (raw CP1252 bytes in C1 range)
      4. Raw bytes → Shift-JIS (partially decoded Shift-JIS)
      5. Char-level CP1252 map + strip unmappable C1 chars
    """
    if not _BAD_CHAR_RE.search(text):
        return text
    # Strategy 1: Latin-1 → UTF-8
    try:
        candidate = text.encode("latin-1").decode("utf-8")
        if not _BAD_CHAR_RE.search(candidate):
            return candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Strategy 2: Latin-2 → UTF-8
    try:
        candidate = text.encode("iso-8859-2").decode("utf-8")
        if not _BAD_CHAR_RE.search(candidate):
            return candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Strategy 3: Latin-1 → CP1252
    try:
        candidate = text.encode("latin-1").decode("cp1252")
        if not _BAD_CHAR_RE.search(candidate):
            return candidate
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Strategy 4: raw-byte recovery → Shift-JIS (only if result contains CJK/kana)
    try:
        raw = _to_raw_bytes(text)
        candidate = raw.decode("shift_jis")
        if not _BAD_CHAR_RE.search(candidate) and _CJK_RE.search(candidate):
            return candidate
    except (ValueError, UnicodeDecodeError):
        pass
    # Strategy 5: char-level CP1252 map; stripping unmappable C1 characters
    return "".join(
        _CP1252_C1.get(ord(ch), "") if 0x80 <= ord(ch) <= 0x9F else ch
        for ch in text
    )


def _fix_df_encoding(df: pd.DataFrame, text_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """
    Repairs encoding-corrupted strings in the given DataFrame.

    Scans *text_cols* for C1 control characters and attempts to
    recover the intended text via a cascade of codec round-trips.

    Returns the (possibly modified) DataFrame and the number of
    rows that were repaired.
    """
    repaired_indices: set[int] = set()
    for col in text_cols:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        series = df.loc[mask, col].astype(str)
        bad_mask = series.str.contains(_BAD_CHAR_RE, na=False)
        if not bad_mask.any():
            continue
        bad_idx = series[bad_mask].index
        df.loc[bad_idx, col] = series[bad_mask].apply(_repair_text)
        repaired_indices.update(bad_idx.tolist())
        log.info("Repaired %d values in column '%s'.", len(bad_idx), col)
    return df, len(repaired_indices)


def fix_encoding() -> dict[str, tuple[int, int]]:
    """
    Repairs encoding-corrupted strings in scrobble.parquet and
    artist_info.parquet.

    Returns
    -------
    dict mapping file label ('scrobble', 'artist_info') to
    (repaired_row_count, total_row_count).
    """
    results: dict[str, tuple[int, int]] = {}
    # Repairing scrobble.parquet
    df = read_parquet(SCROBBLE_PQ)
    if df is not None and not df.empty:
        df, repaired = _fix_df_encoding(df, _SCROBBLE_TEXT_COLS)
        if repaired:
            dump_parquet(df, SCROBBLE_PQ)
        results["scrobble"] = (repaired, len(df))
        log.info("scrobble.parquet: %d rows repaired out of %d.", repaired, len(df))
    else:
        results["scrobble"] = (0, 0)
    # Repairing artist_info.parquet
    ai = read_parquet(ARTIST_INFO_PQ)
    if ai is not None and not ai.empty:
        ai, repaired = _fix_df_encoding(ai, _ARTIST_INFO_TEXT_COLS)
        if repaired:
            dump_parquet(ai, ARTIST_INFO_PQ)
        results["artist_info"] = (repaired, len(ai))
        log.info("artist_info.parquet: %d rows repaired out of %d.", repaired, len(ai))
    else:
        results["artist_info"] = (0, 0)
    return results


# ── Artist-info dedup ─────────────────────────────────────────────────────────
def clean_artist_info() -> tuple[int, int]:
    """
    Deduplicates artist_info.parquet, keeping the most complete row per artist.

    Returns (removed_count, remaining_count).
    """
    df = read_parquet(ARTIST_INFO_PQ)
    if df is None or df.empty:
        return 0, 0
    before = len(df)
    # Scoring completeness: mbid(2) + country(1) + disambiguation(1) + aliases(1)
    df["_score"] = (
        df["mbid"].notna().astype(int) * 2
        + df["country"].notna().astype(int)
        + df["disambiguation_comment"].notna().astype(int)
        + df["aliases"].notna().astype(int)
    )
    df = df.sort_values("_score", ascending=False).drop_duplicates(subset=["artist_name"], keep="first")
    df = df.drop(columns=["_score"])
    dump_parquet(df, ARTIST_INFO_PQ)
    removed = before - len(df)
    log.info("Cleaned artist_info: removed %d dupes, %d remain.", removed, len(df))
    return removed, len(df)
