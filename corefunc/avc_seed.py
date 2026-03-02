"""
Parses a MySQL dump of the artist_variants_canonized table and writes
the result to avc.parquet in the canonical schema.

Intended as a one-time migration helper.
"""

from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from helpers.io import AVC_PQ, dump_parquet

log = logging.getLogger(__name__)

# Matching individual value tuples inside the INSERT statement
_TUPLE_RE = re.compile(r"\((?:[^()]*?'(?:[^'\\]|\\.)*'[^()]*?)*\)")


def _parse_sql_value(raw: str) -> str | None:
    """Unescapes a single MySQL value literal, returning None for NULL."""
    raw = raw.strip()
    if raw.upper() == "NULL":
        return None
    if raw.startswith("'") and raw.endswith("'"):
        inner = raw[1:-1]
        # Unescaping MySQL backslash sequences
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")
        return inner
    return raw


def _parse_tuple(tup: str) -> tuple:
    """
    Splits a MySQL VALUES tuple into individual field values.

    Handles commas inside quoted strings correctly.
    """
    # Stripping outer parentheses
    inner = tup.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    fields: list[str] = []
    buf = ""
    in_quote = False
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and in_quote and i + 1 < len(inner):
            buf += ch + inner[i + 1]
            i += 2
            continue
        if ch == "'":
            in_quote = not in_quote
            buf += ch
        elif ch == "," and not in_quote:
            fields.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    if buf:
        fields.append(buf)
    return tuple(_parse_sql_value(f) for f in fields)


def seed_avc_from_sql(sql_path: str | Path) -> int:
    """
    Reads a MySQL dump file and writes PQ/avc.parquet.

    Expects the dump to contain a single INSERT INTO `artist_variants_canonized`
    with columns: artist_variants_hash, artist_variants_text, to_link,
    canonical_name, comment, timestamp, mbid.

    Returns the number of rows written.
    """
    sql_path = Path(sql_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL dump not found: {sql_path}")
    text = sql_path.read_text(encoding="utf-8")
    # Extracting VALUES portion
    m = re.search(r"INSERT INTO.*?VALUES\s*", text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("No INSERT INTO … VALUES found in the SQL file.")
    values_text = text[m.end() :]
    # Finding all tuples
    tuples = _TUPLE_RE.findall(values_text)
    if not tuples:
        raise ValueError("No value tuples found in the INSERT statement.")
    rows: list[dict] = []
    for raw_tuple in tuples:
        fields = _parse_tuple(raw_tuple)
        if len(fields) < 7:
            log.warning("Skipping malformed tuple with %d fields: %s…", len(fields), raw_tuple[:80])
            continue
        # Mapping MySQL columns → Parquet schema
        to_link_raw = fields[2]
        if to_link_raw is None:
            to_link = None
        else:
            to_link = bool(int(to_link_raw))
        ts_raw = fields[5]
        if ts_raw:
            stamp = pd.Timestamp(datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc))
        else:
            stamp = pd.Timestamp(datetime.now(timezone.utc))
        rows.append(
            {
                "artist_variants_hash": fields[0],
                "artist_variants_text": fields[1],
                "canonical_name": fields[3] or "",
                "to_link": to_link,
                "comment": fields[4] or "",
                "stamp": stamp,
            }
        )
    df = pd.DataFrame(rows)
    # Enforcing correct dtypes
    df["to_link"] = df["to_link"].astype("boolean")
    df["stamp"] = pd.to_datetime(df["stamp"], utc=True)
    dump_parquet(df, AVC_PQ)
    log.info("Seeded avc.parquet with %d rows from %s", len(df), sql_path)
    return len(df)
