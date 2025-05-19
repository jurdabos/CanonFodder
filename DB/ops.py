"""
Implements high-level data-loading helpers: normalises a dataframe of last.fm
scrobbles, bulk-inserts them with dialect-appropriate conflict handling, seeds
the ASCII lookup table, and offers assorted utility queries.
"""
from __future__ import annotations
from dotenv import load_dotenv

from mbAPI import lookup_mb_for

load_dotenv(".env")
from DB import get_session as _get_session, get_engine as _get_engine
from datetime import datetime, UTC
from .models import ArtistCountry, ArtistVariantsCanonized, AsciiChar, Scrobble
import numpy as np
import pandas as pd
import re
from sqlalchemy import func, insert, inspect, literal, select, text, union_all
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Insert


_COLUMN_ALIASES = {
    "Artist": "artist_name",
    "Album": "album_title",
    "Song": "track_title",
    # common spellings coming from XML→dict/json→df conversions
    "artist mbid": "artist_mbid",
    "artist_mbid": "artist_mbid",
    "mbid": "artist_mbid",
}
_PRINTABLE_NONALNUM = [
    (33, '!'), (34, '"'), (35, '#'), (36, '$'), (37, '%'), (38, '&'),
    (39, "'"), (40, '('), (41, ')'), (42, '*'), (43, '+'), (44, ','),
    (45, '-'), (46, '.'), (47, '/'),
    (58, ':'), (59, ';'), (60, '<'), (61, '='), (62, '>'), (63, '?'),
    (64, '@'),
    (91, '['), (92, '\\'), (93, ']'), (94, '^'), (95, '_'), (96, '`'),
    (123, '{'), (124, '|'), (125, '}'), (126, '~')
]
KEEP_COLS = [
    "artist_name",
    "album_title",
    "track_title",
    "play_time",
    "artist_mbid",
]

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _bulk_insert(table, rows: list[dict], engine=None, step=10_000):
    """
    Generic bulk insert utility that handles different database backends.
    :param table: SQLAlchemy table object
    :param rows: List of dictionaries containing row data
    :param engine: Optional SQLAlchemy engine
    :param step: Batch size for inserts
    :return: Number of rows inserted
    """
    if not rows:
        return 0
    eng = engine or _get_engine()
    backend = eng.url.get_backend_name()
    stmt_tpl = insert_ignore(table, backend)
    with _get_session() as sess:
        for i in range(0, len(rows), step):
            sess.execute(stmt_tpl.values(rows[i:i + step]))
        sess.commit()
    return len(rows)


def _prepare_scrobble_rows(df: pd.DataFrame) -> list[dict]:
    """
    Normalises `df` so it fits the Scrobble schema, then returns a list of
    dictionaries ready for SQLAlchemy bulk insert
    """
    # ---------- 1. normalise column names ----------
    df = df.rename(columns=_COLUMN_ALIASES, errors="ignore")
    # ---------- 2. timestamps ----------
    if "uts" in df.columns:
        df["play_time"] = pd.to_datetime(df["uts"].astype(int), unit="s", utc=True)
        dedup_subset = ["artist_name", "track_title", "uts"]
    else:
        df["play_time"] = (
            pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("UTC")
        )
        dedup_subset = ["artist_name", "track_title", "play_time"]
    # ---------- 3. data hygiene ----------
    df["artist_mbid"] = (
        df.get("artist_mbid", "")
        .astype(str)
        .str.strip()
        .mask(lambda s: ~s.str.match(_UUID_RE) | (s == ""))
    )
    # ---------- 4. final shape ----------
    df = df.drop_duplicates(subset=dedup_subset)[KEEP_COLS]
    df = df.replace({np.nan: None, pd.NA: None})
    return df.to_dict(orient="records")


def ascii_freq(engine=None, target_table: str = "scrobble") -> pd.Series:
    """
    Returns a pandas Series indexed by ascii_char with the number of
    *distinct* artists containing that char, sorted descending.
    Example
    -------
    >>> seed_ascii_chars()          # safe to call many times
    >>> ascii_freq()
    !     142
    ,     103
    …      …
    dtype: int64
    """
    eng = engine or _get_engine()
    seed_ascii_chars(eng)  # to ensure lookup exists
    A = AsciiChar.__table__
    S = Scrobble.__table__ if target_table == "scrobble" \
        else text(target_table)
    # LIKE '%!%'  but properly escaped for %, _
    esc = lambda c: c.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    union_queries = []
    for row in _PRINTABLE_NONALNUM:
        code, ch = row
        like_expr = func.count(func.distinct(S.c.artist_name)) \
            .label(f"cnt_{code}")
        subq = (
            select(like_expr)
            .where(S.c.artist_name.like(f"%{esc(ch)}%", escape="\\"))
        ).alias(f"q_{code}")
        union_queries.append(
            select(
                literal(ch).label("ascii_char"),
                subq.c[0].label("unique_artist_count")
            )
        )
    union = union_queries[0]
    combined = union_all(*union_queries).alias("ascii_union")
    df = pd.read_sql(
        select(combined).order_by(text("unique_artist_count DESC")),
        eng
    )
    return df.set_index("ascii_char")["unique_artist_count"]


def bulk_insert_scrobbles(df: pd.DataFrame, hereengine=None) -> str:
    """
    Inserts *df* into the `scrobble` table in reasonably big chunks.
    • MySQL   → IGNORE duplicates (artist, track, play_time must be UNIQUE)
    • Postgres→ ON CONFLICT DO NOTHING
    • SQLite  → INSERT OR IGNORE
    Returns the physical table name so that the caller can parquet-dump it.
    """
    rows = _prepare_scrobble_rows(df)
    _bulk_insert(Scrobble.__table__, rows, hereengine)
    return Scrobble.__tablename__


def bulk_insert_scrobbles_to_sqlite(df, sqliteengine):
    """Proxy that calls bulk_insert_scrobbles with an SQLite engine"""
    bulk_insert_scrobbles(df, sqliteengine)


def bulk_insert_scrobbles_to_mysql(df, mysqlengine):
    """Proxy that calls bulk_insert_scrobbles with a MySQL engine"""
    bulk_insert_scrobbles(df, mysqlengine)


def bulk_upsert_artist_country(session: Session, rows: list[dict]):
    """
    Insert many ArtistCountry rows; keep the first row per MBID.
    Works on SQLite, Postgres and MySQL 8+ without per-call branching.
    """
    dialect = session.bind.dialect.name          # 'sqlite' | 'postgresql' | 'mysql'
    if dialect == "sqlite":
        stmt = (
            sqlite_insert(ArtistCountry)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["mbid"])
        )
    elif dialect == "postgresql":
        stmt = (
            pg_insert(ArtistCountry)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["mbid"])
        )
    elif dialect == "mysql":
        # MySQL 8: INSERT IGNORE … keeps the first row, silently discards dups
        stmt = mysql_insert(ArtistCountry).values(rows).prefix_with("IGNORE")
    else:                                         # fallback: do the safe thing
        seen = {r.mbid for r in session.query(ArtistCountry.mbid)}
        stmt = insert(ArtistCountry).values([r for r in rows if r["mbid"] not in seen])
    session.execute(stmt)


def insert_ignore(table, backend: str) -> Insert:
    """Return an INSERT that silently skips rows violating UNIQUE / PK."""
    if backend.startswith("mysql"):
        return mysql_insert(table).prefix_with("IGNORE")
    if backend in {"postgresql", "postgres"}:
        return pg_insert(table).on_conflict_do_nothing()
    if backend == "sqlite":
        return sqlite_insert(table).prefix_with("OR IGNORE")
    return table.insert()          # generic – will raise on duplicates


def load_scrobble_table_from_db_to_df(engine) -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (dataframe, 'scrobble') if the unified table exists else (None, None)
    """
    table_name = "scrobble"
    # Does the table exist?
    if table_name not in inspect(engine).get_table_names():
        return None, None
    df = pd.read_sql_table(table_name, engine)
    return df, table_name


def save_group(signature: str, canonical: str):
    """Upserts one variant group with its canonical name and timestamp"""
    with _get_session() as ses:
        obj = ses.get(ArtistVariantsCanonized, signature) \
              or ArtistVariantsCanonized(artist_variants_text=signature)
        obj.canonical_name = canonical
        obj.mbid = lookup_mb_for(obj.canonical_name) or obj.mbid
        obj.timestamp = datetime.now(UTC)
        ses.merge(obj)
        ses.commit()


def seed_ascii_chars(engine=None) -> None:
    """
    Inserts the ASCII lookup rows once and becomes a no-op afterwards.
    """
    eng = engine or _get_engine()
    with _get_session() as ses:
        # only seed when empty
        if ses.query(AsciiChar).first() is None:
            ses.bulk_save_objects(
                [AsciiChar(ascii_code=c, ascii_char=s) for c, s in _PRINTABLE_NONALNUM]
            )
            ses.commit()
