"""
Implements high-level data-loading helpers: normalises a dataframe of last.fm
scrobbles, bulk-inserts them with dialect-appropriate conflict handling, seeds
the ASCII lookup table, and offers assorted utility queries.
"""
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv(".env")
from DB import get_session as _get_session, get_engine as _get_engine
from contextlib import contextmanager, suppress
from datetime import datetime, timezone, UTC
from .models import ArtistVariantsCanonized, AsciiChar, Scrobble
import pandas as pd
from sqlalchemy import DateTime, func, insert, inspect, literal, select, String, text, union_all
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Generator

_PRINTABLE_NONALNUM = [
    (33, '!'), (34, '"'), (35, '#'), (36, '$'), (37, '%'), (38, '&'),
    (39, "'"), (40, '('), (41, ')'), (42, '*'), (43, '+'), (44, ','),
    (45, '-'), (46, '.'), (47, '/'),
    (58, ':'), (59, ';'), (60, '<'), (61, '='), (62, '>'), (63, '?'),
    (64, '@'),
    (91, '['), (92, '\\'), (93, ']'), (94, '^'), (95, '_'), (96, '`'),
    (123, '{'), (124, '|'), (125, '}'), (126, '~')
]


def _prepare_scrobble_rows(df: pd.DataFrame) -> list[dict]:
    """
    Normalises `df` so it fits the Scrobble schema, then returns a list of
    dictionaries ready for SQLAlchemy bulk insert
    """
    # ---------- 1. normalise column names ----------
    df = df.rename(
        columns={
            "Artist": "artist_name",
            "Album": "album_title",
            "Song": "track_title",
        },
        errors="ignore",
    )
    # ---------- 2. create `play_time` ----------
    if "uts" in df.columns:
        df["play_time"] = pd.to_datetime(
            df["uts"].astype(int), unit="s", utc=True
        )
        dedup_subset = ["artist_name", "track_title", "uts"]
    else:  # ↔ legacy workflow
        df["play_time"] = (
            pd.to_datetime(df["Timestamp"], utc=True)
            .dt.tz_convert("UTC")
        )
        dedup_subset = ["artist_name", "track_title", "play_time"]
    # ---------- 3. drop duplicates & final shape ----------
    df = (
        df.drop_duplicates(subset=dedup_subset)
        [["artist_name", "album_title", "track_title", "play_time"]]
    )
    return df.to_dict(orient="records")


def ascii_freq(engine=None, target_table: str = "scrobbles") -> pd.Series:
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
    S = Scrobble.__table__ if target_table == "scrobbles" \
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
    Inserts *df* into the `scrobbles` table in reasonably big chunks.
    • MySQL   → IGNORE duplicates (artist, track, play_time must be UNIQUE)
    • Postgres→ ON CONFLICT DO NOTHING
    • SQLite  → INSERT OR IGNORE
    Returns the physical table name so that the caller can parquet-dump it.
    """
    eng = hereengine or _get_engine()
    backend = eng.url.get_backend_name()
    # ---- 1. Build a dialect-specific empty INSERT template ------------------
    if backend.startswith("mysql"):
        insert_tpl = mysql_insert(Scrobble).prefix_with("IGNORE")
    elif backend in {"postgresql", "postgres"}:
        insert_tpl = (
            pg_insert(Scrobble)
            .on_conflict_do_nothing(
                index_elements=["artist_name", "track_title", "play_time"]
            )
        )
    elif backend == "sqlite":
        insert_tpl = sqlite_insert(Scrobble).prefix_with("OR IGNORE")
    else:  # fallback that explodes on duplicate keys
        insert_tpl = Scrobble.__table__.insert()
    # ---- 2. Converting the frame to list-of-dicts *once* ------------------------
    rows = _prepare_scrobble_rows(df)
    # ---- 3. Firing the chunks --------------------------------------------------
    with _get_session() as sess:
        step = 10_000
        for i in range(0, len(rows), step):
            sess.execute(insert_tpl.values(rows[i:i + step]))
        sess.commit()
    return Scrobble.__tablename__


def bulk_insert_scrobbles_to_sqlite(df, sqliteengine):
    """Proxy that calls bulk_insert_scrobbles with an SQLite engine"""
    bulk_insert_scrobbles(df, sqliteengine)


def bulk_insert_scrobbles_to_mysql(df, mysqlengine):
    """Proxy that calls bulk_insert_scrobbles with a MySQL engine"""
    bulk_insert_scrobbles(df, mysqlengine)


def save_group(signature: str, canonical: str):
    """Upserts one variant group with its canonical name and timestamp"""
    with _get_session() as ses:
        obj = ses.get(ArtistVariantsCanonized, signature) \
              or ArtistVariantsCanonized(artist_variants=signature)
        obj.canonical_name = canonical
        obj.timestamp = datetime.now(UTC)
        ses.merge(obj)
        ses.commit()


def latest_scrobble_table_to_df(engine) -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (dataframe, 'scrobbles') if the unified table exists else (None, None)
    """
    table_name = "scrobbles"
    # Does the table exist?
    if table_name not in inspect(engine).get_table_names():
        return None, None
    df = pd.read_sql_table(table_name, engine)
    return df, table_name


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
