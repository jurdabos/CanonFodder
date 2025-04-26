from __future__ import annotations
from dotenv import load_dotenv

load_dotenv(".env")
from DB import get_session as _get_session, get_engine as _get_engine
from contextlib import contextmanager, suppress
from datetime import datetime, timezone, UTC
from .models import Scrobble
import pandas as pd
import re
from DB.models import ArtistVariantsCanonized, Scrobble
from sqlalchemy import DateTime, insert, inspect, select, String
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Generator


def _dialect_insert(model, dialectengine):
    name = dialectengine.url.get_backend_name()
    if name.startswith("mysql"):
        from sqlalchemy.dialects.mysql import insert as dialect_insert
    elif name in {"postgresql", "postgres"}:
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:  # sqlite, oracle, …
        from sqlalchemy import insert as dialect_insert
    return dialect_insert(model)


def _prepare_scrobble_rows(df: pd.DataFrame) -> list[dict]:
    """
    Normalise *df* so it is ready for the SQLAlchemy INSERT.
    Works with either
        •   ['Artist','Album','Song','uts']       # new fetch
        •   ['Artist','Album','Song','Timestamp'] # legacy fetch
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
    if "uts" in df.columns:  # ↔ new workflow
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


def bulk_insert_scrobbles(df: pd.DataFrame, hereengine=None) -> str:
    eng   = hereengine or _get_engine()
    rows  = _prepare_scrobble_rows(df)
    # 1) Just the INSERT … VALUES () skeleton – no rows yet
    insert_tpl = _dialect_insert(Scrobble, eng)
    if eng.url.get_backend_name().startswith("mysql"):
        insert_tpl = insert_tpl.on_duplicate_key_update(id=insert_tpl.inserted.id)
    elif eng.url.get_backend_name() in {"postgresql", "postgres"}:
        insert_tpl = insert_tpl.on_conflict_do_nothing(
            index_elements=["artist_name", "track_title", "play_time"])
    elif eng.url.get_backend_name() == "sqlite":
        insert_tpl = insert_tpl.prefix_with("OR IGNORE")
    with _get_session() as sess:
        for i in range(0, len(rows), 10_000):
            chunk = rows[i : i + 10_000]
            sess.execute(insert_tpl.values(chunk))
        sess.commit()
    # return table name so main() can parquet-dump
    table_name = Scrobble.__tablename__
    return table_name


def bulk_insert_scrobbles_to_sqlite(df, sqliteengine):
    bulk_insert_scrobbles(df, sqliteengine)


def bulk_insert_scrobbles_to_mysql(df, mysqlengine):
    bulk_insert_scrobbles(df, mysqlengine)


# ---------------------------------------------------------------
# Inserting a canonical name for a variant group
# ---------------------------------------------------------------
def save_group(signature: str, canonical: str):
    with _get_session() as ses:
        obj = ses.get(ArtistVariantsCanonized, signature) \
              or ArtistVariantsCanonized(artist_variants=signature)
        obj.canonical_name = canonical
        obj.timestamp = datetime.now(UTC)
        ses.merge(obj)
        ses.commit()


# ---------------------------------------------------------------
# Load latest scrobbles_* table into a df
# ---------------------------------------------------------------
def latest_scrobble_df(current_engine) -> tuple[pd.DataFrame | None, str | None]:
    insp = inspect(current_engine)
    regex = re.compile(r"^scrobbles_(\d{8}_\d{6})$")
    latest_name, latest_dt = None, None
    for name in insp.get_table_names():
        m = regex.match(name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        if latest_dt is None or ts > latest_dt:
            latest_dt, latest_name = ts, name
    if latest_name is None:
        return None, None
    return pd.read_sql_table(latest_name, current_engine), latest_name
