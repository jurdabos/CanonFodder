"""
Implements high-level data-loading helpers: normalises a dataframe of last.fm
scrobbles, bulk-inserts them with dialect-appropriate conflict handling, seeds
the ASCII lookup table, and offers assorted utility queries.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(".env")
from HTTP.mbAPI import lookup_mb_for
from DB import get_session as _get_session, get_engine as _get_engine
from datetime import datetime, UTC
from .models import ArtistVariantsCanonized, AsciiChar, Scrobble
import numpy as np
import pandas as pd
import re
from sqlalchemy import func, inspect, literal, select, text, union_all
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
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
    S = Scrobble.__table__ if target_table == "scrobble" \
        else text(target_table)
    # LIKE '%!%'  but properly escaped for %, _
    def esc(c):
        return c.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
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
    combined = union_all(*union_queries).alias("ascii_union")
    with eng.connect() as connection:
        df = pd.read_sql(
            select(combined).order_by(text("unique_artist_count DESC")),
            connection
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
    # Create a connection from the engine and use it with pandas
    with engine.connect() as connection:
        df = pd.read_sql(f"SELECT * FROM {table_name}", connection)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df, table_name


def populate_artist_info_from_scrobbles(
    session=None,
    progress_cb=None,
    max_artists: int | None = None,
    debug: bool = False,
    **_: dict,
) -> tuple[int, int, int]:
    """
    Synchronise the ArtistInfo table with the unique artists referenced in Scrobble.
    Steps
    -----
    1. Collect all unique artists from Scrobble (MBID first, then raw name).
    2. For every artist:
       • If an ArtistInfo entry exists – update missing details.
       • Otherwise – create a brand-new ArtistInfo record.
    3. Report progress through an optional callback.
    Parameters
    ----------
    session : sqlalchemy.orm.Session | None
        Database session.  A new one is created when *None*.
    progress_cb : Callable[[str, int, str], None] | None
        Callback invoked as ``progress_cb(phase, percent, message)``.
    max_artists : int | None
        Optional hard limit (used mainly for tests).
    debug : bool
        Enable very verbose logging when *True*.
    Returns
    -------
    tuple[int, int, int]
        ``(processed, created, updated)`` counts.
    """
    # ------------------------------------------------------------------ setup
    import logging
    logger = logging.getLogger("populate_artist_info")
    if debug:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    from HTTP.mbAPI import get_complete_artist_info  # local import to avoid circulars
    from DB import get_session
    from DB.models import Scrobble, ArtistInfo  # type: ignore  # noqa: F401
    session = session or get_session()

    # ----------------------------------------------------------- helper funcs
    def _emit_progress(message: str, pct: int | None = None) -> None:
        if progress_cb:
            percent = (
                pct
                if pct is not None
                else int(
                    (processed * 85)
                    / max(1, len(with_mbid) + len(without_mbid))
                )
            )
            progress_cb("Enriching", percent, message)

    # ---------------------------------------------------------- gather input
    with_mbid_q = (
        session.query(Scrobble.artist_mbid, Scrobble.artist_name)
        .filter(Scrobble.artist_mbid.isnot(None))
        .group_by(Scrobble.artist_mbid, Scrobble.artist_name)
    )
    without_mbid_q = (
        session.query(Scrobble.artist_name)
        .filter(Scrobble.artist_mbid.is_(None))
        .group_by(Scrobble.artist_name)
    )
    if max_artists:
        logger.debug("Applying artist limit: %s", max_artists)
        with_mbid = with_mbid_q.limit(max_artists).all()
        without_mbid = without_mbid_q.limit(max_artists).all()
    else:
        with_mbid = with_mbid_q.all()
        without_mbid = without_mbid_q.all()
    logger.info(
        "Discovered %d artists with MBID and %d without",
        len(with_mbid),
        len(without_mbid),
    )
    # -------------------------------------------------------------- main loop
    processed = created = updated = 0
    _emit_progress("Processing artists with MBIDs …")
    # ---------- pass 1 : artists that already provide an MBID
    for mbid, raw_name in with_mbid:
        processed += 1
        _emit_progress(f"{raw_name} ({mbid})")
        try:
            ai: ArtistInfo | None = (
                session.query(ArtistInfo).filter_by(mbid=mbid).one_or_none()
            )
            if ai:
                # Update missing columns
                fields_missing = any(
                    getattr(ai, fld) in (None, "")
                    for fld in ("country", "disambiguation_comment", "aliases")
                )
                if fields_missing:
                    info = get_complete_artist_info(identifier=mbid)
                    logger.debug("MB response for %s: %s", mbid, info)
                    if ai.country in (None, "") and info.get("country"):
                        ai.country = info["country"]
                    if (
                        ai.disambiguation_comment in (None, "")
                        and info.get("disambiguation")
                    ):
                        ai.disambiguation_comment = info["disambiguation"]

                    if ai.aliases in (None, "") and info.get("aliases"):
                        aliases = info["aliases"]
                        ai.aliases = ",".join(aliases) if isinstance(aliases, list) else str(aliases)
                    updated += 1
            else:
                # Brand-new entry
                info = get_complete_artist_info(identifier=mbid)
                aliases_raw = info.get("aliases", [])
                aliases_str = ",".join(aliases_raw) if isinstance(aliases_raw, list) else str(aliases_raw or "")
                session.add(
                    ArtistInfo(
                        artist_name=info.get("name", raw_name),
                        mbid=info.get("id", mbid),
                        country=info.get("country"),
                        disambiguation_comment=info.get("disambiguation"),
                        aliases=aliases_str,
                    )
                )
                created += 1
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed processing artist %s (%s): %s", raw_name, mbid, exc)
            session.rollback()
            continue
        if processed % 100 == 0:  # commit in batches
            session.commit()
            logger.debug("Committed batch (%d processed)", processed)
    # ---------- pass 2 : artists that *lack* an MBID
    _emit_progress("Processing artists without MBIDs …", 90)
    for (raw_name,) in without_mbid:
        processed += 1
        _emit_progress(raw_name)
        try:
            ai: ArtistInfo | None = (
                session.query(ArtistInfo)
                .filter_by(artist_name=raw_name, mbid=None)
                .one_or_none()
            )
            if not ai:
                # MusicBrainz lookup by name
                info = get_complete_artist_info(identifier=raw_name)
                if not info:
                    logger.debug("No MB data for '%s'; skipping", raw_name)
                    continue
                aliases_raw = info.get("aliases", [])
                aliases_str = ",".join(aliases_raw) if isinstance(aliases_raw, list) else str(aliases_raw or "")
                session.add(
                    ArtistInfo(
                        artist_name=info.get("name", raw_name),
                        mbid=info.get("id"),  # may still be None
                        country=info.get("country"),
                        disambiguation_comment=info.get("disambiguation"),
                        aliases=aliases_str,
                    )
                )
                created += 1
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed processing artist '%s': %s", raw_name, exc)
            session.rollback()
            continue
        if processed % 100 == 0:
            session.commit()
    # ---------------------------------------------------------------- finish
    session.commit()
    _emit_progress("Completed", 100)
    logger.info(
        "populate_artist_info_from_scrobbles → processed=%d, created=%d, updated=%d",
        processed,
        created,
        updated,
    )
    return processed, created, updated


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
    with _get_session(engine) as ses:
        # only seed when empty
        if ses.query(AsciiChar).first() is None:
            ses.bulk_save_objects(
                [AsciiChar(ascii_code=c, ascii_char=s) for c, s in _PRINTABLE_NONALNUM]
            )
            ses.commit()
