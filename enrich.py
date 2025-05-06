from dotenv import load_dotenv

load_dotenv()
from DB import engine, SessionLocal  # noqa: I202
from DB.models import ArtistCountry, Scrobble
import asyncio
from corefunc import llm
import logging
from mbAPI import search_artist, _cache_artist  # noqa  (private ok inside project)
import os
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sa
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

PARQUET_PATH = Path("/PQ/ac.parquet")
TEMP_PATH = PARQUET_PATH.with_suffix(".tmp")  # for atomic overwrite


def enrich_artist_country(*, batch: int = 100) -> None:
    """
    Fill `artist_country` with MBID + country + disambiguation.
    ▸ honours MusicBrainz rate-limit via mbAPI.search_artist
    ▸ skips rows that already exist (unique constraint) in a **DB-agnostic** way
    """
    # ── find names we do NOT have yet ────────────────────────────────────
    with SessionLocal() as sess:
        unknown_q = (
            select(Scrobble.artist_name)
            .distinct()
            .where(~Scrobble.artist_name.in_(select(ArtistCountry.artist_name)))
        )
        to_process = [row[0] for row in sess.execute(unknown_q)]
    for chunk_start in range(0, len(to_process), batch):
        chunk_names = to_process[chunk_start: chunk_start + batch]
        rows: list[ArtistCountry] = []
        for name in chunk_names:
            result = search_artist(name, limit=1)
            if not result:
                continue
            cand = result[0]
            _cache_artist(cand)  # to keep mb_artists hot
            rows.append(
                ArtistCountry(
                    artist_name=name,
                    mbid=cand.get("id"),
                    country=cand.get("country"),
                    disambiguation_comment=cand.get("disambiguation"),
                )
            )
        if rows:
            with SessionLocal() as sess:
                for r in rows:
                    with sess.no_autoflush:
                        existing = (
                            sess.query(ArtistCountry)
                            .filter_by(artist_name=r.artist_name)
                            .one_or_none()
                        )
                        if existing:
                            if not existing.mbid and r.mbid:
                                existing.mbid = r.mbid
                            if not existing.disambiguation_comment and r.disambiguation_comment:
                                existing.disambiguation_comment = r.disambiguation_comment
                            if not existing.country and r.country:
                                existing.country = r.country
                        else:
                            sess.add(r)
                try:
                    sess.commit()
                except SQLAlchemyError as exc:
                    logging.warning("artist_country batch failed: %s", exc)
                    sess.rollback()


async def enrich_parquet_missing_countries(path: Path, batch: int = 50):
    df = pq.read_table(path).to_pandas()       # lazy-loads ac.parquet columns
    missing = df[df["country"].isna()].copy()
    if missing.empty:
        print("✓ No rows with NULL country – nothing to do.")
        return
    clf = llm.CanonFodderLLM(os.getenv("OPENAI_API_KEY"))
    cache: dict[str, str] = {}                 # artist → ISO-2
    # Working in small batches to keep token usage flat
    for start in range(0, len(missing), batch):
        rows = missing.iloc[start:start + batch]
        tasks = []
        for artist in rows["artist_name"]:
            if artist in cache:                # to reuse result in the same run
                continue
            tasks.append(asyncio.create_task(
                clf.country_from_context(artist)))
        for t in tasks:                        # to gather as they finish
            result = await t
            if result["confidence"] >= 0.85 and result["country_iso2"]:
                cache[result["artist_name"]] = result["country_iso2"]
        # Writing back into the slice we just processed
        for idx, artist in rows["artist_name"].items():
            if artist in cache:
                df.at[idx, "country"] = cache[artist]
    # ---- Persisting atomically ----
    pq.write_table(pa.Table.from_pandas(df), TEMP_PATH, compression="zstd")
    TEMP_PATH.replace(path)
    print(f"✓ Updated {path.name}: filled {len(cache)} countries")


async def fill_missing_countries(session):
    BATCH = 50
    rows = session.execute(
        sa.text("""
            SELECT id, artist_name
            FROM artist
            WHERE country IS NULL
            ORDER BY id
            LIMIT :n
        """),
        {"n": BATCH}
    ).fetchall()

    clf = llm.CanonFodderLLM(os.getenv("OPENAI_API_KEY"))
    for row in rows:
        msg = await clf.country_from_context(row.artist_name)
        if msg["country_iso2"] and msg["confidence"] >= 0.85:
            session.execute(
                sa.text("UPDATE artist SET country = :c WHERE id = :i"),
                {"c": msg["country_iso2"], "i": row.id}
            )
    session.commit()
