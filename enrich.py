from dotenv import load_dotenv

load_dotenv()
from DB import engine, SessionLocal  # noqa: I202
from DB.models import ArtistCountry, Scrobble
from DB.ops import insert_ignore
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
    eng = engine
    backend = eng.url.get_backend_name()
    with SessionLocal() as s:
        unknown_q = (
            select(Scrobble.artist_name, Scrobble.artist_mbid)
            .distinct()
            .where(~Scrobble.artist_name.in_(select(ArtistCountry.artist_name)))
        )
        to_process = list(s.execute(unknown_q))
    for off in range(0, len(to_process), batch):
        chunk = to_process[off: off + batch]
        rows = []
        for name, mbid in chunk:
            if not mbid:  # need a lookup
                hit = search_artist(name, limit=1)
                if not hit:
                    continue
                cand = hit[0]
                mbid = cand["id"]
                ctry = cand.get("country")
                dis = cand.get("disambiguation")
                _cache_artist(cand)
            else:  # no network
                ctry = dis = None
            rows.append(
                ArtistCountry(
                    artist_name=name,
                    mbid=mbid,
                    country=ctry,
                    disambiguation_comment=dis,
                )
            )
        if not rows:
            continue
        stmt = insert_ignore(ArtistCountry.__table__, backend)
        payload = [r.__dict__ | {"id": None} for r in rows]  # id is autoincr
        with SessionLocal() as s:
            s.execute(stmt, payload)
            s.commit()


async def enrich_parquet_missing_countries(path: Path, batch: int = 50):
    df = pq.read_table(path).to_pandas()  # lazy-loads ac.parquet columns
    missing = df[df["country"].isna()].copy()
    if missing.empty:
        print("✓ No rows with NULL country – nothing to do.")
        return
    clf = llm.CanonFodderLLM(os.getenv("OPENAI_API_KEY"))
    cache: dict[str, str] = {}  # artist → ISO-2
    # Working in small batches to keep token usage flat
    for start in range(0, len(missing), batch):
        rows = missing.iloc[start:start + batch]
        tasks = []
        for artist in rows["artist_name"]:
            if artist in cache:  # to reuse result in the same run
                continue
            tasks.append(asyncio.create_task(
                clf.country_from_context(artist)))
        for t in tasks:  # to gather as they finish
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
