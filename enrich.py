from dotenv import load_dotenv

load_dotenv()
from DB import engine, SessionLocal  # noqa: I202
import logging
from sqlalchemy import select, update
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from DB.models import ArtistCountry, Scrobble
from mbAPI import search_artist, _cache_artist  # noqa  (private ok inside project)


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
        chunk_names = to_process[chunk_start : chunk_start + batch]
        rows: list[ArtistCountry] = []
        for name in chunk_names:
            result = search_artist(name, limit=1)
            if not result:
                continue
            cand = result[0]
            _cache_artist(cand)                         # keep mb_artists hot
            rows.append(
                ArtistCountry(
                    artist_name=name,
                    mbid=cand.get("id"),
                    country=cand.get("country"),
                    disambiguation_comment=cand.get("disambiguation"),
                )
            )
        # ── one INSERT IGNORE / ON CONFLICT DO NOTHING, handled by SQLAlchemy ──
        if rows:
            try:
                with SessionLocal() as sess:
                    sess.bulk_save_objects(rows, ignore_conflicts=True)
                    sess.commit()
            except SQLAlchemyError as exc:
                logging.warning("artist_country batch failed: %s", exc)
