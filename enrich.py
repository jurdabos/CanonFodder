from sqlalchemy import select, update
from DB import get_session
from DB.models import ArtistCountry, Scrobble
import mbAPI  # your wrapper around MusicBrainz


def enrich_artist_country():
    with get_session() as s:
        # Artists we still don't know the country of
        unknown = (
            s.scalars(
                select(Scrobble.artist_name)
                .outerjoin(ArtistCountry,
                           ArtistCountry.artist_name == Scrobble.artist_name)
                .where(ArtistCountry.id.is_(None))
                .distinct()
            )
            .all()
        )
        for name in unknown:
            country = mbAPI.fetch_country(name)
            s.execute(
                update(ArtistCountry)
                .where(ArtistCountry.artist_name == name)
                .values(country=country)
            ) if s.scalar(select(ArtistCountry.id).where(
                ArtistCountry.artist_name == name)) \
                else s.add(ArtistCountry(artist_name=name, country=country))
        s.commit()
