"""
Defines all SQLAlchemy ORM ML used by CanonFodder.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Optional
from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, func, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Serves as the common declarative base so metadata stays centralised"""
    pass


class ArtistInfo(Base):
    """Caches MusicBrainz artist information including country mappings and aliases"""
    __tablename__ = "artist_info"
    __table_args__ = (
        UniqueConstraint("id", name="uq_artist_info"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    artist_name: Mapped[str] = mapped_column(String(750), index=True)
    mbid: Mapped[Optional[str]] = mapped_column(String(36), unique=True, nullable=True)
    disambiguation_comment: Mapped[str] = mapped_column(String(558))
    aliases: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(255))


class ArtistVariantsCanonized(Base):
    """Stores variant strings mapped to a canonical artist name"""
    __tablename__ = "artist_variants_canonized"
    artist_variants_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    artist_variants_text: Mapped[str] = mapped_column(Text)
    canonical_name: Mapped[str] = mapped_column(String(255))
    mbid: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    to_link: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(String(750), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AsciiChar(Base):
    """Lookup table with printable non-alphanumeric ASCII characters 33–126"""
    __tablename__ = "ascii_chars"
    ascii_code: Mapped[int] = mapped_column(Integer, primary_key=True)
    ascii_char: Mapped[str] = mapped_column(String(1), unique=True, nullable=False)


class CountryCode(Base):
    """Lookup table with country codes, English country name and Hungarian country name"""
    __tablename__ = "country_code"
    iso2: Mapped[str] = mapped_column(String(2), primary_key=True, unique=True)
    iso3: Mapped[str] = mapped_column(String(3), unique=True)
    en_name: Mapped[str] = mapped_column(String(350), unique=True, nullable=False)
    hu_name: Mapped[Optional[str]] = mapped_column(String(350), unique=True, nullable=True)


class Scrobble(Base):
    """Represents a single play event identified by artist track and UTC time"""
    __tablename__ = "scrobble"
    __table_args__ = (
        UniqueConstraint(
            "artist_name", "track_title", "play_time",
            name="uq_scrobble"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    artist_name: Mapped[str] = mapped_column(String(350))
    artist_mbid: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    album_title: Mapped[str] = mapped_column(String(350))
    track_title: Mapped[str] = mapped_column(String(350))
    play_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True)


class UserCountry(Base):
    """Holds non-overlapping location periods for the application user
        * start_date inclusive
        * end_date exclusive or null when the user still lives there
    """
    __tablename__ = "user_country"
    # ── surrogate PK ─────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # ── dimensions ───────────────────────────────────────────────────────────
    country_code: Mapped[str] = mapped_column(String(2), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # ── constraints / quality guards ─────────────────────────────────────────
    __table_args__ = (
        # only *one* row per (country, start_date)
        UniqueConstraint(
            "country_code", "start_date",
            name="uq_usercountry_start",
        ),
        CheckConstraint(
            "(end_date IS NULL) OR (end_date > start_date)",
            name="ck_usercountry_period_positive",
        ),
    )

    # helpers ---------------------------------------------------------------
    @property
    def is_current(self) -> bool:
        """Returns true when end_date is null"""
        return self.end_date is None
