"""
Defines all SQLAlchemy ORM ML used by CanonFodder.

This file includes a compatibility layer to support both SQLAlchemy 1.4.x (required by Airflow)
and SQLAlchemy 2.0 (used by the core application). The compatibility layer provides:

1. A common Base class that works with both versions
2. A Mapped type annotation for type hints
3. A mapped_column function that works in both versions

When running with Airflow, SQLAlchemy 1.4.x is used, and the compatibility layer provides
implementations of DeclarativeBase, Mapped, and mapped_column that mimic SQLAlchemy 2.0's API.
"""
from __future__ import annotations
from datetime import date, datetime
from typing import Optional, Any, TypeVar, Generic, Type, get_type_hints, get_origin, get_args
from sqlalchemy import Boolean, CheckConstraint, Column, Date, DateTime, func, Integer, String, Text, UniqueConstraint

# Handle SQLAlchemy version compatibility
try:
    # SQLAlchemy 2.0 style
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    class Base(DeclarativeBase):
        """Serves as the common declarative base so metadata stays centralised"""
        pass
except ImportError:
    # SQLAlchemy 1.4 style (for Airflow compatibility)
    from sqlalchemy.orm import declarative_base

    Base = declarative_base()

    # Add a docstring to maintain documentation
    Base.__doc__ = "Serves as the common declarative base so metadata stays centralised"

    # Create compatibility layer for Mapped and mapped_column
    T = TypeVar('T')

    class Mapped(Generic[T]):
        """Compatibility class for SQLAlchemy 1.4 to mimic SQLAlchemy 2.0's Mapped type"""
        pass

    def mapped_column(*args, **kwargs):
        """
        Compatibility function for SQLAlchemy 1.4 to mimic SQLAlchemy 2.0's mapped_column

        This simply passes through to Column for SQLAlchemy 1.4
        """
        return Column(*args, **kwargs)


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
