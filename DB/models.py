from __future__ import annotations
from datetime import date, datetime, UTC
from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, func, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import Optional


class Base(DeclarativeBase):
    """Shared base class – keeps metadata in one place."""
    pass


class ArtistVariantsCanonized(Base):
    __tablename__ = "artist_variants_canonized"
    artist_variants: Mapped[str] = mapped_column(String(255), primary_key=True)
    to_link: Mapped[bool] = mapped_column(Boolean, nullable=True)
    canonical_name: Mapped[str] = mapped_column(String(255))
    comment: Mapped[Optional[str]] = mapped_column(String(750))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class Scrobble(Base):
    __tablename__ = "scrobbles"
    __table_args__ = (
        UniqueConstraint(
            "artist_name", "track_title", "play_time",
            name="uq_scrobble"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    artist_name: Mapped[str] = mapped_column(String(350))
    album_title: Mapped[str] = mapped_column(String(350))
    track_title: Mapped[str] = mapped_column(String(350))
    play_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True)


class ArtistCountry(Base):
    __tablename__ = "artistcountry"
    __table_args__ = (
        UniqueConstraint("id", name="uq_artistcountry"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    artist_name: Mapped[str] = mapped_column(String(750), index=True)
    mbid: Mapped[Optional[str]] = mapped_column(String(36), unique=True, nullable=True)
    disambiguation_comment: Mapped[Optional[str]] = mapped_column(String(558))
    country: Mapped[Optional[str]] = mapped_column(String(255))


class UserCountry(Base):
    """
    One row = one continuous period in which *the* user lived in `country_name`.
    * `start_date` inclusive
    * `end_date`   exclusive (NULL ⇒ “still lives there”)
    """
    __tablename__ = "user_country"
    # ── surrogate PK ─────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # ── dimensions ───────────────────────────────────────────────────────────
    country_name: Mapped[str] = mapped_column(String(255), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    # ── constraints / quality guards ─────────────────────────────────────────
    __table_args__ = (
        # only *one* row per (country, start_date)
        UniqueConstraint(
            "country_name", "start_date",
            name="uq_usercountry_start",
        ),
        CheckConstraint(
            "(end_date IS NULL) OR (end_date > start_date)",
            name="ck_usercountry_period_positive",
        ),
    )

    # helper ---------------------------------------------------------------
    @property
    def is_current(self) -> bool:
        return self.end_date is None
