"""Add country_code lookup table and shorten user_country.country_code

Revision ID: 139
Revises: 138
Create Date: 2025-05-12 18:34:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# ---- revision identifiers ---------------------------------------------------
revision: str = "139"
down_revision: str | None = "138"
branch_labels: tuple[str] | None = None
depends_on: tuple[str] | None = None
# -----------------------------------------------------------------------------

# ---- helpers (reuse the same column specs both directions) ------------------
_iso2 = sa.String(length=2)
_iso3 = sa.String(length=3)
_name = sa.String(length=350)


# ---- upgrade ----------------------------------------------------------------
def upgrade() -> None:
    # 1. create lookup table ---------------------------------------------------
    op.create_table(
        "country_code",
        sa.Column("iso2", _iso2, primary_key=True, nullable=False, unique=True),
        sa.Column("iso3", _iso3, unique=True, nullable=True),
        sa.Column("en_name", _name, unique=True, nullable=False),
        sa.Column("hu_name", _name, unique=True, nullable=True),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_0900_ai_ci",
    )

    # 2. shrink the FK/lookup column in user_country ---------------------------
    op.alter_column(
        "user_country",
        "country_code",
        existing_type=sa.String(length=255),
        type_=_iso2,
        existing_nullable=False,
    )


# ---- downgrade --------------------------------------------------------------
def downgrade() -> None:
    # 1. grow the column back to 255 chars -------------------------------------
    op.alter_column(
        "user_country",
        "country_code",
        existing_type=_iso2,
        type_=sa.String(length=255),
        existing_nullable=False,
    )

    # 2. drop the lookup table -------------------------------------------------
    op.drop_table("country_code")
