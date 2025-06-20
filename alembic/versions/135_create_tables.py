"""create tables

Revision ID: 135
Revises: 
Create Date: 2025-06-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Boolean, CheckConstraint, Column, Date, DateTime, func, Integer, String, Text, UniqueConstraint


# revision identifiers, used by Alembic.
revision: str = '135'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Create artist_info table
    op.create_table(
        'artist_info',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('artist_name', sa.String(750), index=True),
        sa.Column('mbid', sa.String(36), unique=True, nullable=True),
        sa.Column('disambiguation_comment', sa.String(558)),
        sa.Column('aliases', sa.Text(), nullable=True),
        sa.Column('country', sa.String(255), nullable=True),
        sa.UniqueConstraint('id', name='uq_artist_info')
    )

    # Create artist_variants_canonized table
    op.create_table(
        'artist_variants_canonized',
        sa.Column('artist_variants_hash', sa.String(64), primary_key=True),
        sa.Column('artist_variants_text', sa.Text()),
        sa.Column('canonical_name', sa.String(255)),
        sa.Column('to_link', sa.Boolean(), nullable=True),
        sa.Column('comment', sa.String(750), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Create ascii_chars table
    op.create_table(
        'ascii_chars',
        sa.Column('ascii_code', sa.Integer(), primary_key=True),
        sa.Column('ascii_char', sa.String(1), unique=True, nullable=False)
    )

    # Create country_code table
    op.create_table(
        'country_code',
        sa.Column('iso2', sa.String(2), primary_key=True),
        sa.Column('iso3', sa.String(3), unique=True),
        sa.Column('en_name', sa.String(350), unique=True, nullable=False),
        sa.Column('hu_name', sa.String(350), unique=True, nullable=True)
    )

    # Create scrobble table
    op.create_table(
        'scrobble',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('artist_name', sa.String(350)),
        sa.Column('album_title', sa.String(350)),
        sa.Column('track_title', sa.String(350)),
        sa.Column('play_time', sa.DateTime(timezone=True), index=True),
        sa.UniqueConstraint('artist_name', 'track_title', 'play_time', name='uq_scrobble')
    )

    # Create user_country table
    op.create_table(
        'user_country',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('country_code', sa.String(2), nullable=False),
        sa.Column('start_date', sa.Date(), nullable=False),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.UniqueConstraint('country_code', 'start_date', name='uq_usercountry_start'),
        sa.CheckConstraint('(end_date IS NULL) OR (end_date > start_date)', name='ck_usercountry_period_positive')
    )


def downgrade():
    op.drop_table('user_country')
    op.drop_table('scrobble')
    op.drop_table('country_code')
    op.drop_table('ascii_chars')
    op.drop_table('artist_variants_canonized')
    op.drop_table('artist_info')