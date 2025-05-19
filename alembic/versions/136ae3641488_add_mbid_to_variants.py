"""add mbid to variants

Revision ID: 136ae3641488
Revises: 
Create Date: 2025-05-08 17:10:38.028602

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '136ae3641488'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column(
        "artist_variants_canonized",
        sa.Column("mbid", sa.String(36), nullable=True)
    )


def downgrade():
    op.drop_column("artist_variants_canonized", "mbid")
