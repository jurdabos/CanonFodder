"""add mbid to variants

Revision ID: 137
Revises: Torda B.
Create Date: 2025-05-08 17:50:38.028602

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '137'
down_revision: str = "136ae3641488"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column(
        "scrobbles",
        sa.Column("artist_mbid", sa.String(36), nullable=True, index=True)
    )


def downgrade():
    op.drop_column("scrobbles", "artist_mbid")
