"""rename artistcountry to artist_info and add aliases column

Revision ID: 141
Revises: 140
Create Date: 2025-05-19 17:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import Text


# revision identifiers, used by Alembic.
revision: str = '141'
down_revision: str = "140"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Rename the table
    op.rename_table("artistcountry", "artist_info")
    
    # Rename unique constraint
    op.drop_constraint("uq_artistcountry", "artist_info", type_="unique")
    op.create_unique_constraint("uq_artist_info", "artist_info", ["id"])
    
    # Add the aliases column
    op.add_column("artist_info", sa.Column("aliases", Text, nullable=True))


def downgrade():
    # Remove the aliases column
    op.drop_column("artist_info", "aliases")
    
    # Rename unique constraint back
    op.drop_constraint("uq_artist_info", "artist_info", type_="unique")
    op.create_unique_constraint("uq_artistcountry", "artist_info", ["id"])
    
    # Rename the table back
    op.rename_table("artist_info", "artistcountry")
