"""add new name for scrobble table

Revision ID: 138
Revises: 137
Create Date: 2025-05-12 17:50:38.028602

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '138'
down_revision: str = "137"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.rename_table("scrobbles", "scrobble")


def downgrade():
    op.rename_table("scrobble", "scrobbles")
