"""Fix artist_mbid column location - add to scrobble table if missing

Revision ID: 140
Revises: 139
Create Date: 2025-05-15 14:30:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, Column, String


# ---- revision identifiers ---------------------------------------------------
revision: str = "140"
down_revision: str | None = "139"
branch_labels: tuple[str] | None = None
depends_on: tuple[str] | None = None
# -----------------------------------------------------------------------------


def upgrade() -> None:
    # Check if artist_mbid already exists in scrobble table
    conn = op.get_bind()
    inspector = inspect(conn)
    
    scrobble_columns = [col['name'] for col in inspector.get_columns('scrobble')]
    
    # Only add the column if it doesn't already exist
    if 'artist_mbid' not in scrobble_columns:
        op.add_column(
            "scrobble",
            sa.Column("artist_mbid", sa.String(36), nullable=True, index=True)
        )
        op.create_index('ix_scrobble_artist_mbid', 'scrobble', ['artist_mbid'])
    
    # Check if old "scrobbles" table exists and has data with artist_mbid values
    tables = inspector.get_table_names()
    
    if 'scrobbles' in tables:
        scrobbles_columns = [col['name'] for col in inspector.get_columns('scrobbles')]
        
        # If old table has artist_mbid column, migrate data
        if 'artist_mbid' in scrobbles_columns:
            # SQL to copy non-null artist_mbid values from scrobbles to scrobble
            op.execute("""
                UPDATE scrobble s
                JOIN scrobbles old ON s.artist_name = old.artist_name AND 
                                     s.track_title = old.track_title AND
                                     s.play_time = old.play_time
                SET s.artist_mbid = old.artist_mbid
                WHERE old.artist_mbid IS NOT NULL AND old.artist_mbid != ''
            """)


def downgrade() -> None:
    # Only drop column if it exists (safer)
    conn = op.get_bind()
    inspector = inspect(conn)
    
    scrobble_columns = [col['name'] for col in inspector.get_columns('scrobble')]
    
    if 'artist_mbid' in scrobble_columns:
        op.drop_column('scrobble', 'artist_mbid')
