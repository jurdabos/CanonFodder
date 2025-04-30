"""
Provides shared configuration constants and factory helpers for SQLAlchemy
engines and sessions.
"""
import os
from pathlib import Path
from urllib.parse import urlparse
from sqlalchemy.orm import sessionmaker

DEFAULT_SQLITE_URL = "sqlite:///DB/canonfodder.db"
DB_URL = os.getenv("DB_URL", DEFAULT_SQLITE_URL)
FMT_FALLBACK = "%d %b %Y, %H:%i"
# Failing fast when SQLite file exists but is empty
parsed = urlparse(DB_URL)
if parsed.scheme == "sqlite":
    db_file = Path(parsed.path)
    if db_file.exists() and db_file.stat().st_size == 0:
        raise RuntimeError(f"{db_file} exists but is empty — run migrations or set CF_CREATE_SCHEMA=yes and initialize")


def make_sessionmaker(engine, *, expire_on_commit: bool = False):
    """Returns a new `sessionmaker` bound to `engine`.
    Args:
        engine: SQLAlchemy engine to bind
        expire_on_commit: if true expires objects after commit
    Returns:
        sqlalchemy.orm.sessionmaker instance
    """
    return sessionmaker(bind=engine, expire_on_commit=expire_on_commit)
