"""
Executed on `import DB`
───────────────────────
* guarantees folder DB/ exists
* builds the SQLAlchemy engine once
* makes SessionLocal / get_engine available to the rest of the code-base
"""
from __future__ import annotations
# 1) pick backend:  default mysql when DB_URL starts with mysql+, else sqlite
from urllib.parse import urlparse
from .common import DB_URL
if urlparse(DB_URL).scheme.startswith("mysql"):
    from .mysql import engine, SessionLocal
else:
    from .sqlite import engine, SessionLocal


def get_engine():
    return engine


def get_session():
    """Handy context helper:
        with DB.get_session() as s:
            ...
    """
    return SessionLocal()


print(f"[DB] using {engine.url}")
