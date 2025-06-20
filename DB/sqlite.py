"""
Creates the SQLite production engine (fallback or local use) and exposes
SessionLocal.
"""
from __future__ import annotations
from sqlalchemy import create_engine
from .common import DB_URL, make_sessionmaker
engine = create_engine(DB_URL, future=True, echo=True)
SessionLocal = make_sessionmaker(engine)
