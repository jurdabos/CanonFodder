"""
Creates the MySQL production engine and exposes SessionLocal.
"""
from .common import DB_URL, make_sessionmaker
from sqlalchemy import create_engine

engine = create_engine(DB_URL, future=True, pool_recycle=3600, echo=True)
SessionLocal = make_sessionmaker(engine)          # exported
