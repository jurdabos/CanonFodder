from __future__ import annotations
import os
from sqlalchemy import create_engine
from pathlib import Path
from .common import make_sessionmaker

db_file = Path(__file__).with_name("canonfodder.db")
engine = create_engine(f"sqlite:///{db_file}", future=True, echo=False)
SessionLocal = make_sessionmaker(engine)


def get_session():
    return SessionLocal()


def get_sqlite_engine(echo: bool = False):
    db_path = Path(__file__).with_name("canonfodder.db")
    eng = create_engine(f"sqlite:///{db_path}", future=True, echo=echo)
    return eng, make_sessionmaker(eng)
