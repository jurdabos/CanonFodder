import os
from sqlalchemy.orm import sessionmaker

DEFAULT_SQLITE_URL = "sqlite:///DB/canonfodder.db"
DB_URL = os.getenv("DB_URL", DEFAULT_SQLITE_URL)
FMT_FALLBACK = "%d %b %Y, %H:%i"


def make_sessionmaker(engine, *, expire_on_commit: bool = False):
    """Return a *new* sessionmaker bound to `engine`.
    Args:
        engine ():
        expire_on_commit ():
    """
    return sessionmaker(bind=engine, expire_on_commit=expire_on_commit)
