"""Create all tables on the selected backend."""
from __future__ import annotations

# 0) make .env vars visible *before* importing DB code
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
from DB.mysql import get_mysql_engine
from DB.sqlite import get_sqlite_engine
from DB.models import Base


def main(which: str = "sqlite") -> None:
    engine, _ = (
        get_mysql_engine() if which.lower() == "mysql"
        else get_sqlite_engine()
    )
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sqlite")
