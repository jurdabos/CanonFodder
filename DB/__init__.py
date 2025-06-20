"""
Executed on import DB — guarantees the folder exists, builds the SQLAlchemy
engine once, and exposes `SessionLocal` and `get_engine` to the rest of the
code-base.
"""
from __future__ import annotations
# Picking backend:  default mysql when DB_URL starts with mysql+, else sqlite
import os
from urllib.parse import urlparse
from .common import DB_URL
if urlparse(DB_URL).scheme.startswith("mysql"):
    from .mysql import engine, SessionLocal
else:
    from .sqlite import engine, SessionLocal
from .models import Base
# In case we need a quick schema without Alembic, guard it:
if os.environ.get("CF_CREATE_SCHEMA") == "yes":
    Base.metadata.create_all(engine)


def get_engine():
    """Returns the singleton SQLAlchemy engine"""
    return engine


def get_session():
    """Returns a new `SessionLocal` instance so callers can write
        with DB.get_session() as s: for example
            …
    """
    return SessionLocal()


# Make sure this is visible even if print output is being redirected
import sys
sys.stderr.write(f"[DB] using {engine.url}\n")
sys.stderr.flush()

# Enable SQL statement logging with proper formatting
from helpers.formatting import format_sql_for_display
import logging

# Configure SQLAlchemy logging
sql_logger = logging.getLogger('sqlalchemy.engine')
sql_logger.setLevel(logging.INFO)

# Create a custom handler that formats SQL statements
class SQLFormattingHandler(logging.StreamHandler):
    def emit(self, record):
        if hasattr(record, 'statement'):
            # Format SQL statements using our custom formatter
            record.msg = format_sql_for_display(record.msg)
        super().emit(record)

# Add the custom handler to the logger
handler = SQLFormattingHandler()
handler.setLevel(logging.INFO)
sql_logger.addHandler(handler)
