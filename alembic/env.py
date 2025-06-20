from logging.config import fileConfig

"""Alembic run-environment"""
from pathlib import Path
from alembic import context
import os

# ─── 1  .env + optional -x db_url override ───────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_suffix(".env").parent.parent / ".env")
db_url = context.get_x_argument(as_dictionary=True).get("db_url")
if db_url:
    os.environ["DB_URL"] = db_url

# ─── 2  ML / engine --------------------------------------------------------
from DB.models import Base
from DB import get_engine                  # returns a *single* Engine
engine = get_engine()                      # honours DB_URL

# ─── 3  Alembic bookkeeping ----------------------------------------------------
# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=str(engine.url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,      # detect column-type changes too
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
