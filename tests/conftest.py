# tests/conftest.py
# This is just boilerplate for now. Tests are coming in CanonFodder 1.3.
import contextlib
import sys
import pathlib
from datetime import datetime, UTC

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
import shutil
import tempfile


@pytest.fixture(scope="session")
def tmp_sqlite_url():
    tmp_dir = tempfile.mkdtemp()
    db_path = pathlib.Path(tmp_dir) / "canonfodder_test.sqlite"
    url = f"sqlite:///{db_path}"
    yield url
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def session(tmp_sqlite_url, monkeypatch):
    monkeypatch.setenv("DB_URL", tmp_sqlite_url)
    sys.modules.pop("DB", None)
    from importlib import reload
    import DB
    reload(DB)  # so ML bind to new URL
    engine = DB.get_engine()
    DB.Base.metadata.create_all(engine)  # to bootstrap the tables
    Session = DB.get_session
    with Session() as sess:
        yield sess
"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Add the parent directory to sys.path so we can import the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_scrobble_df():
    """Provide a sample DataFrame with scrobbles for testing."""
    return pd.DataFrame({
        'artist_name': ['Artist1', 'Artist2', 'Artist3'],
        'artist_mbid': ['mbid1', 'mbid2', ''],
        'track_title': ['Track1', 'Track2', 'Track3'],
        'album_title': ['Album1', 'Album2', 'Album3'],
        'play_time': [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-01-02'),
            pd.Timestamp('2023-01-03')
        ]
    })


@pytest.fixture
def mock_progress_callback():
    """Provide a mock progress callback for testing."""
    callback = MagicMock()
    return callback


@pytest.fixture
def mock_session():
    """Provide a mock database session for testing."""
    mock = MagicMock()
    # Configure the mock to return itself from context manager methods
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    return mock


@pytest.fixture
def mock_engine():
    """Provide a mock SQLAlchemy engine for testing."""
    mock = MagicMock()
    # Configure the mock to handle with statement
    mock.connect.return_value.__enter__.return_value = mock.connect.return_value
    return mock

@pytest.fixture
def toy_scrobbles(session):
    from DB.models import Scrobble
    now = datetime.now(UTC)
    rows = [
        Scrobble(artist_name="Bohren & der Club of Gore",
                 album_title="", track_title="A", play_time=now),
        Scrobble(artist_name="Bohren und der Club of Gore",
                 album_title="", track_title="A", play_time=now),
        Scrobble(artist_name="Ry Cooder & V. M. Bhatt",
                 album_title="", track_title="B", play_time=now),
    ]
    session.add_all(rows)
    session.commit()


def make_groups():
    return [["Bohren & der Club of Gore", "Bohren und der Club of Gore"]]


def auto_answers(monkeypatch):
    # always pick the first variant as canonical, no comment
    monkeypatch.setattr(
        "helpers.cli.questionary.select",
        lambda *a, **k: type("Ans", (), {"ask": lambda self: k["choices"][0]})()
    )
    monkeypatch.setattr(
        "helpers.cli.questionary.text",
        lambda *a, **k: type("Ans", (), {"ask": lambda self: ""})()
    )

