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

