# This is just boilerplate for now. Tests are coming in CanonFodder 1.3.

from conftest import auto_answers, make_groups
import contextlib
from DB.models import Scrobble
import pandas as pd


def _df_from_session(sess):
    from DB.models import Scrobble
    rows = sess.query(Scrobble.artist_name).all()
    return pd.DataFrame(rows, columns=["Artist"])


def test_unify_cli_smoketest(session, toy_scrobbles, monkeypatch):
    from helpers import cli
    from sqlalchemy.orm import sessionmaker
    # ⚠make a *factory* that yields a brand-new Session on the same engine
    SessionFactory = sessionmaker(bind=session.get_bind())
    monkeypatch.setattr(
        cli, "_get_session",
        lambda: SessionFactory()  # works as context manager
    )
    auto_answers(monkeypatch)
    # DataFrame built from the toy rows
    df = pd.DataFrame(
        session.query(Scrobble.artist_name).all(),
        columns=["Artist"]
    )
    artcount = df["Artist"].value_counts().reset_index(name="Count")
    groups = make_groups()
    df2, artcount2 = cli.unify_artist_names_cli(df, artcount, groups)
    # --- assertions ----------------------------------------------
    from DB.models import ArtistVariantsCanonized as AVC
    assert session.query(AVC).count() == 1
    row = session.query(AVC).first()
    assert row.canonical_name == "Bohren & der Club of Gore"
    assert row.to_link is True
    # canonisation really happened in the dataframe
    assert not df2["Artist"].str.contains(" und ").any()
