"""
Pytest configuration and shared Parquet-based fixtures for c9r.
"""
import pathlib
import sys
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_pq_dir(monkeypatch, tmp_path):
    """Creates a temporary PQ directory and patches all modules that import PQ path constants."""
    pq = tmp_path / "PQ"
    pq.mkdir()
    import helpers.io as io_mod
    import helpers.query as q_mod
    # Mapping of constant name → temp path
    paths = {
        "PQ_DIR": pq,
        "SCROBBLE_PQ": pq / "scrobble.parquet",
        "ARTIST_INFO_PQ": pq / "artist_info.parquet",
        "AVC_PQ": pq / "avc.parquet",
        "C_PQ": pq / "c.parquet",
        "UC_PQ": pq / "uc.parquet",
    }
    # Patching helpers.io (canonical source)
    for name, path in paths.items():
        monkeypatch.setattr(io_mod, name, path)
    # Patching every module that copies these constants at import time
    import_map = {
        q_mod: ["SCROBBLE_PQ", "ARTIST_INFO_PQ", "AVC_PQ"],
    }
    # Lazily importing optional consumer modules
    for mod_path, attrs in [
        ("corefunc.data_cleaning", ["ARTIST_INFO_PQ", "SCROBBLE_PQ"]),
        ("corefunc.enrich", ["ARTIST_INFO_PQ", "SCROBBLE_PQ"]),
        ("corefunc.canon", ["AVC_PQ"]),
        ("helpers.cli", ["PQ_DIR", "AVC_PQ", "UC_PQ"]),
        ("HTTP.lfAPI", ["C_PQ", "UC_PQ", "SCROBBLE_PQ"]),
        ("HTTP.mbAPI", ["ARTIST_INFO_PQ"]),
    ]:
        try:
            mod = __import__(mod_path, fromlist=["_"])
            import_map[mod] = attrs
        except ImportError:
            pass
    for mod, attrs in import_map.items():
        for attr in attrs:
            if attr in paths and hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, paths[attr])
    return pq


@pytest.fixture()
def sample_scrobble_df():
    """Provides a small scrobble DataFrame with UTC timestamps."""
    return pd.DataFrame({
        "artist_name": ["Bohren & der Club of Gore", "Ry Cooder", "Bohren & der Club of Gore"],
        "album_title": ["Sunset Mission", "Paris, Texas", "Sunset Mission"],
        "track_title": ["Prowler", "Paris, Texas", "Midnight Walker"],
        "artist_mbid": [
            "a4074512-87e0-4820-b609-0c4a18142a70",
            "4d6b954c-3022-4515-966e-30c3e7081bce",
            "a4074512-87e0-4820-b609-0c4a18142a70",
        ],
        "play_time": pd.to_datetime([
            "2024-01-15 20:00:00",
            "2024-01-15 20:05:00",
            "2024-01-15 20:10:00",
        ], utc=True),
    })


@pytest.fixture()
def sample_artist_info_df():
    """Provides a small artist_info DataFrame."""
    return pd.DataFrame({
        "artist_name": ["Bohren & der Club of Gore", "Ry Cooder"],
        "mbid": [
            "a4074512-87e0-4820-b609-0c4a18142a70",
            "4d6b954c-3022-4515-966e-30c3e7081bce",
        ],
        "country": ["DE", "US"],
        "disambiguation_comment": ["", ""],
        "aliases": ["", ""],
    })


@pytest.fixture()
def populated_pq(tmp_pq_dir, sample_scrobble_df, sample_artist_info_df):
    """Writes sample data into the temp PQ directory and returns the path."""
    sample_scrobble_df.to_parquet(tmp_pq_dir / "scrobble.parquet", index=False)
    sample_artist_info_df.to_parquet(tmp_pq_dir / "artist_info.parquet", index=False)
    return tmp_pq_dir

