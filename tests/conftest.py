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
        "SCROBBLE_PQ_DIR": pq / "scrobble",
        "ARTIST_INFO_PQ": pq / "artist_info.parquet",
        "AVC_PQ": pq / "avc.parquet",
        "C_PQ": pq / "c.parquet",
        "UC_PQ": pq / "uc.parquet",
        "QA_REPORT_PQ": pq / "qa_report.parquet",
        "GS_MB_PQ": pq / "gs_mb.parquet",
    }
    # Patching helpers.io (canonical source)
    for name, path in paths.items():
        monkeypatch.setattr(io_mod, name, path)
    # Patching every module that copies these constants at import time
    import_map = {
        q_mod: ["SCROBBLE_PQ", "ARTIST_INFO_PQ", "AVC_PQ", "QA_REPORT_PQ", "UC_PQ"],
    }
    # Lazily importing optional consumer modules
    for mod_path, attrs in [
        ("corefunc.data_cleaning", ["ARTIST_INFO_PQ", "SCROBBLE_PQ"]),
        ("corefunc.enrich", ["ARTIST_INFO_PQ", "SCROBBLE_PQ"]),
        ("corefunc.mb_local", ["ARTIST_INFO_PQ", "SCROBBLE_PQ"]),
        ("corefunc.canon", ["AVC_PQ"]),
        ("corefunc.canon.workflow", ["AVC_PQ", "ARTIST_INFO_PQ", "PQ_DIR", "PREDICTIONS_LOG_PQ"]),
        ("corefunc.qa", ["SCROBBLE_PQ", "ARTIST_INFO_PQ", "AVC_PQ", "UC_PQ", "GS_MB_PQ", "QA_REPORT_PQ"]),
        ("corefunc.profile", ["SCROBBLE_PQ", "ARTIST_INFO_PQ", "AVC_PQ", "C_PQ"]),
        ("helpers.cli", ["PQ_DIR", "AVC_PQ", "UC_PQ"]),
        ("HTTP.lfAPI", ["C_PQ", "UC_PQ", "SCROBBLE_PQ"]),
        ("HTTP.mbAPI", ["ARTIST_INFO_PQ"]),
        ("helpers.inference", ["PQ_DIR", "SCROBBLE_PQ"]),
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
    # Patching derived paths not in helpers.io
    pred_log_path = pq / "predictions_log.parquet"
    for mod_path in ("corefunc.canon.workflow", "corefunc.qa"):
        try:
            mod = __import__(mod_path, fromlist=["_"])
            if hasattr(mod, "PREDICTIONS_LOG_PQ"):
                monkeypatch.setattr(mod, "PREDICTIONS_LOG_PQ", pred_log_path)
        except ImportError:
            pass
    return pq


@pytest.fixture()
def sample_scrobble_df():
    """Provides a small scrobble DataFrame with UTC timestamps."""
    return pd.DataFrame(
        {
            "artist_name": ["Bohren & der Club of Gore", "Ry Cooder", "Bohren & der Club of Gore"],
            "album_title": ["Sunset Mission", "Paris, Texas", "Sunset Mission"],
            "track_title": ["Prowler", "Paris, Texas", "Midnight Walker"],
            "artist_mbid": [
                "a4074512-87e0-4820-b609-0c4a18142a70",
                "4d6b954c-3022-4515-966e-30c3e7081bce",
                "a4074512-87e0-4820-b609-0c4a18142a70",
            ],
            "play_time": pd.to_datetime(
                [
                    "2024-01-15 20:00:00",
                    "2024-01-15 20:05:00",
                    "2024-01-15 20:10:00",
                ],
                utc=True,
            ),
        }
    )


@pytest.fixture()
def sample_artist_info_df():
    """Provides a small artist_info DataFrame."""
    return pd.DataFrame(
        {
            "artist_name": ["Bohren & der Club of Gore", "Ry Cooder"],
            "mbid": [
                "a4074512-87e0-4820-b609-0c4a18142a70",
                "4d6b954c-3022-4515-966e-30c3e7081bce",
            ],
            "country": ["DE", "US"],
            "disambiguation_comment": ["", ""],
            "aliases": ["", ""],
        }
    )


@pytest.fixture()
def populated_pq(tmp_pq_dir, sample_scrobble_df, sample_artist_info_df):
    """Writes sample data into the temp PQ directory and returns the path."""
    sample_scrobble_df.to_parquet(tmp_pq_dir / "scrobble.parquet", index=False)
    sample_artist_info_df.to_parquet(tmp_pq_dir / "artist_info.parquet", index=False)
    return tmp_pq_dir


@pytest.fixture()
def temporal_scrobble_df():
    """
    Provides a multi-year, multi-month, multi-hour scrobble DataFrame.

    Spans 2023-01 through 2025-02 with varied hours and weekdays so that
    monthly_summary, yearly_top_artists, streak_analysis, and listening_clock
    all have meaningful data to exercise.
    """
    rows = []
    # 2023: Jan (3 scrobbles), Jul (2), Dec (4)
    for day, hour in [(1, 9), (2, 14), (3, 22)]:
        rows.append(("Alpha", "AlbumA", "TrackA", None, f"2023-01-{day:02d} {hour:02d}:00:00"))
    for day, hour in [(10, 11), (11, 15)]:
        rows.append(("Beta", "AlbumB", "TrackB", None, f"2023-07-{day:02d} {hour:02d}:00:00"))
    for day, hour in [(1, 20), (2, 21), (3, 23), (4, 8)]:
        rows.append(("Alpha", "AlbumA", "TrackC", None, f"2023-12-{day:02d} {hour:02d}:00:00"))
    # 2024: Jan (5), Jul (1), Dec (3)
    for day, hour in [(1, 10), (2, 10), (3, 11), (4, 12), (5, 13)]:
        rows.append(("Alpha", "AlbumA", "TrackA", None, f"2024-01-{day:02d} {hour:02d}:00:00"))
    rows.append(("Gamma", "AlbumG", "TrackG", None, "2024-07-15 16:00:00"))
    for day, hour in [(10, 19), (11, 20), (12, 21)]:
        rows.append(("Beta", "AlbumB", "TrackB", None, f"2024-12-{day:02d} {hour:02d}:00:00"))
    # 2025: Jan (2), Feb (1)
    for day, hour in [(1, 9), (2, 10)]:
        rows.append(("Alpha", "AlbumA", "TrackA", None, f"2025-01-{day:02d} {hour:02d}:00:00"))
    rows.append(("Beta", "AlbumB", "TrackD", None, "2025-02-01 14:00:00"))
    df = pd.DataFrame(rows, columns=["artist_name", "album_title", "track_title", "artist_mbid", "play_time"])
    df["play_time"] = pd.to_datetime(df["play_time"], utc=True)
    return df


@pytest.fixture()
def temporal_pq(tmp_pq_dir, temporal_scrobble_df):
    """Writes temporal scrobble data into the temp PQ directory."""
    temporal_scrobble_df.to_parquet(tmp_pq_dir / "scrobble.parquet", index=False)
    return tmp_pq_dir
