"""
Unit tests for corefunc.mb_local (local MusicBrainz mirror enrichment).

All subprocess calls are mocked so Docker is not required in CI.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from corefunc.mb_local import (
    ARTIST_INFO_COLS,
    _escape_pg,
    _psql_csv,
    check_local_mb,
    enrich_from_local_mb,
)


# ── _escape_pg ────────────────────────────────────────────────────────────────
class TestEscapePg:
    """Tests PostgreSQL literal escaping."""

    def test_plain_string(self):
        """Leaves a plain string unchanged."""
        assert _escape_pg("Animal Collective") == "Animal Collective"

    def test_single_quote(self):
        """Doubles single quotes."""
        assert _escape_pg("Guns N' Roses") == "Guns N'' Roses"

    def test_multiple_quotes(self):
        """Handles multiple single quotes."""
        assert _escape_pg("'hello' 'world'") == "''hello'' ''world''"


# ── check_local_mb ────────────────────────────────────────────────────────────
class TestCheckLocalMb:
    """Tests the connectivity check."""

    @patch("corefunc.mb_local.subprocess.run")
    def test_reachable(self, mock_run):
        """Returns True when psql SELECT 1 succeeds."""
        mock_run.return_value = MagicMock(returncode=0)
        assert check_local_mb() is True

    @patch("corefunc.mb_local.subprocess.run")
    def test_unreachable(self, mock_run):
        """Returns False when psql fails."""
        mock_run.return_value = MagicMock(returncode=1)
        assert check_local_mb() is False

    @patch("corefunc.mb_local.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10))
    def test_timeout(self, mock_run):
        """Returns False on timeout."""
        assert check_local_mb() is False

    @patch("corefunc.mb_local.subprocess.run", side_effect=FileNotFoundError)
    def test_docker_not_installed(self, mock_run):
        """Returns False when docker binary is missing."""
        assert check_local_mb() is False


# ── _psql_csv ─────────────────────────────────────────────────────────────────
class TestPsqlCsv:
    """Tests the low-level CSV query runner."""

    @patch("corefunc.mb_local.subprocess.run")
    def test_parses_csv(self, mock_run):
        """Parses a simple CSV output from psql."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="artist_name,mbid,country,disambiguation_comment,aliases\nTestBand,abc-123,US,rock,alias1\n",
        )
        df = _psql_csv("SELECT 1")
        assert len(df) == 1
        assert df.iloc[0]["artist_name"] == "TestBand"
        assert df.iloc[0]["country"] == "US"

    @patch("corefunc.mb_local.subprocess.run")
    def test_empty_result(self, mock_run):
        """Returns an empty DataFrame when psql produces no output."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        df = _psql_csv("SELECT 1")
        assert df.empty
        assert list(df.columns) == ARTIST_INFO_COLS

    @patch("corefunc.mb_local.subprocess.run")
    def test_error_raises(self, mock_run):
        """Raises RuntimeError when psql returns non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stderr="connection refused")
        with pytest.raises(RuntimeError, match="psql error"):
            _psql_csv("SELECT 1")


# ── enrich_from_local_mb ──────────────────────────────────────────────────────
class TestEnrichFromLocalMb:
    """Tests the main orchestrator with mocked queries."""

    @patch("corefunc.mb_local.check_local_mb", return_value=False)
    def test_unreachable_raises(self, mock_check, tmp_pq_dir):
        """Raises RuntimeError when the local mirror is unreachable."""
        with pytest.raises(RuntimeError, match="Cannot reach"):
            enrich_from_local_mb()

    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    def test_no_scrobbles(self, mock_check, tmp_pq_dir):
        """Returns 0 when no scrobbles exist."""
        assert enrich_from_local_mb() == 0

    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    def test_all_known(self, mock_check, tmp_pq_dir):
        """Returns 0 when all artists are already enriched."""
        import helpers.io as io_mod

        scrobbles = pd.DataFrame(
            {
                "artist_name": ["A"],
                "album_title": ["Al"],
                "track_title": ["T"],
                "artist_mbid": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
                "play_time": pd.to_datetime(["2024-01-01"], utc=True),
            }
        )
        artist_info = pd.DataFrame(
            {
                "artist_name": ["A"],
                "mbid": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
                "country": ["DE"],
                "disambiguation_comment": [""],
                "aliases": [""],
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        artist_info.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        assert enrich_from_local_mb() == 0

    @patch("corefunc.mb_local._enrich_by_name")
    @patch("corefunc.mb_local._enrich_by_mbid")
    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    def test_tier1_mbid_lookup(self, mock_check, mock_mbid, mock_name, tmp_pq_dir):
        """Resolves artists via MBID lookup (Tier 1)."""
        import helpers.io as io_mod

        mbid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        scrobbles = pd.DataFrame(
            {
                "artist_name": ["TestBand"],
                "album_title": ["Al"],
                "track_title": ["T"],
                "artist_mbid": [mbid],
                "play_time": pd.to_datetime(["2024-01-01"], utc=True),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        mock_mbid.return_value = pd.DataFrame(
            {
                "artist_name": ["TestBand"],
                "mbid": [mbid],
                "country": ["GB"],
                "disambiguation_comment": ["punk"],
                "aliases": ["TB,Test Band"],
            }
        )
        mock_name.return_value = pd.DataFrame(columns=ARTIST_INFO_COLS)
        n = enrich_from_local_mb(rebuild=True)
        assert n == 1
        mock_mbid.assert_called_once()
        result = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert result.iloc[0]["country"] == "GB"
        assert result.iloc[0]["aliases"] == "TB,Test Band"

    @patch("corefunc.mb_local._enrich_by_name")
    @patch("corefunc.mb_local._enrich_by_mbid")
    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    def test_tier2_name_lookup(self, mock_check, mock_mbid, mock_name, tmp_pq_dir):
        """Resolves artists via name lookup (Tier 2) when no MBID is available."""
        import helpers.io as io_mod

        scrobbles = pd.DataFrame(
            {
                "artist_name": ["NoBand"],
                "album_title": ["Al"],
                "track_title": ["T"],
                "artist_mbid": [None],
                "play_time": pd.to_datetime(["2024-01-01"], utc=True),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        mock_name.return_value = pd.DataFrame(
            {
                "artist_name": ["NoBand"],
                "mbid": ["found-mbid-123456789012345678"],
                "country": ["SE"],
                "disambiguation_comment": ["metal"],
                "aliases": [""],
            }
        )
        n = enrich_from_local_mb(rebuild=True)
        assert n == 1
        mock_name.assert_called_once()
        result = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert result.iloc[0]["country"] == "SE"

    @patch("corefunc.mb_local._enrich_by_name")
    @patch("corefunc.mb_local._enrich_by_mbid")
    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    def test_stub_rows_for_unresolved(self, mock_check, mock_mbid, mock_name, tmp_pq_dir):
        """Creates stub rows for artists not found in MB at all."""
        import helpers.io as io_mod

        scrobbles = pd.DataFrame(
            {
                "artist_name": ["Unknown, Artist"],
                "album_title": ["Al"],
                "track_title": ["T"],
                "artist_mbid": [None],
                "play_time": pd.to_datetime(["2024-01-01"], utc=True),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        mock_name.return_value = pd.DataFrame(columns=ARTIST_INFO_COLS)
        n = enrich_from_local_mb(rebuild=True)
        assert n == 1
        result = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert result.iloc[0]["artist_name"] == "Unknown, Artist"
        assert result.iloc[0]["mbid"] == ""

    @patch("corefunc.mb_local._enrich_by_name")
    @patch("corefunc.mb_local._enrich_by_mbid")
    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    def test_rebuild_overwrites(self, mock_check, mock_mbid, mock_name, tmp_pq_dir):
        """With rebuild=True, overwrites existing artist_info.parquet."""
        import helpers.io as io_mod

        # Pre-existing artist_info with one row
        old = pd.DataFrame(
            {
                "artist_name": ["OldBand"],
                "mbid": ["old-mbid"],
                "country": ["JP"],
                "disambiguation_comment": [""],
                "aliases": [""],
            }
        )
        old.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        scrobbles = pd.DataFrame(
            {
                "artist_name": ["NewBand"],
                "album_title": ["Al"],
                "track_title": ["T"],
                "artist_mbid": [None],
                "play_time": pd.to_datetime(["2024-01-01"], utc=True),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        mock_name.return_value = pd.DataFrame(columns=ARTIST_INFO_COLS)
        enrich_from_local_mb(rebuild=True)
        result = pd.read_parquet(io_mod.ARTIST_INFO_PQ)
        # Rebuild should only contain the new artist, not OldBand
        assert len(result) == 1
        assert result.iloc[0]["artist_name"] == "NewBand"
