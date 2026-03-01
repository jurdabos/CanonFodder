"""
Tests for flows/cf_ingest.py Prefect tasks and flow.

Uses unittest.mock to isolate from external dependencies (Last.fm, MusicBrainz).
Requires the prefect optional dependency.
"""
import logging
import pytest
from unittest.mock import patch
prefect = pytest.importorskip("prefect")


@pytest.fixture(autouse=True)
def _set_lastfm_user(monkeypatch):
    """Ensures LASTFM_USER is available for every test."""
    monkeypatch.setenv("LASTFM_USER", "testuser")


@pytest.fixture(autouse=True)
def _mock_prefect_logger(monkeypatch):
    """Replaces get_run_logger with a standard Python logger for tests."""
    monkeypatch.setattr(
        "flows.cf_ingest.get_run_logger",
        lambda: logging.getLogger("test.cf_ingest"),
    )


class TestFetchScrobblesTask:
    """Tests the fetch_scrobbles Prefect task."""

    @patch("corefunc.workflow.lfAPI.sync_user_country", return_value=False)
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    def test_returns_count(self, mock_fetch, mock_country, tmp_pq_dir, sample_scrobble_df):
        """Returns the number of ingested scrobbles."""
        mock_fetch.return_value = sample_scrobble_df
        from flows.cf_ingest import fetch_scrobbles
        n = fetch_scrobbles.fn("testuser")
        assert n == 3

    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    def test_returns_zero_when_empty(self, mock_fetch, tmp_pq_dir):
        """Returns 0 when the API yields nothing."""
        import pandas as pd
        mock_fetch.return_value = pd.DataFrame()
        from flows.cf_ingest import fetch_scrobbles
        n = fetch_scrobbles.fn("testuser")
        assert n == 0


class TestEnrichArtistsTask:
    """Tests the enrich_artists Prefect task."""

    @patch("HTTP.mbAPI.search_artist", return_value=[])
    def test_returns_zero_when_no_unknowns(self, mock_search, populated_pq):
        """Returns 0 when all artists are already known."""
        from flows.cf_ingest import enrich_artists
        n = enrich_artists.fn()
        assert n == 0


class TestCleanArtistsTask:
    """Tests the clean_artists Prefect task."""

    def test_deduplicates(self, populated_pq):
        """Runs deduplication and returns (removed, remaining)."""
        from flows.cf_ingest import clean_artists
        removed, remaining = clean_artists.fn()
        assert removed == 0
        assert remaining == 2


class TestWeeklyIngestFlow:
    """Tests the full weekly_ingest_flow."""

    @patch("flows.cf_ingest.retrain_model_task", return_value={"models_trained": 0, "skipped": True})
    @patch("flows.cf_ingest.augment_gs_task", return_value={"rows_written": 0, "skipped": True})
    @patch("flows.cf_ingest.propagate_avc_task", return_value={"updated": 0, "aliases_added": 0})
    @patch("flows.cf_ingest.canonise_batch", return_value={"flagged_for_review": 0, "skipped": 0})
    @patch("flows.cf_ingest.clean_artists", return_value=(0, 2))
    @patch("flows.cf_ingest.enrich_artists", return_value=0)
    @patch("flows.cf_ingest.fix_encoding_task", return_value={"scrobble": (0, 3), "artist_info": (0, 2)})
    @patch("flows.cf_ingest.fetch_scrobbles", return_value=3)
    def test_full_flow(
        self, mock_fetch, mock_enc, mock_enrich, mock_clean,
        mock_canon, mock_prop, mock_gs, mock_retrain,
        tmp_pq_dir,
    ):
        """Runs the complete flow and returns a result dict."""
        from flows.cf_ingest import weekly_ingest_flow
        result = weekly_ingest_flow.fn()
        assert result["new_scrobbles"] == 3
        assert result["encoding_fixed"] == 0
        assert result["flagged_for_review"] == 0
        assert result["cleaned"] == 0
        assert result["remaining"] == 2
        assert "duration" in result
