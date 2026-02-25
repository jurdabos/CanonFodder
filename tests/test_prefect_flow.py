"""
Tests for flows/cf_ingest.py Prefect tasks and flow.

Uses unittest.mock to isolate from external dependencies (Last.fm, MusicBrainz).
Requires the prefect optional dependency.
"""
import os
import pytest
from unittest.mock import patch
prefect = pytest.importorskip("prefect")


@pytest.fixture(autouse=True)
def _set_lastfm_user(monkeypatch):
    """Ensures LASTFM_USER is available for every test."""
    monkeypatch.setenv("LASTFM_USER", "testuser")


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

    @patch("corefunc.workflow.lfAPI.sync_user_country", return_value=False)
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    @patch("HTTP.mbAPI.search_artist", return_value=[])
    def test_full_flow(self, mock_search, mock_fetch, mock_country, tmp_pq_dir, sample_scrobble_df):
        """Runs the complete flow and returns a result dict."""
        mock_fetch.return_value = sample_scrobble_df
        from flows.cf_ingest import weekly_ingest_flow
        result = weekly_ingest_flow.fn()
        assert result["new_scrobbles"] == 3
        assert "duration" in result
