"""
Integration tests for corefunc.workflow (scrobble ingestion pipeline).
"""
import pandas as pd
import pytest
from unittest.mock import patch


class TestRunDataGatheringWorkflow:
    """Tests run_data_gathering_workflow end-to-end."""

    @patch("corefunc.workflow.lfAPI.sync_user_country")
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    def test_ingests_scrobbles(
        self, mock_fetch, mock_country, tmp_pq_dir, sample_scrobble_df,
    ):
        """Fetches, normalises, and persists scrobbles to Parquet."""
        mock_fetch.return_value = sample_scrobble_df
        mock_country.return_value = False
        import helpers.io as io_mod
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("testuser")
        assert n == 3
        assert io_mod.SCROBBLE_PQ.exists()
        loaded = pd.read_parquet(io_mod.SCROBBLE_PQ)
        assert len(loaded) == 3

    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    def test_returns_zero_when_empty(self, mock_fetch, tmp_pq_dir):
        """Returns 0 when the API returns no new scrobbles."""
        mock_fetch.return_value = pd.DataFrame()
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("testuser")
        assert n == 0

    @patch("corefunc.workflow.lfAPI.sync_user_country")
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    def test_full_flag_ignores_existing(
        self, mock_fetch, mock_country, tmp_pq_dir, sample_scrobble_df,
    ):
        """With full=True, latest_scrobble_ts is bypassed (since=None)."""
        mock_fetch.return_value = sample_scrobble_df
        mock_country.return_value = False
        from corefunc.workflow import run_data_gathering_workflow
        run_data_gathering_workflow("testuser", full=True)
        mock_fetch.assert_called_once_with("testuser", None)


class TestRunDataGatheringWorkflowListenBrainz:
    """Tests the ListenBrainz path in run_data_gathering_workflow."""

    @patch("HTTP.lblink.fetch_scrobbles_since")
    def test_lb_ingests_scrobbles(
        self, mock_fetch, tmp_pq_dir, sample_scrobble_df,
    ):
        """Fetches and persists scrobbles from ListenBrainz."""
        mock_fetch.return_value = sample_scrobble_df
        import helpers.io as io_mod
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("lbuser", source="listenbrainz")
        assert n == 3
        assert io_mod.SCROBBLE_PQ.exists()

    @patch("HTTP.lblink.fetch_scrobbles_since")
    def test_lb_skips_country_sync(
        self, mock_fetch, tmp_pq_dir, sample_scrobble_df,
    ):
        """Does not call sync_user_country for ListenBrainz source."""
        mock_fetch.return_value = sample_scrobble_df
        from corefunc.workflow import run_data_gathering_workflow
        with patch("corefunc.workflow.lfAPI.sync_user_country") as mock_country:
            run_data_gathering_workflow("lbuser", source="listenbrainz")
            mock_country.assert_not_called()

    @patch("HTTP.lblink.fetch_scrobbles_since")
    def test_lb_empty_result(self, mock_fetch, tmp_pq_dir):
        """Returns 0 when ListenBrainz returns no listens."""
        mock_fetch.return_value = pd.DataFrame()
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("lbuser", source="listenbrainz")
        assert n == 0
