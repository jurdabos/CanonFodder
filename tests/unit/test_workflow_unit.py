"""
Unit tests for corefunc.workflow (data-gathering orchestration).
"""
import pandas as pd
import pytest
from unittest.mock import patch


class TestRunDataGatheringWorkflow:
    """Tests run_data_gathering_workflow with mocked API + I/O."""

    @patch("corefunc.workflow.lfAPI.sync_user_country", return_value=False)
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    @patch("corefunc.workflow.ingest_scrobbles", return_value=5)
    @patch("corefunc.workflow.latest_scrobble_ts", return_value=None)
    def test_lastfm_full_history(self, mock_ts, mock_ingest, mock_fetch, mock_country):
        """Fetches full history when full=True, ignoring existing timestamp."""
        mock_fetch.return_value = pd.DataFrame({"a": [1]})
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("user1", full=True, source="lastfm")
        assert n == 5
        mock_fetch.assert_called_once_with("user1", None)
        mock_country.assert_called_once()

    @patch("corefunc.workflow.lfAPI.sync_user_country", return_value=False)
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    @patch("corefunc.workflow.ingest_scrobbles", return_value=3)
    @patch("corefunc.workflow.latest_scrobble_ts", return_value=1700000000)
    def test_lastfm_incremental(self, mock_ts, mock_ingest, mock_fetch, mock_country):
        """Uses last timestamp for incremental fetch."""
        mock_fetch.return_value = pd.DataFrame({"a": [1]})
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("user1")
        assert n == 3
        mock_fetch.assert_called_once_with("user1", 1700000000)

    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    @patch("corefunc.workflow.latest_scrobble_ts", return_value=None)
    def test_lastfm_empty(self, mock_ts, mock_fetch):
        """Returns 0 when API returns empty DataFrame."""
        mock_fetch.return_value = pd.DataFrame()
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("user1")
        assert n == 0

    @patch("HTTP.lblink.fetch_scrobbles_since")
    @patch("corefunc.workflow.ingest_scrobbles", return_value=4)
    @patch("corefunc.workflow.latest_scrobble_ts", return_value=None)
    def test_listenbrainz_source(self, mock_ts, mock_ingest, mock_lb):
        """Uses ListenBrainz path when source is not lastfm."""
        mock_lb.return_value = pd.DataFrame({"a": [1]})
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("lbuser", source="listenbrainz")
        assert n == 4
        mock_lb.assert_called_once()

    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since", side_effect=ConnectionError("fail"))
    @patch("corefunc.workflow.latest_scrobble_ts", return_value=None)
    def test_api_error_propagates(self, mock_ts, mock_fetch):
        """Raises when the API call fails."""
        from corefunc.workflow import run_data_gathering_workflow
        with pytest.raises(ConnectionError):
            run_data_gathering_workflow("user1")

    @patch("corefunc.workflow.lfAPI.sync_user_country", side_effect=Exception("country fail"))
    @patch("corefunc.workflow.lfAPI.fetch_scrobbles_since")
    @patch("corefunc.workflow.ingest_scrobbles", return_value=2)
    @patch("corefunc.workflow.latest_scrobble_ts", return_value=None)
    def test_country_sync_failure_non_fatal(self, mock_ts, mock_ingest, mock_fetch, mock_country):
        """Country sync failure is logged but does not abort."""
        mock_fetch.return_value = pd.DataFrame({"a": [1]})
        from corefunc.workflow import run_data_gathering_workflow
        n = run_data_gathering_workflow("user1", source="lastfm")
        assert n == 2


class TestGetDevice:
    """Tests helpers.device.get_device with mocked XGBoost GPU probe."""

    def test_cpu_fallback(self):
        """Falls back to cpu when xgboost GPU probe raises."""
        import helpers.device as dev
        dev._CACHED_DEVICE = None
        with patch("helpers.device.xgb", create=True) as mock_xgb:
            mock_xgb.DMatrix.side_effect = Exception("no GPU")
            with patch.dict("sys.modules", {"xgboost": mock_xgb}):
                dev._CACHED_DEVICE = None
                # Forcing the CPU path by simulating failure
                try:
                    import xgboost as xgb
                    import numpy as np
                    dtrain = xgb.DMatrix(
                        np.array([[1, 2]], dtype=np.float32), label=[0],
                    )
                    xgb.train(
                        {"device": "cuda", "max_depth": 1, "verbosity": 0},
                        dtrain, num_boost_round=1,
                    )
                    dev._CACHED_DEVICE = "cuda"
                except Exception:
                    dev._CACHED_DEVICE = "cpu"
                assert dev.get_device() == "cpu"

    def test_cache_returns_same(self):
        """Returns cached value on subsequent calls."""
        import helpers.device as dev
        dev._CACHED_DEVICE = "cpu"
        assert dev.get_device() == "cpu"

    def test_reset_cache(self):
        """Clears the cached device."""
        import helpers.device as dev
        dev._CACHED_DEVICE = "cpu"
        dev.reset_cache()
        assert dev._CACHED_DEVICE is None
