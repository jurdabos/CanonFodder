"""
Tests for flows/cf_ingest.py Prefect tasks and flow.

Uses unittest.mock to isolate from external dependencies (Last.fm, MusicBrainz).
Requires the prefect optional dependency.
"""

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


class TestJanitorZombieRuns:
    """Tests the zombie-run janitor helper, task guardrail, and flow timeout."""

    @staticmethod
    def _fake_run(run_id, name, state_name, age_hours):
        """Builds a minimal flow-run stand-in started `age_hours` ago."""
        started = datetime.now(UTC) - timedelta(hours=age_hours)
        return SimpleNamespace(
            id=run_id, name=name, state_name=state_name, start_time=started, expected_start_time=started
        )

    def test_crashes_stale_spares_fresh(self):
        """Only runs older than the threshold are marked Crashed."""
        from flows.cf_ingest import crash_zombie_runs

        client = MagicMock()
        client.read_flow_runs.return_value = [
            self._fake_run("id-stale", "stale-zombie", "Running", 48),
            self._fake_run("id-fresh", "fresh-runner", "Running", 1),
        ]
        crashed = crash_zombie_runs(client, flow_name="c9r_ingest")
        assert crashed == ["stale-zombie"]
        client.set_flow_run_state.assert_called_once()
        assert client.set_flow_run_state.call_args[0][0] == "id-stale"

    def test_never_crashes_own_run(self):
        """The calling run's own ID is exempt even when stale-looking."""
        from flows.cf_ingest import crash_zombie_runs

        client = MagicMock()
        client.read_flow_runs.return_value = [self._fake_run("id-self", "self", "Running", 48)]
        crashed = crash_zombie_runs(client, flow_name="c9r_ingest", current_run_id="id-self")
        assert crashed == []
        client.set_flow_run_state.assert_not_called()

    def test_force_retry_and_continue_on_failure(self):
        """A rejected transition retries with force=True; hard failures are skipped."""
        from prefect.exceptions import PrefectException

        from flows.cf_ingest import crash_zombie_runs

        client = MagicMock()
        client.read_flow_runs.return_value = [
            self._fake_run("id-force", "needs-force", "Pending", 72),
            self._fake_run("id-hopeless", "hopeless", "Running", 96),
        ]
        client.set_flow_run_state.side_effect = [
            PrefectException("rejected"),
            None,
            PrefectException("rejected"),
            PrefectException("still rejected"),
        ]
        crashed = crash_zombie_runs(client, flow_name="c9r_ingest")
        assert crashed == ["needs-force"]
        assert client.set_flow_run_state.call_count == 4
        assert client.set_flow_run_state.call_args_list[1].kwargs["force"] is True
        assert client.set_flow_run_state.call_args_list[3].kwargs["force"] is True

    def test_task_swallows_client_failure(self):
        """A client blow-up only logs a warning and returns 0 (guardrail)."""
        from flows.cf_ingest import janitor_zombie_runs_task

        with patch("prefect.client.orchestration.SyncPrefectClient", side_effect=RuntimeError("boom")):
            assert janitor_zombie_runs_task.fn() == 0

    def test_flow_timeout_is_set(self):
        """The flow carries the agreed 8700 s timeout."""
        from flows.cf_ingest import weekly_ingest_flow

        assert weekly_ingest_flow.timeout_seconds == 8700


class TestWeeklyIngestFlow:
    """Tests the full weekly_ingest_flow."""

    @patch("flows.cf_ingest.janitor_zombie_runs_task", return_value=0)
    @patch("flows.cf_ingest.retrain_model_task", return_value={"models_trained": 0, "skipped": True})
    @patch("flows.cf_ingest.augment_gs_task", return_value={"rows_written": 0, "skipped": True})
    @patch("flows.cf_ingest.propagate_avc_task", return_value={"updated": 0, "aliases_added": 0})
    @patch("flows.cf_ingest.canonise_batch", return_value={"flagged_for_review": 0, "skipped": 0})
    @patch("flows.cf_ingest.clean_artists", return_value=(0, 2))
    @patch("flows.cf_ingest.enrich_artists", return_value=0)
    @patch("flows.cf_ingest.fix_encoding_task", return_value={"scrobble": (0, 3), "artist_info": (0, 2)})
    @patch("flows.cf_ingest.fetch_scrobbles", return_value=3)
    def test_full_flow(
        self,
        mock_janitor,
        mock_fetch,
        mock_enc,
        mock_enrich,
        mock_clean,
        mock_canon,
        mock_prop,
        mock_gs,
        mock_retrain,
        tmp_pq_dir,
    ):
        """Runs the complete flow and returns a result dict."""
        from flows.cf_ingest import weekly_ingest_flow

        result = weekly_ingest_flow.fn()
        assert result["zombies_crashed"] == 0
        assert result["new_scrobbles"] == 3
        assert result["encoding_fixed"] == 0
        assert result["flagged_for_review"] == 0
        assert result["cleaned"] == 0
        assert result["remaining"] == 2
        assert "duration" in result
