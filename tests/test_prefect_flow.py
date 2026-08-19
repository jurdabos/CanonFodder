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


class TestJanitorTaskPaths:
    """Tests janitor_zombie_runs_task success and empty paths with a mocked client."""

    @staticmethod
    def _ctx():
        """Builds a fake run context exposing the current flow-run ID."""
        return SimpleNamespace(flow_run=SimpleNamespace(id="self-id"))

    def test_crashes_and_counts(self):
        """Crashes the zombies the helper reports and returns their count."""
        from flows.cf_ingest import janitor_zombie_runs_task

        client = MagicMock()
        cm = MagicMock()
        cm.__enter__.return_value = client
        with (
            patch("prefect.context.get_run_context", return_value=self._ctx()),
            patch("prefect.client.orchestration.SyncPrefectClient", return_value=cm),
            patch("flows.cf_ingest.crash_zombie_runs", return_value=["z1", "z2"]) as mock_crash,
        ):
            assert janitor_zombie_runs_task.fn() == 2
        mock_crash.assert_called_once_with(client, current_run_id="self-id")

    def test_no_zombies_returns_zero(self):
        """Returns 0 and logs the info line when nothing is stale."""
        from flows.cf_ingest import janitor_zombie_runs_task

        cm = MagicMock()
        cm.__enter__.return_value = MagicMock()
        with (
            patch("prefect.context.get_run_context", return_value=self._ctx()),
            patch("prefect.client.orchestration.SyncPrefectClient", return_value=cm),
            patch("flows.cf_ingest.crash_zombie_runs", return_value=[]),
        ):
            assert janitor_zombie_runs_task.fn() == 0


class TestFetchScrobblesQABranches:
    """Tests the QA passed/failed logging branches of fetch_scrobbles."""

    @patch("corefunc.workflow.run_data_gathering_workflow", return_value=3)
    @patch("corefunc.qa.qa_lb_ingest")
    def test_qa_passed(self, mock_qa, mock_gather):
        """Logs the pass line when QA succeeds."""
        from flows.cf_ingest import fetch_scrobbles

        mock_qa.return_value = {"status": "ok", "passed": True, "row_count": 3}
        assert fetch_scrobbles.fn("testuser") == 3

    @patch("corefunc.workflow.run_data_gathering_workflow", return_value=3)
    @patch("corefunc.qa.qa_lb_ingest")
    def test_qa_failed_logs_all_issue_kinds(self, mock_qa, mock_gather):
        """Exercises schema/timestamp/duplicate/encoding warning branches."""
        from flows.cf_ingest import fetch_scrobbles

        mock_qa.return_value = {
            "status": "ok",
            "passed": False,
            "row_count": 3,
            "schema": {"pass": False, "missing": ["artist_mbid"], "unexpected": ["extra"]},
            "timestamps": {"issues": ["future timestamp"]},
            "duplicates": {"pass": False, "duplicate_count": 2, "duplicate_pct": 66.7},
            "encoding": {"pass": False, "bad_char_rows": 1},
        }
        assert fetch_scrobbles.fn("testuser") == 3


class TestFixEncodingTask:
    """Tests the fix_encoding task wrapper."""

    @patch("corefunc.data_cleaning.fix_encoding", return_value={"scrobble": (2, 10), "artist_info": (0, 5)})
    def test_returns_results(self, mock_fix):
        """Returns the per-file repair counts unchanged."""
        from flows.cf_ingest import fix_encoding_task

        assert fix_encoding_task.fn() == {"scrobble": (2, 10), "artist_info": (0, 5)}


class TestPropagateAvcTask:
    """Tests the propagate_avc task wrapper."""

    @patch("corefunc.canon.workflow.propagate_avc", return_value={"updated": 5, "aliases_added": 1})
    def test_returns_summary(self, mock_prop):
        """Returns the propagation summary unchanged."""
        from flows.cf_ingest import propagate_avc_task

        assert propagate_avc_task.fn() == {"updated": 5, "aliases_added": 1}


class TestAugmentGsTask:
    """Tests both paths of the gold-standard augmentation task."""

    @patch("corefunc.mb_local.check_local_mb", return_value=False)
    def test_skips_when_mirror_unreachable(self, mock_mb):
        """Skips gracefully without the local MB mirror."""
        from flows.cf_ingest import augment_gs_task

        assert augment_gs_task.fn() == {"rows_written": 0, "skipped": True}

    @patch("corefunc.mb_local.check_local_mb", return_value=True)
    @patch("corefunc.canon.augment.augment_gold_standard", return_value=9998)
    def test_writes_rows_when_mirror_up(self, mock_augment, mock_mb):
        """Reports the written row count when the mirror is reachable."""
        from flows.cf_ingest import augment_gs_task

        assert augment_gs_task.fn() == {"rows_written": 9998, "skipped": False}


class TestRetrainModelTask:
    """Tests the skip and success paths of the retrain task."""

    @patch("helpers.io.read_parquet", return_value=None)
    def test_skips_without_avc(self, mock_read):
        """Skips when the AVC parquet is missing."""
        from flows.cf_ingest import retrain_model_task

        assert retrain_model_task.fn() == {"models_trained": 0, "skipped": True}

    @patch("helpers.io.read_parquet")
    def test_skips_below_decided_threshold(self, mock_read):
        """Skips with fewer than 20 decided AVC rows."""
        import pandas as pd

        from flows.cf_ingest import retrain_model_task

        mock_read.return_value = pd.DataFrame({"to_link": [None] * 5})
        assert retrain_model_task.fn() == {"models_trained": 0, "skipped": True}

    @patch("corefunc.canon.tuner.save_best_historical_models", return_value={"lightgbm": "ML/lightgbm_best.pkl"})
    @patch("corefunc.canon.trainer.run_training", return_value=[{"model": "lgbm"}])
    @patch("helpers.io.read_parquet")
    def test_trains_and_exports(self, mock_read, mock_train, mock_save):
        """Trains and exports pickles when enough decided rows exist."""
        import pandas as pd

        from flows.cf_ingest import retrain_model_task

        mock_read.return_value = pd.DataFrame({"to_link": [True] * 25})
        assert retrain_model_task.fn() == {"models_trained": 1, "skipped": False}
        mock_train.assert_called_once_with(run_name="flow_retrain")
        mock_save.assert_called_once_with()


class TestCanoniseBatchTask:
    """Tests the model-missing, no-candidates, and candidates paths."""

    @patch("helpers.inference.load_model", side_effect=FileNotFoundError("no pickle"))
    def test_skips_when_model_missing(self, mock_load):
        """Skips gracefully without the model pickle."""
        from flows.cf_ingest import canonise_batch

        assert canonise_batch.fn() == {"flagged_for_review": 0, "skipped": 0}

    @patch("corefunc.canon.workflow.discover_candidates", return_value=[])
    @patch("helpers.inference.load_model")
    def test_no_candidates(self, mock_load, mock_discover):
        """Reports zeroes when discovery finds nothing."""
        from flows.cf_ingest import canonise_batch

        mock_load.return_value = MagicMock(feature_names_in_=[f"f{i}" for i in range(46)])
        assert canonise_batch.fn() == {"flagged_for_review": 0, "skipped": 0}

    @patch("corefunc.canon.workflow.write_new_candidates", return_value=2)
    @patch("corefunc.canon.workflow.discover_candidates", return_value=[{"a": 1}])
    @patch("helpers.inference.load_model")
    def test_flags_candidates(self, mock_load, mock_discover, mock_write):
        """Flags the written candidate groups for review."""
        from flows.cf_ingest import canonise_batch

        mock_load.return_value = MagicMock(feature_names_in_=[f"f{i}" for i in range(46)])
        assert canonise_batch.fn() == {"flagged_for_review": 2, "skipped": 0}


class TestFlowUserValidation:
    """Tests the username env-var guards of weekly_ingest_flow."""

    def test_requires_lastfm_user(self, monkeypatch):
        """Raises ValueError when LASTFM_USER is unset for lastfm."""
        from flows.cf_ingest import weekly_ingest_flow

        monkeypatch.delenv("LASTFM_USER", raising=False)
        with pytest.raises(ValueError, match="LASTFM_USER"):
            weekly_ingest_flow.fn(source="lastfm")

    def test_requires_lb_user(self, monkeypatch):
        """Raises ValueError when LB_USER is unset for listenbrainz."""
        from flows.cf_ingest import weekly_ingest_flow

        monkeypatch.delenv("LB_USER", raising=False)
        with pytest.raises(ValueError, match="LB_USER"):
            weekly_ingest_flow.fn(source="listenbrainz")
