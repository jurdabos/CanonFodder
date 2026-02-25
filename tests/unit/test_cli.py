"""
Unit tests for the Click CLI (main.py).
"""
import pandas as pd
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from main import cli


@pytest.fixture()
def runner():
    """Provides a Click CliRunner."""
    return CliRunner()


class TestDashboardCommand:
    """Tests the 'dashboard' command group."""

    def test_dashboard_help(self, runner):
        """Shows help when invoked without a subcommand."""
        result = runner.invoke(cli, ["dashboard"])
        assert result.exit_code == 0
        assert "artist" in result.output
        assert "album" in result.output
        assert "track" in result.output
        assert "recent" in result.output

    def test_dashboard_artist(self, runner, populated_pq):
        """Shows scrobble stats and top artists."""
        result = runner.invoke(cli, ["dashboard", "artist", "--top", "2"])
        assert result.exit_code == 0
        assert "Scrobbles:" in result.output
        assert "Unique artists:" in result.output
        assert "Bohren" in result.output

    def test_dashboard_album(self, runner, populated_pq):
        """Shows top albums with 'artist: album' format."""
        result = runner.invoke(cli, ["dashboard", "album", "-n", "2"])
        assert result.exit_code == 0
        assert "Sunset Mission" in result.output
        assert "Bohren" in result.output

    def test_dashboard_track(self, runner, populated_pq):
        """Shows top tracks with 'artist: track (album)' format."""
        result = runner.invoke(cli, ["dashboard", "track", "-n", "3"])
        assert result.exit_code == 0
        assert "Prowler" in result.output
        assert "Sunset Mission" in result.output

    def test_dashboard_recent(self, runner, populated_pq):
        """Shows most recent scrobbles with timestamps."""
        result = runner.invoke(cli, ["dashboard", "recent", "-n", "2"])
        assert result.exit_code == 0
        assert "Midnight Walker" in result.output
        assert "2024-01-15" in result.output


class TestPurgeCommand:
    """Tests the 'purge' command."""

    def test_purge_all(self, runner, populated_pq):
        """Deletes all Parquet files when --all is given."""
        from helpers.io import PQ_DIR
        assert any(PQ_DIR.glob("*.parquet"))
        result = runner.invoke(cli, ["purge", "--all", "--yes"])
        assert result.exit_code == 0
        assert not any(PQ_DIR.glob("*.parquet"))

    def test_purge_all_prompts_without_yes(self, runner, populated_pq):
        """Prompts for confirmation when --all is given without --yes."""
        result = runner.invoke(cli, ["purge", "--all"], input="y\n")
        assert result.exit_code == 0
        assert "This will delete all Parquet data files" in result.output

    def test_purge_all_aborts_on_no(self, runner, populated_pq):
        """Aborts when user declines the --all confirmation."""
        from helpers.io import PQ_DIR
        before = len(list(PQ_DIR.glob("*.parquet")))
        result = runner.invoke(cli, ["purge", "--all"], input="n\n")
        assert result.exit_code != 0
        assert len(list(PQ_DIR.glob("*.parquet"))) == before

    def test_purge_no_files(self, runner, tmp_pq_dir):
        """Reports no files found when PQ directory is empty."""
        result = runner.invoke(cli, ["purge"])
        assert result.exit_code == 0
        assert "No Parquet files found" in result.output

    def test_purge_interactive_select_all(self, runner, populated_pq):
        """Deletes all files when user confirms each one interactively."""
        from helpers.io import PQ_DIR
        count = len(list(PQ_DIR.glob("*.parquet")))
        result = runner.invoke(cli, ["purge"], input="y\n" * count)
        assert result.exit_code == 0
        assert f"Purged {count} of {count} file(s)" in result.output
        assert not any(PQ_DIR.glob("*.parquet"))

    def test_purge_interactive_skip_all(self, runner, populated_pq):
        """Skips all files when user declines each one interactively."""
        from helpers.io import PQ_DIR
        count = len(list(PQ_DIR.glob("*.parquet")))
        result = runner.invoke(cli, ["purge"], input="n\n" * count)
        assert result.exit_code == 0
        assert f"Purged 0 of {count} file(s)" in result.output
        assert len(list(PQ_DIR.glob("*.parquet"))) == count

    def test_purge_interactive_partial(self, runner, populated_pq):
        """Deletes only the files the user confirms."""
        from helpers.io import PQ_DIR
        count = len(list(PQ_DIR.glob("*.parquet")))
        assert count >= 2, "Need at least 2 files for partial test"
        # Confirming first, skipping the rest
        answers = "y\n" + "n\n" * (count - 1)
        result = runner.invoke(cli, ["purge"], input=answers)
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert "Skipped" in result.output
        assert f"Purged 1 of {count} file(s)" in result.output


class TestIngestCommand:
    """Tests the 'ingest' command."""

    @patch("HTTP.lfAPI.fetch_scrobbles_since")
    def test_ingest_no_new(self, mock_fetch, runner, tmp_pq_dir):
        """Reports no new scrobbles."""
        mock_fetch.return_value = pd.DataFrame()
        result = runner.invoke(cli, ["ingest", "--user", "testuser", "--source", "lastfm"])
        assert result.exit_code == 0
        assert "No new scrobbles" in result.output

    @patch("HTTP.lfAPI.fetch_scrobbles_since")
    def test_ingest_with_data(self, mock_fetch, runner, tmp_pq_dir, sample_scrobble_df):
        """Ingests scrobbles and reports count."""
        mock_fetch.return_value = sample_scrobble_df
        result = runner.invoke(cli, ["ingest", "--user", "testuser", "--source", "lastfm"])
        assert result.exit_code == 0
        assert "Ingested" in result.output


class TestEnrichCommand:
    """Tests the unified 'enrich' command."""

    @patch("corefunc.enrich.enrich_all")
    def test_enrich_default_local(self, mock_all, runner, tmp_pq_dir):
        """Defaults to local MB mirror backend."""
        mock_all.return_value = {"artist_info_rows": 10, "mbids_backfilled": 5}
        result = runner.invoke(cli, ["enrich"])
        assert result.exit_code == 0
        mock_all.assert_called_once_with(backend="local", rebuild=False)
        assert "local MB mirror" in result.output
        assert "10" in result.output

    @patch("corefunc.enrich.enrich_all")
    def test_enrich_mbapi_flag(self, mock_all, runner, tmp_pq_dir):
        """Selects remote MB API backend with --mbapi."""
        mock_all.return_value = {"artist_info_rows": 3, "mbids_backfilled": 1}
        result = runner.invoke(cli, ["enrich", "--mbapi"])
        assert result.exit_code == 0
        mock_all.assert_called_once_with(backend="mbapi", rebuild=False)
        assert "remote MB API" in result.output

    @patch("corefunc.enrich.enrich_all")
    def test_enrich_lastfmapi_flag(self, mock_all, runner, tmp_pq_dir):
        """Selects Last.fm + remote MB backend with --lastfmapi."""
        mock_all.return_value = {"artist_info_rows": 7, "mbids_backfilled": 2}
        result = runner.invoke(cli, ["enrich", "--lastfmapi"])
        assert result.exit_code == 0
        mock_all.assert_called_once_with(backend="lastfmapi", rebuild=False)

    def test_enrich_mutual_exclusion(self, runner, tmp_pq_dir):
        """Rejects --mbapi and --lastfmapi together."""
        result = runner.invoke(cli, ["enrich", "--mbapi", "--lastfmapi"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    @patch("corefunc.enrich.enrich_all")
    def test_enrich_rebuild(self, mock_all, runner, tmp_pq_dir):
        """Passes rebuild=True to enrich_all."""
        mock_all.return_value = {"artist_info_rows": 20, "mbids_backfilled": 0}
        result = runner.invoke(cli, ["enrich", "--rebuild"])
        assert result.exit_code == 0
        mock_all.assert_called_once_with(backend="local", rebuild=True)

    @patch("HTTP.lfAPI.sync_user_country", return_value=False)
    @patch("corefunc.enrich.enrich_all", return_value={"artist_info_rows": 0, "mbids_backfilled": 0})
    def test_enrich_country_flag(self, mock_all, mock_country, runner, tmp_pq_dir):
        """Runs country sync only when --country is passed."""
        result = runner.invoke(cli, ["enrich", "--country", "--user", "testuser", "--source", "lastfm"])
        assert result.exit_code == 0
        mock_country.assert_called_once()

    @patch("corefunc.enrich.enrich_all", return_value={"artist_info_rows": 0, "mbids_backfilled": 0})
    def test_enrich_no_country_by_default(self, mock_all, runner, tmp_pq_dir):
        """Does not run country sync by default."""
        result = runner.invoke(cli, ["enrich"])
        assert result.exit_code == 0
        assert "Country" not in result.output

    @patch("corefunc.enrich.enrich_all")
    def test_enrich_runtime_error(self, mock_all, runner, tmp_pq_dir):
        """Reports RuntimeError (e.g. local mirror unavailable) gracefully."""
        mock_all.side_effect = RuntimeError("Cannot reach local MB mirror")
        result = runner.invoke(cli, ["enrich"])
        assert result.exit_code == 0
        assert "Cannot reach local MB mirror" in result.output


class TestIngestListenBrainz:
    """Tests the 'ingest --source listenbrainz' command."""

    @patch("HTTP.lblink.fetch_scrobbles_since")
    def test_ingest_lb_no_new(self, mock_fetch, runner, tmp_pq_dir):
        """Reports no new scrobbles from ListenBrainz."""
        mock_fetch.return_value = pd.DataFrame()
        result = runner.invoke(cli, ["ingest", "--user", "lbuser", "--source", "listenbrainz"])
        assert result.exit_code == 0
        assert "No new scrobbles" in result.output

    @patch("HTTP.lblink.fetch_scrobbles_since")
    def test_ingest_lb_with_data(self, mock_fetch, runner, tmp_pq_dir, sample_scrobble_df):
        """Ingests ListenBrainz scrobbles and reports count."""
        mock_fetch.return_value = sample_scrobble_df
        result = runner.invoke(cli, ["ingest", "--user", "lbuser", "--source", "listenbrainz"])
        assert result.exit_code == 0
        assert "Ingested" in result.output

    @patch("HTTP.lblink.fetch_scrobbles_since")
    def test_ingest_lb_alias(self, mock_fetch, runner, tmp_pq_dir):
        """Accepts 'lb' as an alias for 'listenbrainz'."""
        mock_fetch.return_value = pd.DataFrame()
        result = runner.invoke(cli, ["ingest", "--user", "lbuser", "--source", "lb"])
        assert result.exit_code == 0
        assert "listenbrainz" in result.output


class TestEnrichListenBrainzCountry:
    """Tests the --country + --source listenbrainz interaction."""

    @patch("corefunc.enrich.enrich_all", return_value={"artist_info_rows": 0, "mbids_backfilled": 0})
    def test_enrich_lb_skips_country(self, mock_all, runner, tmp_pq_dir):
        """Skips country sync for ListenBrainz source."""
        result = runner.invoke(cli, ["enrich", "--country", "--source", "listenbrainz"])
        assert result.exit_code == 0
        assert "Skipping country sync" in result.output


class TestSourceHelpers:
    """Tests the _normalise_source and _resolve_user helpers."""

    def test_normalise_lb(self):
        """Normalises 'lb' to 'listenbrainz'."""
        from main import _normalise_source
        assert _normalise_source("lb") == "listenbrainz"
        assert _normalise_source("LB") == "listenbrainz"

    def test_normalise_lastfm(self):
        """Keeps 'lastfm' unchanged."""
        from main import _normalise_source
        assert _normalise_source("lastfm") == "lastfm"

    def test_resolve_user_explicit(self):
        """Returns the user when explicitly provided."""
        from main import _resolve_user
        assert _resolve_user("lastfm", "myuser") == "myuser"

    @patch.dict("os.environ", {"LASTFM_USER": "envuser"}, clear=False)
    def test_resolve_user_from_env_lastfm(self):
        """Falls back to LASTFM_USER env var."""
        from main import _resolve_user
        assert _resolve_user("lastfm", None) == "envuser"

    @patch.dict("os.environ", {"LB_USER": "lbenvuser"}, clear=False)
    def test_resolve_user_from_env_lb(self):
        """Falls back to LB_USER env var for listenbrainz."""
        from main import _resolve_user
        assert _resolve_user("listenbrainz", None) == "lbenvuser"

    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_user_missing_raises(self):
        """Raises UsageError when neither flag nor env var is set."""
        import click
        from main import _resolve_user
        with pytest.raises(click.UsageError, match="--user is required"):
            _resolve_user("lastfm", None)


class TestQaGroup:
    """Tests the 'qa' command group."""

    def test_qa_no_subcommand_shows_help(self, runner):
        """Shows help text when invoked without a subcommand."""
        result = runner.invoke(cli, ["qa"])
        assert result.exit_code == 0
        assert "scrobble" in result.output
        assert "show" in result.output

    @patch("corefunc.qa.qa_lb_ingest")
    def test_qa_scrobble_pass(self, mock_qa, runner, tmp_pq_dir):
        """Reports PASS when all scrobble checks succeed."""
        mock_qa.return_value = {
            "row_count": 100,
            "passed": True,
            "schema": {"pass": True, "missing": [], "unexpected": []},
            "nulls": {"artist_name": {"null_pct": 0, "empty_pct": 0}},
            "timestamps": {"pass": True, "issues": []},
            "duplicates": {"duplicate_count": 0, "duplicate_pct": 0.0, "pass": True},
            "mbids": {"fill_rate": 90, "valid_rate": 95},
            "encoding": {"pass": True, "bad_char_rows": 0},
            "reconciliation": {"fetched": None, "stored": 100, "pass": True},
        }
        result = runner.invoke(cli, ["qa", "scrobble"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    @patch("corefunc.qa.qa_artist_info")
    def test_qa_a_i_pass(self, mock_qa, runner, tmp_pq_dir):
        """Reports PASS when artist_info checks succeed."""
        mock_qa.return_value = {
            "row_count": 50,
            "passed": True,
            "schema": {"pass": True, "missing": [], "unexpected": []},
            "nulls": {"artist_name": {"null_pct": 0, "empty_pct": 0}},
            "duplicates": {"duplicate_count": 0, "duplicate_pct": 0.0, "pass": True},
            "mbids": {"fill_rate": 95, "valid_rate": 100},
            "encoding": {"pass": True, "bad_char_rows": 0},
        }
        result = runner.invoke(cli, ["qa", "a_i"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    @patch("corefunc.qa.qa_avc")
    def test_qa_avc_pass(self, mock_qa, runner, tmp_pq_dir):
        """Reports PASS when avc checks succeed."""
        mock_qa.return_value = {
            "row_count": 20,
            "passed": True,
            "schema": {"pass": True, "missing": [], "unexpected": []},
            "nulls": {},
            "duplicates": {"duplicate_count": 0, "duplicate_pct": 0.0, "pass": True},
            "timestamps": {"pass": True, "issues": []},
            "encoding": {"pass": True, "bad_char_rows": 0},
        }
        result = runner.invoke(cli, ["qa", "avc"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    @patch("corefunc.qa.qa_uc")
    def test_qa_uc_summary(self, mock_qa, runner, tmp_pq_dir):
        """Displays entry count and unique countries."""
        mock_qa.return_value = {
            "row_count": 3,
            "unique_countries": 2,
            "passed": True,
        }
        result = runner.invoke(cli, ["qa", "uc"])
        assert result.exit_code == 0
        assert "Entries: 3" in result.output
        assert "Unique countries: 2" in result.output

    def test_qa_show_empty(self, runner, tmp_pq_dir):
        """Reports no reports when qa_report.parquet does not exist."""
        result = runner.invoke(cli, ["qa", "show"])
        assert result.exit_code == 0
        assert "No QA reports found" in result.output

    def test_qa_show_with_data(self, runner, tmp_pq_dir):
        """Displays report rows with source/target in src= field."""
        df = pd.DataFrame([{
            "timestamp": "2026-02-24T12:00:00",
            "source": "listenbrainz",
            "target": "scrobble",
            "row_count": 500,
            "passed": True,
            "schema_ok": True,
            "artist_null_pct": 0.0,
            "track_null_pct": 0.0,
            "album_null_pct": 5.0,
            "mbid_fill_rate": 80.0,
            "mbid_valid_rate": 99.0,
            "duplicate_count": 2,
            "duplicate_pct": 0.4,
            "ts_before_min": 0,
            "ts_after_now": 0,
            "bad_char_rows": 0,
            "fetched": None,
            "stored": 500,
        }])
        df.to_parquet(tmp_pq_dir / "qa_report.parquet", index=False)
        result = runner.invoke(cli, ["qa", "show"])
        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "500" in result.output
        assert "listenbrainz/scrobble" in result.output

    def test_qa_show_target_only(self, runner, tmp_pq_dir):
        """Shows only target when source is absent."""
        df = pd.DataFrame([{
            "timestamp": "2026-02-24T14:00:00",
            "source": None,
            "target": "artist_info",
            "row_count": 100,
            "passed": True,
            "schema_ok": True,
            "artist_null_pct": 0.0,
            "track_null_pct": 0.0,
            "album_null_pct": 0.0,
            "mbid_fill_rate": 90.0,
            "mbid_valid_rate": 100.0,
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "ts_before_min": 0,
            "ts_after_now": 0,
            "bad_char_rows": 0,
            "fetched": None,
            "stored": 100,
        }])
        df.to_parquet(tmp_pq_dir / "qa_report.parquet", index=False)
        result = runner.invoke(cli, ["qa", "show"])
        assert result.exit_code == 0
        assert "src=artist_info" in result.output
        # Ensuring no "n/a" or double slash appears
        assert "n/a" not in result.output

    def test_qa_show_fail_only(self, runner, tmp_pq_dir):
        """Filters to failed reports when --fail-only is set."""
        df = pd.DataFrame([
            {"timestamp": "2026-02-24T12:00:00", "source": "lastfm",
             "target": "scrobble", "row_count": 500, "passed": True,
             "schema_ok": True, "artist_null_pct": 0.0, "track_null_pct": 0.0,
             "album_null_pct": 0.0, "mbid_fill_rate": 80.0,
             "mbid_valid_rate": 99.0, "duplicate_count": 0, "duplicate_pct": 0.0,
             "ts_before_min": 0, "ts_after_now": 0, "bad_char_rows": 0,
             "fetched": None, "stored": 500},
            {"timestamp": "2026-02-24T13:00:00", "source": "listenbrainz",
             "target": "scrobble", "row_count": 600, "passed": False,
             "schema_ok": False, "artist_null_pct": 10.0, "track_null_pct": 0.0,
             "album_null_pct": 0.0, "mbid_fill_rate": 50.0,
             "mbid_valid_rate": 80.0, "duplicate_count": 50, "duplicate_pct": 8.3,
             "ts_before_min": 0, "ts_after_now": 0, "bad_char_rows": 3,
             "fetched": None, "stored": 600},
        ])
        df.to_parquet(tmp_pq_dir / "qa_report.parquet", index=False)
        result = runner.invoke(cli, ["qa", "show", "--fail-only"])
        assert result.exit_code == 0
        assert "FAIL" in result.output
        # Ensuring only the failed row appears (not 500-row PASS)
        assert "500" not in result.output


class TestTrainCommand:
    """Tests the 'train' command."""

    def test_train_runs(self, runner, monkeypatch):
        """Invokes train_model and reports done."""
        mock_train = MagicMock()
        monkeypatch.setattr("corefunc.canon.train_model", mock_train)
        result = runner.invoke(cli, ["train"])
        assert result.exit_code == 0
        assert "Done" in result.output
        mock_train.assert_called_once()
