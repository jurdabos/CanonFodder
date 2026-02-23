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
    """Tests the 'dashboard' command."""

    def test_dashboard_output(self, runner, populated_pq):
        """Prints scrobble stats and top artists."""
        result = runner.invoke(cli, ["dashboard", "--top", "2"])
        assert result.exit_code == 0
        assert "Scrobbles:" in result.output
        assert "Unique artists:" in result.output
        assert "Bohren" in result.output


class TestPurgeCommand:
    """Tests the 'purge' command."""

    def test_purge_all(self, runner, populated_pq):
        """Deletes all Parquet files when --all is given."""
        from helpers.io import PQ_DIR
        assert any(PQ_DIR.glob("*.parquet"))
        result = runner.invoke(cli, ["purge", "--all", "--yes"])
        assert result.exit_code == 0
        assert not any(PQ_DIR.glob("*.parquet"))

    def test_purge_without_all(self, runner, populated_pq):
        """Reports nothing to purge without --all."""
        result = runner.invoke(cli, ["purge", "--yes"])
        assert result.exit_code == 0
        assert "Nothing to purge" in result.output


class TestIngestCommand:
    """Tests the 'ingest' command."""

    @patch("HTTP.lfAPI.fetch_scrobbles_since")
    def test_ingest_no_new(self, mock_fetch, runner, tmp_pq_dir):
        """Reports no new scrobbles."""
        mock_fetch.return_value = pd.DataFrame()
        result = runner.invoke(cli, ["ingest", "--user", "testuser"])
        assert result.exit_code == 0
        assert "No new scrobbles" in result.output

    @patch("HTTP.lfAPI.fetch_scrobbles_since")
    def test_ingest_with_data(self, mock_fetch, runner, tmp_pq_dir, sample_scrobble_df):
        """Ingests scrobbles and reports count."""
        mock_fetch.return_value = sample_scrobble_df
        result = runner.invoke(cli, ["ingest", "--user", "testuser"])
        assert result.exit_code == 0
        assert "Ingested" in result.output


class TestEnrichCommand:
    """Tests the 'enrich' command."""

    @patch("HTTP.lfAPI.sync_user_country")
    @patch("HTTP.lfAPI.enrich_artist_mbids")
    def test_enrich_runs(self, mock_mbids, mock_country, runner, tmp_pq_dir):
        """Runs enrichment without errors."""
        mock_mbids.return_value = {"status": "ok", "message": "done"}
        mock_country.return_value = False
        result = runner.invoke(cli, ["enrich", "--user", "testuser"])
        assert result.exit_code == 0


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


class TestEnrichListenBrainz:
    """Tests the 'enrich --source listenbrainz' command."""

    def test_enrich_lb_skips_mbids(self, runner, tmp_pq_dir):
        """Skips MBID enrichment for ListenBrainz source."""
        result = runner.invoke(cli, ["enrich", "--user", "lbuser", "--source", "listenbrainz"])
        assert result.exit_code == 0
        assert "Skipping MBID enrichment" in result.output

    def test_enrich_lb_skips_country(self, runner, tmp_pq_dir):
        """Skips country sync for ListenBrainz source."""
        result = runner.invoke(cli, ["enrich", "--user", "lbuser", "--source", "listenbrainz"])
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
