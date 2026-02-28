"""
Tests for uncovered CLI commands and helper functions in main.py.
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


# ═══════════════════════════════════════════════════════════════════════════════
# avc subcommands
# ═══════════════════════════════════════════════════════════════════════════════
class TestAvcShowCommand:
    """Tests 'canon avc show'."""

    @patch("corefunc.canon.workflow.avc_summary", return_value=[])
    def test_show_empty(self, mock_summary, runner, tmp_pq_dir):
        """Reports no rows when avc is empty."""
        result = runner.invoke(cli, ["canon", "avc", "show"])
        assert result.exit_code == 0
        assert "No avc rows found" in result.output

    @patch("corefunc.canon.workflow.avc_summary")
    def test_show_with_rows(self, mock_summary, runner, tmp_pq_dir):
        """Displays avc rows in table format."""
        mock_summary.return_value = [
            {"idx": 1, "to_link_display": "✓", "canonical_name": "Alpha", "stamp": "2024-01-01", "artist_variants_text": "Alpha{Alfa"},
            {"idx": 2, "to_link_display": "?", "canonical_name": "", "stamp": "2024-01-02", "artist_variants_text": "Beta{Beto"},
        ]
        result = runner.invoke(cli, ["canon", "avc", "show"])
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Beta" in result.output
        assert "2 rows" in result.output


class TestAvcPropagateCommand:
    """Tests 'canon avc propagate'."""

    @patch("corefunc.canon.workflow.propagate_avc")
    def test_propagate(self, mock_prop, runner, tmp_pq_dir):
        """Propagates AVC decisions and reports results."""
        mock_prop.return_value = {"updated": 3, "aliases_added": 5}
        result = runner.invoke(cli, ["canon", "avc", "propagate"])
        assert result.exit_code == 0
        assert "3 row(s) updated" in result.output
        assert "5 alias(es) added" in result.output


class TestAvcSeedCommand:
    """Tests 'canon avc seed'."""

    @patch("corefunc.avc_seed.seed_avc_from_sql", return_value=42)
    def test_seed(self, mock_seed, runner, tmp_path):
        """Seeds from SQL file and reports row count."""
        sql_file = tmp_path / "dump.sql"
        sql_file.write_text("-- dummy")
        result = runner.invoke(cli, ["canon", "avc", "seed", str(sql_file)])
        assert result.exit_code == 0
        assert "42 rows" in result.output


class TestAvcAugmentCommand:
    """Tests 'canon avc augment'."""

    @patch("corefunc.canon.augment.augment_gold_standard", return_value=100)
    def test_augment_ok(self, mock_aug, runner, tmp_pq_dir):
        """Augments gold standard and reports count."""
        result = runner.invoke(cli, ["canon", "avc", "augment"])
        assert result.exit_code == 0
        assert "100 rows" in result.output

    @patch("corefunc.canon.augment.augment_gold_standard", side_effect=RuntimeError("DB error"))
    def test_augment_error(self, mock_aug, runner, tmp_pq_dir):
        """Reports RuntimeError gracefully."""
        result = runner.invoke(cli, ["canon", "avc", "augment"])
        assert result.exit_code == 0
        assert "DB error" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# canon experiment / machine
# ═══════════════════════════════════════════════════════════════════════════════
class TestCanonExperiment:
    """Tests 'canon experiment'."""

    @patch("corefunc.canon.experiment_runner.run_experiment")
    def test_experiment_default(self, mock_run, runner, tmp_pq_dir):
        """Runs experiment with defaults."""
        result = runner.invoke(cli, ["canon", "experiment"])
        assert result.exit_code == 0
        assert "Experiment complete" in result.output
        mock_run.assert_called_once()

    @patch("corefunc.canon.experiment_runner.run_experiment")
    def test_experiment_with_models(self, mock_run, runner, tmp_pq_dir):
        """Passes model filter list."""
        result = runner.invoke(cli, ["canon", "experiment", "--models", "LightGBM,XGBoost"])
        assert result.exit_code == 0
        assert "LightGBM" in result.output


class TestCanonMachine:
    """Tests 'canon machine'."""

    @patch("corefunc.canon.workflow.write_new_candidates", return_value=2)
    @patch("corefunc.canon.workflow.discover_candidates")
    @patch("helpers.inference.load_model")
    def test_machine_with_candidates(self, mock_load, mock_discover, mock_write, runner, tmp_pq_dir):
        """Discovers candidates and writes them after confirmation."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["f1", "f2"]
        mock_load.return_value = mock_model
        mock_discover.return_value = [
            {"variants": ["Alpha", "Alfa"], "max_prob": 0.95, "signature": "Alfa{Alpha", "hash": "h1"},
        ]
        result = runner.invoke(cli, ["canon", "machine"], input="y\n")
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Written" in result.output

    @patch("helpers.inference.load_model", side_effect=FileNotFoundError)
    def test_machine_no_model(self, mock_load, runner, tmp_pq_dir):
        """Reports missing model file."""
        result = runner.invoke(cli, ["canon", "machine"])
        assert result.exit_code == 0
        assert "Model not found" in result.output

    @patch("corefunc.canon.workflow.discover_candidates", return_value=[])
    @patch("helpers.inference.load_model")
    def test_machine_no_candidates(self, mock_load, mock_discover, runner, tmp_pq_dir):
        """Reports no candidates found."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["f1"]
        mock_load.return_value = mock_model
        result = runner.invoke(cli, ["canon", "machine"])
        assert result.exit_code == 0
        assert "No new variant candidates" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# train tcn, tune
# ═══════════════════════════════════════════════════════════════════════════════
class TestTrainTcnCommand:
    """Tests 'train tcn'."""

    @patch("corefunc.canon.tcn_trainer.run_tcn_training")
    def test_tcn_default(self, mock_run, runner, tmp_pq_dir):
        """Runs TCN training with defaults."""
        result = runner.invoke(cli, ["train", "tcn"])
        assert result.exit_code == 0
        assert "TCN training" in result.output
        assert "TCN training complete" in result.output


class TestTuneCommand:
    """Tests 'tune'."""

    @patch("corefunc.canon.tuner.run_tuning")
    def test_tune_default(self, mock_run, runner, tmp_pq_dir):
        """Runs tuning with defaults."""
        result = runner.invoke(cli, ["tune"])
        assert result.exit_code == 0
        assert "Optuna tuning" in result.output
        assert "Tuning complete" in result.output

    @patch("corefunc.canon.tuner.run_tuning")
    def test_tune_with_models(self, mock_run, runner, tmp_pq_dir):
        """Passes model filter list."""
        result = runner.invoke(cli, ["tune", "--models", "LightGBM"])
        assert result.exit_code == 0
        assert "LightGBM" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# fix-encoding, migrate-scrobbles
# ═══════════════════════════════════════════════════════════════════════════════
class TestFixEncodingCommand:
    """Tests 'fix-encoding'."""

    @patch("corefunc.data_cleaning.fix_encoding")
    def test_fix_encoding_nothing(self, mock_fix, runner, tmp_pq_dir):
        """Reports no issues found."""
        mock_fix.return_value = {"scrobble artist": (0, 100), "artist_info name": (0, 50)}
        result = runner.invoke(cli, ["fix-encoding"])
        assert result.exit_code == 0
        assert "No encoding issues found" in result.output

    @patch("corefunc.data_cleaning.fix_encoding")
    def test_fix_encoding_repaired(self, mock_fix, runner, tmp_pq_dir):
        """Reports repaired rows."""
        mock_fix.return_value = {"scrobble artist": (3, 100)}
        result = runner.invoke(cli, ["fix-encoding"])
        assert result.exit_code == 0
        assert "repaired 3 of 100" in result.output


class TestMigrateScrobblesCommand:
    """Tests 'migrate-scrobbles'."""

    @patch("helpers.io.migrate_scrobble_to_partitioned", return_value=0)
    def test_migrate_nothing(self, mock_mig, runner, tmp_pq_dir):
        """Reports nothing to migrate."""
        result = runner.invoke(cli, ["migrate-scrobbles"])
        assert result.exit_code == 0
        assert "Nothing to migrate" in result.output

    @patch("helpers.io.migrate_scrobble_to_partitioned", return_value=5000)
    def test_migrate_with_data(self, mock_mig, runner, tmp_pq_dir):
        """Reports migrated row count."""
        result = runner.invoke(cli, ["migrate-scrobbles"])
        assert result.exit_code == 0
        assert "5,000 scrobbles" in result.output

    @patch("helpers.io.SCROBBLE_PQ")
    @patch("helpers.io.migrate_scrobble_to_partitioned", return_value=100)
    def test_migrate_remove_legacy(self, mock_mig, mock_pq, runner, tmp_pq_dir):
        """Removes legacy file when --remove-legacy is passed."""
        result = runner.invoke(cli, ["migrate-scrobbles", "--remove-legacy"])
        assert result.exit_code == 0
        assert "Removed legacy" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# schema show / migrate
# ═══════════════════════════════════════════════════════════════════════════════
class TestSchemaCommands:
    """Tests 'schema show' and 'schema migrate'."""

    @patch("helpers.schema.validate_schema")
    def test_schema_show(self, mock_validate, runner, populated_pq):
        """Displays schema version info for existing files."""
        mock_validate.return_value = {
            "table": "scrobble", "file_version": "v2", "current_version": "v2",
            "status": "up-to-date", "missing_cols": [],
        }
        result = runner.invoke(cli, ["schema", "show"])
        assert result.exit_code == 0
        assert "Table" in result.output

    @patch("helpers.schema.migrate_all")
    def test_schema_migrate(self, mock_mig, runner, tmp_pq_dir):
        """Runs schema migration and reports results."""
        mock_mig.return_value = {"scrobble": "migrated v1→v2", "artist_info": "up-to-date"}
        result = runner.invoke(cli, ["schema", "migrate"])
        assert result.exit_code == 0
        assert "migrated" in result.output

    @patch("helpers.schema.migrate_all", return_value={})
    def test_schema_migrate_empty(self, mock_mig, runner, tmp_pq_dir):
        """Reports no files when PQ dir is empty."""
        result = runner.invoke(cli, ["schema", "migrate"])
        assert result.exit_code == 0
        assert "No Parquet files found" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# dashboard yearly + empty cases
# ═══════════════════════════════════════════════════════════════════════════════
class TestDashboardExtras:
    """Tests for dashboard commands missing coverage."""

    @patch("corefunc.profile.yearly_top_artists_profile")
    def test_yearly(self, mock_ytap, runner, tmp_pq_dir):
        """Displays yearly top artists with medal labels."""
        mock_ytap.return_value = {
            "top_n": 3,
            "years": [
                {"year": 2024, "artists": [{"rank": 1, "name": "Alpha", "plays": 100}]},
            ],
        }
        result = runner.invoke(cli, ["dashboard", "yearly"])
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "GOLD" in result.output

    @patch("corefunc.profile.yearly_top_artists_profile")
    def test_yearly_error(self, mock_ytap, runner, tmp_pq_dir):
        """Reports error from profile function."""
        mock_ytap.return_value = {"error": "No data"}
        result = runner.invoke(cli, ["dashboard", "yearly"])
        assert result.exit_code == 0
        assert "No data" in result.output

    @patch("helpers.query.top_albums", return_value=pd.DataFrame())
    def test_album_empty(self, _mock, runner, tmp_pq_dir):
        """Reports no album data."""
        result = runner.invoke(cli, ["dashboard", "album"])
        assert result.exit_code == 0
        assert "No album data found" in result.output

    @patch("helpers.query.top_tracks", return_value=pd.DataFrame())
    def test_track_empty(self, _mock, runner, tmp_pq_dir):
        """Reports no track data."""
        result = runner.invoke(cli, ["dashboard", "track"])
        assert result.exit_code == 0
        assert "No track data found" in result.output

    @patch("helpers.query.recent_scrobbles", return_value=pd.DataFrame())
    def test_recent_empty(self, _mock, runner, tmp_pq_dir):
        """Reports no scrobbles."""
        result = runner.invoke(cli, ["dashboard", "recent"])
        assert result.exit_code == 0
        assert "No scrobbles found" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# profile commands
# ═══════════════════════════════════════════════════════════════════════════════
class TestProfileCommands:
    """Tests for 'profile' subcommands."""

    @patch("corefunc.profile.overview_stats")
    def test_overview(self, mock_stats, runner, tmp_pq_dir):
        """Displays eagle-level stats."""
        mock_stats.return_value = {
            "total_scrobbles": 10000, "unique_artists": 500, "unique_tracks": 3000,
            "unique_albums": 800, "earliest": "2020-01-01", "latest": "2025-02-28",
            "yearly": [(2024, 5000), (2025, 5000)],
            "distribution": {
                "min": 1, "q25": 2, "median": 5, "q75": 20, "max": 500,
                "mean": 15.0, "singletons": 100, "lte5": 250, "total_artists": 500,
            },
        }
        result = runner.invoke(cli, ["profile", "overview"])
        assert result.exit_code == 0
        assert "10,000" in result.output
        assert "Yearly scrobbles" in result.output

    @patch("corefunc.profile.overview_stats", return_value={"error": "No data"})
    def test_overview_error(self, mock_stats, runner, tmp_pq_dir):
        """Reports error."""
        result = runner.invoke(cli, ["profile", "overview"])
        assert result.exit_code == 0
        assert "No data" in result.output

    @patch("corefunc.profile.variant_candidates")
    def test_variants(self, mock_vc, runner, tmp_pq_dir):
        """Displays variant candidates."""
        mock_vc.return_value = [
            {"variants": [{"name": "Alpha", "plays": 50}, {"name": "Alfa", "plays": 30}],
             "combined_count": 80, "similarity": 92},
        ]
        result = runner.invoke(cli, ["profile", "variants"])
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "92%" in result.output

    @patch("corefunc.profile.variant_candidates", return_value=[])
    def test_variants_none(self, mock_vc, runner, tmp_pq_dir):
        """Reports no near-duplicates."""
        result = runner.invoke(cli, ["profile", "variants"])
        assert result.exit_code == 0
        assert "No near-duplicate pairs found" in result.output

    @patch("corefunc.profile.top_artists_profile")
    def test_top_raw(self, mock_top, runner, tmp_pq_dir):
        """Displays raw top artists."""
        mock_top.return_value = {
            "raw_top": [{"rank": 1, "name": "Alpha", "plays": 200}],
        }
        result = runner.invoke(cli, ["profile", "top", "-n", "1"])
        assert result.exit_code == 0
        assert "Alpha" in result.output

    @patch("corefunc.profile.top_artists_profile")
    def test_top_canonized(self, mock_top, runner, tmp_pq_dir):
        """Displays canonised top artists."""
        mock_top.return_value = {
            "raw_top": [{"rank": 1, "name": "Alpha", "plays": 200}],
            "canon_top": [{"rank": 1, "name": "Alpha (canonical)", "plays": 250}],
        }
        result = runner.invoke(cli, ["profile", "top", "--canonized"])
        assert result.exit_code == 0
        assert "canonical" in result.output

    @patch("corefunc.profile.top_artists_profile")
    def test_top_canonized_no_artist_info(self, mock_top, runner, tmp_pq_dir):
        """Reports unavailable when artist_info is missing."""
        mock_top.return_value = {
            "raw_top": [{"rank": 1, "name": "Alpha", "plays": 200}],
        }
        result = runner.invoke(cli, ["profile", "top", "--canonized"])
        assert result.exit_code == 0
        assert "unavailable" in result.output

    @patch("corefunc.profile.top_artists_profile", return_value={"error": "No data"})
    def test_top_error(self, mock_top, runner, tmp_pq_dir):
        """Reports error from profile function."""
        result = runner.invoke(cli, ["profile", "top"])
        assert result.exit_code == 0
        assert "No data" in result.output

    @patch("corefunc.profile.trusted_companions")
    def test_companions(self, mock_tc, runner, tmp_pq_dir):
        """Displays trusted companions."""
        mock_tc.return_value = {
            "years": [2023, 2024, 2025], "year_count": 3,
            "companions": [
                {"name": "Alpha", "total_plays": 300, "mean_per_year": 100, "std_dev": 5.0},
            ],
        }
        result = runner.invoke(cli, ["profile", "companions"])
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "300" in result.output

    @patch("corefunc.profile.trusted_companions")
    def test_companions_none(self, mock_tc, runner, tmp_pq_dir):
        """Reports no companions."""
        mock_tc.return_value = {"years": [2023, 2024], "year_count": 2, "companions": []}
        result = runner.invoke(cli, ["profile", "companions"])
        assert result.exit_code == 0
        assert "No artists appear" in result.output

    @patch("corefunc.profile.country_breakdown")
    def test_countries(self, mock_cb, runner, tmp_pq_dir):
        """Displays country breakdown."""
        mock_cb.return_value = [
            {"country": "DE", "play_count": 5000, "artist_count": 50, "pct": 33.3, "name": "Germany"},
        ]
        result = runner.invoke(cli, ["profile", "countries"])
        assert result.exit_code == 0
        assert "Germany" in result.output

    @patch("corefunc.profile.country_breakdown", return_value=[])
    def test_countries_empty(self, mock_cb, runner, tmp_pq_dir):
        """Reports no country data."""
        result = runner.invoke(cli, ["profile", "countries"])
        assert result.exit_code == 0
        assert "No enriched country data" in result.output

    @patch("corefunc.profile.monthly_summary")
    def test_timeline(self, mock_ms, runner, tmp_pq_dir):
        """Displays monthly summary."""
        mock_ms.return_value = {
            "months": [
                {"name": "January", "mean": 500, "min": 400, "max": 600, "total": 2500, "year_count": 5},
            ],
            "strongest": {"name": "January", "mean": 500},
            "weakest": {"name": "July", "mean": 200},
        }
        result = runner.invoke(cli, ["profile", "timeline"])
        assert result.exit_code == 0
        assert "January" in result.output
        assert "Strongest" in result.output

    @patch("corefunc.profile.streak_analysis")
    def test_streaks(self, mock_sa, runner, tmp_pq_dir):
        """Displays streak statistics."""
        mock_sa.return_value = {
            "total_active_days": 365, "first_day": "2024-01-01", "last_day": "2025-02-28",
            "longest_streak": 30, "longest_streak_start": "2024-06-01", "longest_streak_end": "2024-06-30",
            "current_streak": 5, "longest_gap_days": 3, "longest_gap_start": "2024-03-01", "longest_gap_end": "2024-03-03",
        }
        result = runner.invoke(cli, ["profile", "streaks"])
        assert result.exit_code == 0
        assert "365" in result.output
        assert "30 day(s)" in result.output

    @patch("corefunc.profile.listening_clock_profile")
    def test_clock(self, mock_lcp, runner, tmp_pq_dir):
        """Displays listening clock."""
        mock_lcp.return_value = {
            "hours": [{"label": "00:00", "count": 100, "pct": 4.2}],
            "peak_hour": {"label": "20:00", "count": 500},
            "quiet_hour": {"label": "05:00", "count": 10},
            "weekdays": [{"name": "Mon", "count": 1000, "pct": 14.3}],
            "peak_day": {"name": "Sat", "count": 2000},
            "quiet_day": {"name": "Tue", "count": 800},
        }
        result = runner.invoke(cli, ["profile", "clock"])
        assert result.exit_code == 0
        assert "Hour of day" in result.output
        assert "Day of week" in result.output

    @patch("corefunc.profile.population_vs_scrobbles")
    def test_population(self, mock_pvs, runner, tmp_pq_dir):
        """Displays population vs scrobbles."""
        mock_pvs.return_value = {
            "total_countries": 10,
            "by_absolute": [
                {"country": "US", "play_count": 5000, "artist_count": 100,
                 "population": 330000000, "per_million": 15.15, "name": "United States"},
            ],
            "by_per_capita": [
                {"country": "IS", "per_million": 500.0, "play_count": 200,
                 "population": 400000, "name": "Iceland"},
            ],
        }
        result = runner.invoke(cli, ["profile", "population"])
        assert result.exit_code == 0
        assert "United States" in result.output
        assert "Iceland" in result.output

    @patch("corefunc.profile.user_country_profile")
    def test_where(self, mock_ucp, runner, tmp_pq_dir):
        """Displays user country profile."""
        mock_ucp.return_value = {
            "total_scrobbles_matched": 5000, "unique_countries": 3,
            "countries": [
                {"country": "DE", "scrobble_count": 3000, "pct": 60.0, "name": "Germany"},
            ],
        }
        result = runner.invoke(cli, ["profile", "where"])
        assert result.exit_code == 0
        assert "Germany" in result.output

    @patch("corefunc.profile.user_country_medal_profile")
    def test_uc(self, mock_ucmp, runner, tmp_pq_dir):
        """Displays medal tables per user country."""
        mock_ucmp.return_value = {
            "top_n": 3, "ucn": 1,
            "countries": [{
                "country": "DE", "name": "Germany", "scrobble_count": 5000,
                "artists": [{"rank": 1, "name": "Alpha", "plays": 300}],
                "albums": [], "tracks": [],
            }],
        }
        result = runner.invoke(cli, ["profile", "uc"])
        assert result.exit_code == 0
        assert "Germany" in result.output
        assert "GOLD" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# flow command
# ═══════════════════════════════════════════════════════════════════════════════
class TestFlowCommand:
    """Tests 'flow'."""

    def test_flow(self, runner, tmp_pq_dir):
        """Runs Prefect flow and reports results."""
        mock_mod = MagicMock()
        mock_mod.weekly_ingest_flow.return_value = {
            "new_scrobbles": 100, "enriched_artists": 10, "flagged_for_review": 2,
            "avc_propagated": 1, "gs_rows_written": 50, "models_trained": 1,
        }
        with patch.dict("sys.modules", {"flows.cf_ingest": mock_mod}):
            result = runner.invoke(cli, ["flow"])
        assert result.exit_code == 0
        assert "100 scrobbles" in result.output


# ═══════════════════════════════════════════════════════════════════════════════
# helper functions
# ═══════════════════════════════════════════════════════════════════════════════
class TestParseRankRanges:
    """Tests _parse_rank_ranges."""

    def test_valid(self):
        """Parses well-formed rank range string."""
        from main import _parse_rank_ranges
        result = _parse_rank_ranges("(1,5),(27,29)")
        assert result == [(1, 5), (27, 29)]

    def test_invalid_format(self):
        """Raises BadParameter for unparseable input."""
        import click
        from main import _parse_rank_ranges
        with pytest.raises(click.BadParameter):
            _parse_rank_ranges("foobar")

    def test_invalid_range(self):
        """Raises BadParameter for end < start."""
        import click
        from main import _parse_rank_ranges
        with pytest.raises(click.BadParameter, match="Invalid range"):
            _parse_rank_ranges("(5,2)")


class TestEchoRangedEntries:
    """Tests _echo_ranged_entries."""

    def test_filters_by_range(self, runner):
        """Prints only entries within specified ranges."""
        from main import _echo_ranged_entries, cli
        entries = [
            {"rank": 1, "name": "A", "plays": 100},
            {"rank": 2, "name": "B", "plays": 90},
            {"rank": 5, "name": "C", "plays": 50},
        ]
        # Using CliRunner to capture output
        @cli.command("_test_echo", hidden=True)
        def _test():
            _echo_ranged_entries(entries, [(1, 2)])
        result = runner.invoke(cli, ["_test_echo"])
        assert "A" in result.output
        assert "B" in result.output
        assert "C" not in result.output


class TestFormatQaHelpers:
    """Tests _format_qa_src and _format_qa_row."""

    def test_format_qa_src_both(self):
        """Joins source and target with '/'."""
        from main import _format_qa_src
        assert _format_qa_src({"source": "lastfm", "target": "scrobble"}) == "lastfm/scrobble"

    def test_format_qa_src_none(self):
        """Returns empty string when both are None."""
        from main import _format_qa_src
        assert _format_qa_src({"source": None, "target": None}) == ""

    def test_format_qa_src_nan(self):
        """Handles NaN values."""
        from main import _format_qa_src
        assert _format_qa_src({"source": float("nan"), "target": "scrobble"}) == "scrobble"

    def test_format_qa_row_scrobble(self):
        """Formats a scrobble QA row."""
        from main import _format_qa_row
        row = {
            "passed": True, "timestamp": "2024-01-01T00:00:00",
            "source": "lastfm", "target": "scrobble",
            "row_count": 1000, "duplicate_pct": 0.5,
            "mbid_fill_rate": 80, "hash_fill_rate": 0,
            "bad_char_rows": 0, "unique_countries": None,
        }
        line = _format_qa_row(row)
        assert "PASS" in line
        assert "1,000" in line

    def test_format_qa_row_user_country(self):
        """Formats a user_country QA row."""
        from main import _format_qa_row
        row = {
            "passed": True, "timestamp": "2024-01-01T00:00:00",
            "source": None, "target": "user_country",
            "row_count": 5, "duplicate_pct": 0,
            "mbid_fill_rate": 0, "hash_fill_rate": 0,
            "bad_char_rows": 0, "unique_countries": 3,
        }
        line = _format_qa_row(row)
        assert "countries=3" in line

    def test_format_qa_row_artist_info(self):
        """Formats an artist_info QA row."""
        from main import _format_qa_row
        row = {
            "passed": False, "timestamp": "2024-01-01T00:00:00",
            "source": None, "target": "artist_info",
            "row_count": 200, "duplicate_pct": 1.5,
            "mbid_fill_rate": 90, "hash_fill_rate": 0,
            "bad_char_rows": 2, "unique_countries": None,
            "country_fill_rate": 85, "disambiguation_fill_rate": 10,
            "aliases_fill_rate": 30,
        }
        line = _format_qa_row(row)
        assert "FAIL" in line
        assert "country=85%" in line

    def test_format_qa_row_gs_mb(self):
        """Formats a gs_mb QA row."""
        from main import _format_qa_row
        row = {
            "passed": True, "timestamp": "2024-01-01T00:00:00",
            "source": None, "target": "gs_mb",
            "row_count": 500, "duplicate_pct": 0,
            "mbid_fill_rate": 0, "hash_fill_rate": 0,
            "bad_char_rows": 0, "unique_countries": None,
        }
        line = _format_qa_row(row)
        assert "500" in line

    def test_format_qa_row_avc(self):
        """Formats an avc QA row."""
        from main import _format_qa_row
        row = {
            "passed": True, "timestamp": "2024-01-01T00:00:00",
            "source": None, "target": "artist_variants_canonized",
            "row_count": 30, "duplicate_pct": 0,
            "mbid_fill_rate": 0, "hash_fill_rate": 95,
            "bad_char_rows": 0, "unique_countries": None,
        }
        line = _format_qa_row(row)
        assert "hash_fill=95%" in line


class TestParseCountryCodes:
    """Tests _parse_country_codes."""

    def test_parenthesised(self):
        """Parses '(HU, ES, DK)' into upper-cased codes."""
        from main import _parse_country_codes
        assert _parse_country_codes("(HU, ES, DK)") == ["HU", "ES", "DK"]

    def test_comma_separated(self):
        """Parses 'hu,es' without parens."""
        from main import _parse_country_codes
        assert _parse_country_codes("hu,es") == ["HU", "ES"]


class TestParseCategories:
    """Tests _parse_categories."""

    def test_valid(self):
        """Parses '(artist, track)'."""
        from main import _parse_categories
        assert _parse_categories("(artist, track)") == ["artists", "tracks"]

    def test_invalid_raises(self):
        """Raises BadParameter for unknown categories."""
        import click
        from main import _parse_categories
        with pytest.raises(click.BadParameter):
            _parse_categories("(invalid)")


# ═══════════════════════════════════════════════════════════════════════════════
# QA commands — failure/skipped paths
# ═══════════════════════════════════════════════════════════════════════════════
class TestQaFailurePaths:
    """Tests QA command edge cases: skipped, schema failures, etc."""

    @patch("corefunc.qa.qa_lb_ingest")
    def test_qa_scrobble_skipped(self, mock_qa, runner, tmp_pq_dir):
        """Reports skipped when no data."""
        mock_qa.return_value = {"status": "skipped", "reason": "No scrobble data"}
        result = runner.invoke(cli, ["qa", "scrobble"])
        assert result.exit_code == 0
        assert "Skipped" in result.output

    @patch("corefunc.qa.qa_artist_info")
    def test_qa_a_i_skipped(self, mock_qa, runner, tmp_pq_dir):
        """Reports skipped when no artist_info."""
        mock_qa.return_value = {"status": "skipped", "reason": "No artist_info data"}
        result = runner.invoke(cli, ["qa", "a_i"])
        assert result.exit_code == 0
        assert "Skipped" in result.output

    @patch("corefunc.qa.qa_avc")
    def test_qa_avc_skipped(self, mock_qa, runner, tmp_pq_dir):
        """Reports skipped when no avc data."""
        mock_qa.return_value = {"status": "skipped", "reason": "No avc data"}
        result = runner.invoke(cli, ["qa", "avc"])
        assert result.exit_code == 0
        assert "Skipped" in result.output

    @patch("corefunc.qa.qa_gs_mb")
    def test_qa_gs_mb_full(self, mock_qa, runner, tmp_pq_dir):
        """Displays full gs_mb report including label dist and sources."""
        mock_qa.return_value = {
            "row_count": 1000, "passed": True,
            "schema": {"pass": True, "missing": [], "unexpected": []},
            "nulls": {},
            "duplicates": {"duplicate_count": 0, "duplicate_pct": 0.0, "pass": True},
            "encoding": {"pass": True, "bad_char_rows": 0},
            "label_distribution": {"positive": 500, "negative": 500, "null": 0},
            "source_breakdown": {"avc": 300, "mbdb": 700},
        }
        result = runner.invoke(cli, ["qa", "gs_mb"])
        assert result.exit_code == 0
        assert "500 positive" in result.output
        assert "avc=300" in result.output

    @patch("corefunc.qa.qa_uc")
    def test_qa_uc_skipped(self, mock_qa, runner, tmp_pq_dir):
        """Reports skipped when no uc data."""
        mock_qa.return_value = {"status": "skipped", "reason": "No uc data"}
        result = runner.invoke(cli, ["qa", "uc"])
        assert result.exit_code == 0
        assert "Skipped" in result.output

    @patch("corefunc.qa.qa_artist_info")
    def test_qa_a_i_enrichment(self, mock_qa, runner, tmp_pq_dir):
        """Displays enrichment fill rates for artist_info."""
        mock_qa.return_value = {
            "row_count": 100, "passed": True,
            "schema": {"pass": True, "missing": [], "unexpected": []},
            "nulls": {"artist_name": {"null_pct": 0, "empty_pct": 0}},
            "duplicates": {"duplicate_count": 0, "duplicate_pct": 0.0, "pass": True},
            "mbids": {"fill_rate": 95, "valid_rate": 100},
            "encoding": {"pass": True, "bad_char_rows": 0},
            "enrichment": {
                "country": {"filled": 80, "fill_rate": 80.0},
                "disambiguation": {"filled": 20, "fill_rate": 20.0},
                "aliases": {"filled": 50, "fill_rate": 50.0},
            },
        }
        result = runner.invoke(cli, ["qa", "a_i"])
        assert result.exit_code == 0
        assert "Country" in result.output
        assert "80" in result.output

    @patch("corefunc.qa.qa_lb_ingest")
    def test_qa_scrobble_schema_fail(self, mock_qa, runner, tmp_pq_dir):
        """Reports schema failures."""
        mock_qa.return_value = {
            "row_count": 100, "passed": False,
            "schema": {"pass": False, "missing": ["col_x"], "unexpected": ["col_y"]},
            "nulls": {"artist_name": {"null_pct": 5, "empty_pct": 2}},
            "timestamps": {"pass": False, "issues": ["Future timestamps detected"]},
            "duplicates": {"duplicate_count": 10, "duplicate_pct": 10.0, "pass": False},
            "mbids": {"fill_rate": 50, "valid_rate": 80},
            "encoding": {"pass": False, "bad_char_rows": 3},
            "reconciliation": {"fetched": 120, "stored": 100, "pass": False},
        }
        result = runner.invoke(cli, ["qa", "scrobble"])
        assert result.exit_code == 0
        assert "FAIL" in result.output
        assert "Schema FAIL" in result.output
        assert "Future timestamps" in result.output
        assert "Duplicate rate exceeds" in result.output
