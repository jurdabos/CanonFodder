"""
Unit tests for helpers.cli interactive functions with mocked prompts.
"""

from unittest.mock import patch, MagicMock
import pandas as pd
from helpers.cli import (
    ask,
    choose_lastfm_user,
    choose_timeline,
    edit_country_timeline,
    make_signature,
    unify_artist_names_cli,
    verify_commas,
    yes_no,
)


class TestAsk:
    """Tests the prompt-until-answer helper."""

    @patch("builtins.input", return_value="yes")
    def test_returns_answer(self, mock_input):
        """Returns the user's input."""
        assert ask("Continue?") == "yes"

    @patch("builtins.input", return_value="")
    def test_returns_default(self, mock_input):
        """Returns default when user presses Enter."""
        assert ask("Continue?", default="y") == "y"

    @patch("builtins.input", side_effect=["", "ok"])
    @patch("builtins.print")
    def test_re_prompts_without_default(self, mock_print, mock_input):
        """Keeps asking when no default is set and input is empty."""
        assert ask("Continue?") == "ok"


class TestChooseLastfmUser:
    """Tests the Last.fm user prompt."""

    @patch("builtins.input", return_value="testuser")
    def test_returns_typed_user(self, mock_input):
        """Returns the typed username."""
        assert choose_lastfm_user() == "testuser"

    @patch("builtins.input", return_value="")
    @patch.dict("os.environ", {"LASTFM_USER": "envuser"})
    def test_returns_env_default(self, mock_input):
        """Returns LASTFM_USER from env when user presses Enter."""
        assert choose_lastfm_user() == "envuser"


class TestChooseTimeline:
    """Tests the timeline choice prompt."""

    @patch("sys.stdin")
    @patch("builtins.input", return_value="y")
    def test_returns_y(self, mock_input, mock_stdin):
        """Returns 'y' for yes."""
        mock_stdin.isatty.return_value = True
        assert choose_timeline() == "y"

    @patch("os.getenv", return_value="1")
    def test_non_tty_returns_default(self, mock_getenv):
        """Returns default when no TTY is detected (e.g. PyCharm)."""
        assert choose_timeline(default="Y") == "y"


class TestYesNo:
    """Tests the yes/no helper."""

    @patch("builtins.input", return_value="yes")
    def test_returns_true(self, mock_input):
        """Returns True for 'yes'."""
        assert yes_no("Delete?") is True

    @patch("builtins.input", return_value="no")
    def test_returns_false(self, mock_input):
        """Returns False for 'no'."""
        assert yes_no("Delete?") is False


class TestEditCountryTimeline:
    """Tests the timeline editor with mocked Click prompts."""

    @patch("helpers.cli.click")
    def test_empty_timeline(self, mock_click, tmp_pq_dir):
        """Returns empty DataFrame when user immediately finishes."""
        mock_click.prompt.return_value = ""
        mock_click.echo = MagicMock()
        result = edit_country_timeline()
        assert result.empty

    @patch("helpers.cli.dump_parquet")
    @patch("helpers.cli.click")
    def test_adds_entry(self, mock_click, mock_dump, tmp_pq_dir):
        """Adds a country entry to the timeline."""
        mock_click.prompt.side_effect = ["HU", "2020-01-01", "", ""]
        mock_click.echo = MagicMock()
        result = edit_country_timeline()
        assert len(result) == 1
        assert result.iloc[0]["country_code"] == "HU"


class TestUnifyArtistNamesCli:
    """Tests the CLI dedup resolver with pre-loaded avc decisions."""

    def test_auto_applies_previous_link(self, tmp_pq_dir):
        """Automatically applies a previous 'link' decision from avc.parquet."""
        import helpers.io as io_mod

        # Pre-loading a decision into avc.parquet
        sig = make_signature(["Beatles", "The Beatles"])
        avc = pd.DataFrame(
            [
                {
                    "artist_variants_hash": "test",
                    "artist_variants_text": sig,
                    "canonical_name": "The Beatles",
                    "to_link": True,
                    "comment": "",
                    "stamp": "2024-01-01",
                }
            ]
        )
        avc.to_parquet(io_mod.AVC_PQ, index=False)
        data = pd.DataFrame({"Artist": ["Beatles", "The Beatles", "Radiohead"]})
        artcounts = pd.DataFrame({"Artist": ["Beatles", "The Beatles", "Radiohead"], "Count": [5, 10, 20]})
        groups = [["Beatles", "The Beatles"]]
        result_data, result_counts = unify_artist_names_cli(data, artcounts, groups)
        assert (result_data["Artist"] == "The Beatles").sum() == 2

    def test_auto_skips_previous_skip(self, tmp_pq_dir):
        """Skips a group that was previously marked as skip."""
        import helpers.io as io_mod

        sig = make_signature(["Alpha", "Beta"])
        avc = pd.DataFrame(
            [
                {
                    "artist_variants_hash": "test",
                    "artist_variants_text": sig,
                    "canonical_name": "__SKIP__",
                    "to_link": False,
                    "comment": "",
                    "stamp": "2024-01-01",
                }
            ]
        )
        avc.to_parquet(io_mod.AVC_PQ, index=False)
        data = pd.DataFrame({"Artist": ["Alpha", "Beta", "Gamma"]})
        artcounts = pd.DataFrame({"Artist": ["Alpha", "Beta", "Gamma"], "Count": [5, 10, 20]})
        groups = [["Alpha", "Beta"]]
        result_data, _ = unify_artist_names_cli(data, artcounts, groups)
        # Both should remain unchanged since the group was skipped
        assert (result_data["Artist"] == "Alpha").sum() == 1
        assert (result_data["Artist"] == "Beta").sum() == 1


class TestVerifyCommas:
    """Tests the CSV comma verifier."""

    def test_runs_without_error(self, tmp_path, capsys):
        """Prints diagnostic output for a CSV file."""
        csv = tmp_path / "test.csv"
        csv.write_text(
            'Artist,Album,Song\n"Emerson, Lake & Palmer","Grey Tickles, Black Pressure","A Song"\n',
            encoding="utf-8",
        )
        verify_commas(csv)
        out = capsys.readouterr().out
        assert "Checking" in out
