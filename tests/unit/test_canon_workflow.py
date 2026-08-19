"""Tests for corefunc/canon/workflow.py — summary, propagation, decisions, discovery."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _write_avc(pq_dir, rows):
    """Writes avc.parquet rows into the temp PQ dir."""
    df = pd.DataFrame(rows)
    df["to_link"] = df["to_link"].astype("boolean")
    df.to_parquet(pq_dir / "avc.parquet", index=False)


def _write_artist_info(pq_dir, rows):
    """Writes artist_info.parquet rows into the temp PQ dir."""
    pd.DataFrame(rows).to_parquet(pq_dir / "artist_info.parquet", index=False)


def _write_scrobbles(pq_dir, artist_plays):
    """Writes a scrobble parquet with the given artist→play-count mapping."""
    rows = [
        {"artist_name": name, "track_title": "t", "play_time": pd.Timestamp("2024-01-01", tz="UTC")}
        for name, plays in artist_plays.items()
        for _ in range(plays)
    ]
    pd.DataFrame(rows).to_parquet(pq_dir / "scrobble.parquet", index=False)


class TestAvcSummaryBranches:
    """Tests the to_link display branches of avc_summary."""

    def test_all_display_states(self, tmp_pq_dir):
        """Renders ✓/✗/? for True/False/NULL decisions."""
        from corefunc.canon.workflow import avc_summary

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "A",
                    "stamp": pd.Timestamp("2024-01-03", tz="UTC"),
                    "artist_variants_text": "A{B",
                },
                {
                    "to_link": False,
                    "canonical_name": "",
                    "stamp": pd.Timestamp("2024-01-02", tz="UTC"),
                    "artist_variants_text": "C{D",
                },
                {
                    "to_link": pd.NA,
                    "canonical_name": "",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "E{F",
                },
            ],
        )
        rows = avc_summary()
        assert [r["to_link_display"] for r in rows] == ["✓", "✗", "?"]

    def test_decided_and_undecided_filters(self, tmp_pq_dir):
        """Applies decided_only/undecided_only WHERE clauses."""
        from corefunc.canon.workflow import avc_summary

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "A",
                    "stamp": pd.Timestamp("2024-01-02", tz="UTC"),
                    "artist_variants_text": "A{B",
                },
                {
                    "to_link": pd.NA,
                    "canonical_name": "",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "E{F",
                },
            ],
        )
        assert [r["to_link_display"] for r in avc_summary(decided_only=True)] == ["✓"]
        assert [r["to_link_display"] for r in avc_summary(undecided_only=True)] == ["?"]
        assert len(avc_summary(last_n=1)) == 1


class TestPropagateAvcBranches:
    """Tests the early-return and merge paths of propagate_avc."""

    def test_missing_avc_returns_zero(self, tmp_pq_dir):
        """Returns zeroes when avc.parquet is absent."""
        from corefunc.canon.workflow import propagate_avc

        assert propagate_avc() == {"updated": 0, "aliases_added": 0}

    def test_missing_artist_info_returns_zero(self, tmp_pq_dir):
        """Returns zeroes when artist_info.parquet is absent."""
        from corefunc.canon.workflow import propagate_avc

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "A",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "A{B",
                }
            ],
        )
        assert propagate_avc() == {"updated": 0, "aliases_added": 0}

    def test_skip_sentinel_ignored(self, tmp_pq_dir):
        """Ignores rows whose canonical name is the __SKIP__ sentinel."""
        from corefunc.canon.workflow import propagate_avc

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "__SKIP__",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "A{B",
                }
            ],
        )
        _write_artist_info(tmp_pq_dir, [{"artist_name": "A", "aliases": ""}])
        assert propagate_avc() == {"updated": 0, "aliases_added": 0}

    def test_single_variant_group_skipped(self, tmp_pq_dir):
        """Ignores groups without non-canonical variants."""
        from corefunc.canon.workflow import propagate_avc

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "A",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "A",
                }
            ],
        )
        _write_artist_info(tmp_pq_dir, [{"artist_name": "A", "aliases": ""}])
        assert propagate_avc() == {"updated": 0, "aliases_added": 0}

    def test_canonical_absent_from_artist_info(self, tmp_pq_dir):
        """Continues when neither the canonical name nor its variants exist in artist_info."""
        from corefunc.canon.workflow import propagate_avc

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "Alpha",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "Alpha{Beta2",
                }
            ],
        )
        _write_artist_info(tmp_pq_dir, [{"artist_name": "Beta", "aliases": ""}])
        assert propagate_avc() == {"updated": 0, "aliases_added": 0}

    def test_merges_variant_row_into_canonical(self, tmp_pq_dir):
        """Drops the variant row and appends it to the canonical artist's aliases."""
        from corefunc.canon.workflow import propagate_avc

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "Alpha",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "Alpha{Alfa",
                }
            ],
        )
        _write_artist_info(
            tmp_pq_dir, [{"artist_name": "Alpha", "aliases": ""}, {"artist_name": "Alfa", "aliases": ""}]
        )
        result = propagate_avc()
        assert result == {"updated": 1, "aliases_added": 1}
        ai = pd.read_parquet(tmp_pq_dir / "artist_info.parquet")
        assert ai["artist_name"].tolist() == ["Alpha"]
        assert ai.at[0, "aliases"] == "Alfa"

    def test_renames_variant_when_canonical_missing(self, tmp_pq_dir):
        """Renames a variant row when the canonical name has no row yet; the old name becomes an alias."""
        from corefunc.canon.workflow import propagate_avc

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": True,
                    "canonical_name": "Alpha",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_text": "Alpha{Alfa",
                }
            ],
        )
        _write_artist_info(tmp_pq_dir, [{"artist_name": "Alfa", "aliases": ""}])
        result = propagate_avc()
        assert result == {"updated": 1, "aliases_added": 1}
        ai = pd.read_parquet(tmp_pq_dir / "artist_info.parquet")
        assert ai["artist_name"].tolist() == ["Alpha"]
        assert ai.at[0, "aliases"] == "Alfa"


class TestUndecidedAndDecision:
    """Tests undecided_rows and update_avc_decision guard branches."""

    def test_undecided_empty_without_avc(self, tmp_pq_dir):
        """Returns an empty DataFrame when avc.parquet is absent."""
        from corefunc.canon.workflow import undecided_rows

        assert undecided_rows().empty

    def test_update_without_avc_is_noop(self, tmp_pq_dir):
        """Logs and returns when there is nothing to update."""
        from corefunc.canon.workflow import update_avc_decision

        update_avc_decision("h1", True, "A")
        assert not (tmp_pq_dir / "avc.parquet").exists()

    def test_update_unknown_hash_is_noop(self, tmp_pq_dir):
        """Leaves the table unchanged when the hash is not found."""
        from corefunc.canon.workflow import update_avc_decision

        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": pd.NA,
                    "canonical_name": "",
                    "comment": "",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_hash": "h1",
                    "artist_variants_text": "A{B",
                }
            ],
        )
        update_avc_decision("unknown-hash", True, "A")
        avc = pd.read_parquet(tmp_pq_dir / "avc.parquet")
        assert avc["to_link"].isna().all()


class TestListMlflowRuns:
    """Tests list_mlflow_runs filtering and formatting."""

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.get_experiment_by_name", return_value=None)
    def test_no_experiment_returns_empty(self, mock_exp, mock_uri):
        """Returns [] when the experiment does not exist."""
        from corefunc.canon.workflow import list_mlflow_runs

        assert list_mlflow_runs() == []

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.get_experiment_by_name", return_value=SimpleNamespace(experiment_id="1"))
    @patch("mlflow.search_runs", return_value=pd.DataFrame())
    def test_empty_runs_returns_empty(self, mock_search, mock_exp, mock_uri):
        """Returns [] when the experiment has no finished runs."""
        from corefunc.canon.workflow import list_mlflow_runs

        assert list_mlflow_runs() == []

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.get_experiment_by_name", return_value=SimpleNamespace(experiment_id="1"))
    @patch("mlflow.search_runs")
    def test_lists_runs_and_skips_folds(self, mock_search, mock_exp, mock_uri):
        """Formats run rows and skips nested fold runs."""
        from corefunc.canon.workflow import list_mlflow_runs

        mock_search.return_value = pd.DataFrame(
            {
                "run_id": ["r1", "r2"],
                "tags.mlflow.runName": ["exp1_lightgbm", "fold_0"],
                "start_time": [pd.Timestamp("2026-08-01 10:00:00"), pd.Timestamp("2026-08-01 10:01:00")],
                "metrics.precision": [0.912345, 0.1],
                "metrics.recall": [0.856789, 0.2],
                "metrics.f1": [0.885566, 0.15],
                "metrics.auc": [0.954321, 0.5],
            }
        )
        runs = list_mlflow_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "r1"
        assert runs[0]["precision"] == 0.9123
        assert runs[0]["start_time"] == "2026-08-01 10:00:00"


class TestDiscoverCandidates:
    """Tests discover_candidates early exits and the full discovery pass."""

    def test_no_scrobble_data(self, tmp_pq_dir):
        """Returns [] when no scrobble data exists."""
        from corefunc.canon.workflow import discover_candidates

        with patch("corefunc.canon.workflow.scrobble_data_exists", return_value=False):
            assert discover_candidates(MagicMock()) == []

    def test_no_artists_meeting_min_plays(self, tmp_pq_dir):
        """Returns [] when every artist falls below min_plays."""
        from corefunc.canon.workflow import discover_candidates

        _write_scrobbles(tmp_pq_dir, {"Alpha": 1, "Beta": 1})
        with (
            patch("corefunc.canon.workflow.scrobble_data_exists", return_value=True),
            patch(
                "corefunc.canon.workflow.scrobble_duckdb_from",
                return_value=f"'{(tmp_pq_dir / 'scrobble.parquet').as_posix()}'",
            ),
        ):
            assert discover_candidates(MagicMock(), min_plays=2) == []

    def test_all_names_covered(self, tmp_pq_dir):
        """Returns [] when the exclusion set already covers every name."""
        from corefunc.canon.workflow import discover_candidates

        _write_scrobbles(tmp_pq_dir, {"Alpha": 3})
        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": pd.NA,
                    "canonical_name": "",
                    "comment": "",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_hash": "h1",
                    "artist_variants_text": "Alpha",
                }
            ],
        )
        with (
            patch("corefunc.canon.workflow.scrobble_data_exists", return_value=True),
            patch(
                "corefunc.canon.workflow.scrobble_duckdb_from",
                return_value=f"'{(tmp_pq_dir / 'scrobble.parquet').as_posix()}'",
            ),
        ):
            assert discover_candidates(MagicMock()) == []

    def test_discovers_group_and_logs_predictions(self, tmp_pq_dir):
        """Groups similar names, scores pairs, and persists the prediction log."""
        from corefunc.canon.workflow import discover_candidates

        _write_scrobbles(tmp_pq_dir, {"Metallica": 3, "Metalica": 2, "Beta": 2})
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.9]])
        with (
            patch("corefunc.canon.workflow.scrobble_data_exists", return_value=True),
            patch(
                "corefunc.canon.workflow.scrobble_duckdb_from",
                return_value=f"'{(tmp_pq_dir / 'scrobble.parquet').as_posix()}'",
            ),
            patch("helpers.inference.compute_inference_features", return_value={"f": 1.0}),
        ):
            candidates = discover_candidates(model)
        assert len(candidates) == 1
        assert candidates[0]["variants"] == ["Metalica", "Metallica"]
        assert candidates[0]["max_prob"] == 0.9
        assert (tmp_pq_dir / "predictions_log.parquet").exists()

    def test_prediction_failure_skips_pair(self, tmp_pq_dir):
        """Skips pairs whose feature computation fails."""
        from corefunc.canon.workflow import discover_candidates

        _write_scrobbles(tmp_pq_dir, {"Metallica": 3, "Metalica": 2})
        with (
            patch("corefunc.canon.workflow.scrobble_data_exists", return_value=True),
            patch(
                "corefunc.canon.workflow.scrobble_duckdb_from",
                return_value=f"'{(tmp_pq_dir / 'scrobble.parquet').as_posix()}'",
            ),
            patch("helpers.inference.compute_inference_features", side_effect=RuntimeError("boom")),
        ):
            assert discover_candidates(MagicMock()) == []
        assert not (tmp_pq_dir / "predictions_log.parquet").exists()

    def test_existing_hash_skipped(self, tmp_pq_dir):
        """Does not re-flag a group whose hash is already in avc.parquet."""
        from corefunc.canon.workflow import _make_hash, _make_signature, discover_candidates

        _write_scrobbles(tmp_pq_dir, {"Metallica": 3, "Metalica": 2})
        _write_avc(
            tmp_pq_dir,
            [
                {
                    "to_link": pd.NA,
                    "canonical_name": "",
                    "comment": "",
                    "stamp": pd.Timestamp("2024-01-01", tz="UTC"),
                    "artist_variants_hash": _make_hash(_make_signature(["Metallica", "Metalica"])),
                    "artist_variants_text": "Gamma{Delta",
                }
            ],
        )
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.9]])
        with (
            patch("corefunc.canon.workflow.scrobble_data_exists", return_value=True),
            patch(
                "corefunc.canon.workflow.scrobble_duckdb_from",
                return_value=f"'{(tmp_pq_dir / 'scrobble.parquet').as_posix()}'",
            ),
            patch("helpers.inference.compute_inference_features", return_value={"f": 1.0}),
        ):
            assert discover_candidates(model) == []
