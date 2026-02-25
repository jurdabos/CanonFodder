"""
Tests for corefunc.qa — post-ingestion scrobble quality checks.
"""
import pandas as pd
from corefunc.qa import (
    _check_duplicates,
    _check_encoding,
    _check_mbids,
    _check_nulls,
    _check_schema,
    _check_timestamps,
    _real_fill,
    _reconcile_rows,
    qa_artist_info,
    qa_lb_ingest,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _good_df(n: int = 5) -> pd.DataFrame:
    """Builds a minimal valid scrobble DataFrame with *n* rows."""
    return pd.DataFrame({
        "artist_name": [f"Artist {i}" for i in range(n)],
        "album_title": [f"Album {i}" for i in range(n)],
        "track_title": [f"Track {i}" for i in range(n)],
        "artist_mbid": ["a4074512-87e0-4820-b609-0c4a18142a70"] * n,
        "play_time": pd.date_range("2024-06-01", periods=n, freq="5min", tz="UTC"),
    })


# ── Schema ────────────────────────────────────────────────────────────────────
class TestCheckSchema:
    """Tests for _check_schema."""

    def test_valid_schema(self):
        """Passes when columns match SCROBBLE_COLS exactly."""
        result = _check_schema(_good_df())
        assert result["pass"] is True
        assert result["missing"] == []
        assert result["unexpected"] == []

    def test_missing_column(self):
        """Fails when a required column is absent."""
        df = _good_df().drop(columns=["album_title"])
        result = _check_schema(df)
        assert result["pass"] is False
        assert "album_title" in result["missing"]

    def test_unexpected_column(self):
        """Fails when an extra column is present."""
        df = _good_df()
        df["extra"] = "oops"
        result = _check_schema(df)
        assert result["pass"] is False
        assert "extra" in result["unexpected"]


# ── Nulls ─────────────────────────────────────────────────────────────────────
class TestCheckNulls:
    """Tests for _check_nulls."""

    def test_no_nulls(self):
        """Reports 0 % for clean data."""
        result = _check_nulls(_good_df())
        assert result["artist_name"]["null_pct"] == 0
        assert result["track_title"]["null_pct"] == 0

    def test_with_nulls(self):
        """Reports correct percentages when nulls are present."""
        df = _good_df(4)
        df.loc[0, "artist_name"] = None
        df.loc[1, "album_title"] = None
        result = _check_nulls(df)
        assert result["artist_name"]["null_count"] == 1
        assert result["artist_name"]["null_pct"] == 25.0
        assert result["album_title"]["null_count"] == 1

    def test_with_empty_strings(self):
        """Distinguishes empty strings from nulls."""
        df = _good_df(4)
        df.loc[0, "track_title"] = ""
        result = _check_nulls(df)
        assert result["track_title"]["empty_count"] == 1
        assert result["track_title"]["null_count"] == 0


# ── Timestamps ────────────────────────────────────────────────────────────────
class TestCheckTimestamps:
    """Tests for _check_timestamps."""

    def test_valid_timestamps(self):
        """Passes with plausible, tz-aware, sorted timestamps."""
        result = _check_timestamps(_good_df())
        assert result["pass"] is True
        assert result["issues"] == []

    def test_pre_2000_timestamp(self):
        """Flags timestamps before the plausible minimum year."""
        df = _good_df(2)
        df.loc[0, "play_time"] = pd.Timestamp("1999-12-31", tz="UTC")
        result = _check_timestamps(df)
        assert result["pass"] is False
        assert result["before_min_count"] == 1

    def test_future_timestamp(self):
        """Flags timestamps in the future."""
        df = _good_df(2)
        df.loc[0, "play_time"] = pd.Timestamp("2099-01-01", tz="UTC")
        result = _check_timestamps(df)
        assert result["pass"] is False
        assert result["after_now_count"] == 1

    def test_tz_naive_timestamps(self):
        """Flags tz-naive timestamps but still performs range checks."""
        df = _good_df(2)
        df["play_time"] = df["play_time"].dt.tz_localize(None)
        result = _check_timestamps(df)
        assert result["pass"] is False
        assert any("timezone" in i.lower() for i in result["issues"])
        assert "before_min_count" in result  # to confirm range checks ran

    def test_missing_play_time(self):
        """Fails gracefully when play_time column is absent."""
        df = _good_df().drop(columns=["play_time"])
        result = _check_timestamps(df)
        assert result["pass"] is False


# ── Duplicates ────────────────────────────────────────────────────────────────
class TestCheckDuplicates:
    """Tests for _check_duplicates."""

    def test_no_duplicates(self):
        """Passes with unique rows."""
        result = _check_duplicates(_good_df())
        assert result["duplicate_count"] == 0
        assert result["pass"] is True

    def test_with_duplicates(self):
        """Detects duplicate rows on the dedup key."""
        df = _good_df(2)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = _check_duplicates(df)
        assert result["duplicate_count"] == 1

    def test_high_duplicate_rate(self):
        """Fails when duplicate rate exceeds threshold."""
        df = _good_df(1)
        # 100 % duplication
        df = pd.concat([df] * 20, ignore_index=True)
        result = _check_duplicates(df)
        assert result["pass"] is False


# ── MBIDs ─────────────────────────────────────────────────────────────────────
class TestCheckMbids:
    """Tests for _check_mbids."""

    def test_all_valid(self):
        """Reports 100 % fill and valid rates."""
        result = _check_mbids(_good_df())
        assert result["fill_rate"] == 100.0
        assert result["valid_rate"] == 100.0

    def test_some_missing(self):
        """Reports correct fill rate with some nulls."""
        df = _good_df(4)
        df.loc[0, "artist_mbid"] = None
        result = _check_mbids(df)
        assert result["filled"] == 3
        assert result["fill_rate"] == 75.0

    def test_invalid_uuid(self):
        """Detects malformed MBIDs."""
        df = _good_df(2)
        df.loc[0, "artist_mbid"] = "not-a-valid-uuid"
        result = _check_mbids(df)
        assert result["valid"] == 1
        assert result["valid_rate"] == 50.0


# ── Encoding ──────────────────────────────────────────────────────────────────
class TestCheckEncoding:
    """Tests for _check_encoding."""

    def test_clean_text(self):
        """Passes with normal characters."""
        result = _check_encoding(_good_df())
        assert result["pass"] is True
        assert result["bad_char_rows"] == 0

    def test_replacement_char(self):
        """Detects Unicode replacement character."""
        df = _good_df(2)
        df.loc[0, "artist_name"] = "Böhren \ufffd"
        result = _check_encoding(df)
        assert result["pass"] is False
        assert result["bad_char_rows"] >= 1

    def test_control_chars(self):
        """Detects C0 control characters."""
        df = _good_df(2)
        df.loc[0, "track_title"] = "Track\x00Name"
        result = _check_encoding(df)
        assert result["pass"] is False

    def test_non_latin_ok(self):
        """Does not flag valid non-Latin scripts."""
        df = _good_df(3)
        df.loc[0, "artist_name"] = "אריק איינשטיין"   # Hebrew
        df.loc[1, "artist_name"] = "Σωκράτης Μάλαμας"  # Greek
        df.loc[2, "artist_name"] = "ลุลา"               # Thai
        result = _check_encoding(df)
        assert result["pass"] is True


# ── Real fill ───────────────────────────────────────────────────────────────
class TestRealFill:
    """Tests for _real_fill."""

    def test_all_real(self):
        """Counts all values when none are None/empty."""
        s = pd.Series(["US", "DE", "GB"])
        result = _real_fill(s, 3)
        assert result["filled"] == 3
        assert result["fill_rate"] == 100.0

    def test_none_excluded(self):
        """Excludes Python None values."""
        s = pd.Series(["US", None, "GB"])
        result = _real_fill(s, 3)
        assert result["filled"] == 2
        assert result["fill_rate"] == 66.67

    def test_empty_excluded(self):
        """Excludes empty strings."""
        s = pd.Series(["US", "", "GB"])
        result = _real_fill(s, 3)
        assert result["filled"] == 2

    def test_none_string_excluded(self):
        """Excludes the literal string 'None' (case-insensitive)."""
        s = pd.Series(["US", "None", "none", "NONE", "GB"])
        result = _real_fill(s, 5)
        assert result["filled"] == 2
        assert result["fill_rate"] == 40.0

    def test_mixed_junk(self):
        """Excludes all non-real values at once."""
        s = pd.Series(["US", None, "", "None", "   ", "DE"])
        result = _real_fill(s, 6)
        assert result["filled"] == 2
        assert result["fill_rate"] == 33.33

    def test_empty_series(self):
        """Returns 0 filled for an empty series."""
        s = pd.Series([], dtype=str)
        result = _real_fill(s, 0)
        assert result["filled"] == 0
        assert result["fill_rate"] == 0.0


# ── Row-count reconciliation ────────────────────────────────────────────────────
class TestReconcileRows:
    """Tests for _reconcile_rows."""

    def test_exact_match(self):
        """Passes when fetched equals stored."""
        df = _good_df(10)
        result = _reconcile_rows(df, fetched_count=10)
        assert result["pass"] is True
        assert result["diff"] == 0

    def test_no_fetched_count(self):
        """Passes when fetched_count is not provided."""
        result = _reconcile_rows(_good_df(), fetched_count=None)
        assert result["pass"] is True

    def test_large_discrepancy(self):
        """Fails when discrepancy exceeds threshold."""
        df = _good_df(5)
        result = _reconcile_rows(df, fetched_count=100)
        assert result["pass"] is False


# ── Full qa_lb_ingest ─────────────────────────────────────────────────────────
class TestQaLbIngest:
    """Integration tests for the top-level qa_lb_ingest function."""

    def test_skips_on_empty(self, tmp_pq_dir):
        """Returns skipped status when scrobble.parquet is missing."""
        result = qa_lb_ingest()
        assert result["status"] == "skipped"

    def test_full_report_on_good_data(self, populated_pq, monkeypatch):
        """Returns a passing report for clean sample data."""
        import corefunc.qa as qa_mod
        monkeypatch.setattr(qa_mod, "SCROBBLE_PQ", populated_pq / "scrobble.parquet")
        monkeypatch.setattr(qa_mod, "QA_REPORT_PQ", populated_pq / "qa_report.parquet")
        result = qa_lb_ingest()
        assert result["passed"] is True
        assert result["row_count"] == 3
        assert (populated_pq / "qa_report.parquet").exists()

    def test_report_persisted(self, populated_pq, monkeypatch):
        """Verifies that a row was appended to qa_report.parquet."""
        import corefunc.qa as qa_mod
        monkeypatch.setattr(qa_mod, "SCROBBLE_PQ", populated_pq / "scrobble.parquet")
        monkeypatch.setattr(qa_mod, "QA_REPORT_PQ", populated_pq / "qa_report.parquet")
        qa_lb_ingest()
        report_df = pd.read_parquet(populated_pq / "qa_report.parquet")
        assert len(report_df) == 1
        assert "passed" in report_df.columns
        assert "mbid_fill_rate" in report_df.columns


# ── qa_artist_info ─────────────────────────────────────────────────────────
class TestQaArtistInfo:
    """Tests for qa_artist_info enrichment reporting."""

    def test_skips_on_empty(self, tmp_pq_dir):
        """Returns skipped status when artist_info.parquet is missing."""
        result = qa_artist_info()
        assert result["status"] == "skipped"

    def test_enrichment_rates(self, tmp_pq_dir):
        """Reports correct real-fill enrichment rates."""
        import helpers.io as io_mod
        df = pd.DataFrame({
            "artist_name": ["A", "B", "C", "D"],
            "mbid": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee", "", ""],
            "country": ["DE", None, "US", ""],
            "disambiguation_comment": ["rock", "None", "", None],
            "aliases": ["a1,a2", "", "None", None],
        })
        df.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        result = qa_artist_info()
        enr = result["enrichment"]
        assert enr["country"]["filled"] == 2      # "DE", "US"
        assert enr["country"]["fill_rate"] == 50.0
        assert enr["disambiguation"]["filled"] == 1  # "rock" only
        assert enr["aliases"]["filled"] == 1        # "a1,a2" only

    def test_enrichment_persisted(self, tmp_pq_dir):
        """Verifies enrichment fill rates are written to qa_report.parquet."""
        import helpers.io as io_mod
        df = pd.DataFrame({
            "artist_name": ["A"], "mbid": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"],
            "country": ["GB"], "disambiguation_comment": ["punk"], "aliases": ["alias1"],
        })
        df.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        qa_artist_info()
        report_df = pd.read_parquet(io_mod.QA_REPORT_PQ)
        assert "country_fill_rate" in report_df.columns
        assert "disambiguation_fill_rate" in report_df.columns
        assert "aliases_fill_rate" in report_df.columns
        assert report_df.iloc[0]["country_fill_rate"] == 100.0
