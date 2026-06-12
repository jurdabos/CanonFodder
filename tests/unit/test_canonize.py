"""
Tests for the canonisation workflow modules:
corefunc/canonize.py and corefunc/avc_seed.py.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pandas as pd
import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def tmp_pq(tmp_path):
    """Provides patched PQ paths pointing to a temporary directory."""
    avc = tmp_path / "avc.parquet"
    ai = tmp_path / "artist_info.parquet"
    scrobble = tmp_path / "scrobble.parquet"
    return {"avc": avc, "ai": ai, "scrobble": scrobble, "dir": tmp_path}


def _make_avc_df(rows: list[dict]) -> pd.DataFrame:
    """Builds a minimal avc DataFrame."""
    df = pd.DataFrame(rows)
    if "to_link" in df.columns:
        df["to_link"] = df["to_link"].astype("boolean")
    if "stamp" in df.columns:
        df["stamp"] = pd.to_datetime(df["stamp"], utc=True)
    return df


def _make_ai_df(rows: list[dict]) -> pd.DataFrame:
    """Builds a minimal artist_info DataFrame."""
    return pd.DataFrame(rows)


def _sig(variants: list[str]) -> str:
    """Builds a canonical signature from a variant list."""
    return "{".join(sorted(variants))


def _hash(sig: str) -> str:
    """Returns sha256 hex digest."""
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


# ── avc_summary tests ─────────────────────────────────────────────────────────
class TestAvcSummary:
    """Tests for avc_summary()."""

    def test_returns_empty_when_no_file(self, tmp_pq):
        """Returns empty list when avc.parquet does not exist."""
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import avc_summary

            assert avc_summary() == []

    def test_returns_rows_with_correct_keys(self, tmp_pq):
        """Returns dicts with the expected keys."""
        sig = _sig(["Beatles", "The Beatles"])
        rows = [
            {
                "artist_variants_hash": _hash(sig),
                "artist_variants_text": sig,
                "canonical_name": "The Beatles",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:49:38+00:00",
            }
        ]
        _make_avc_df(rows).to_parquet(tmp_pq["avc"], index=False)
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import avc_summary

            result = avc_summary()
            assert len(result) == 1
            assert result[0]["to_link_display"] == "✓"
            assert result[0]["canonical_name"] == "The Beatles"
            assert "artist_variants_text" in result[0]

    def test_decided_filter(self, tmp_pq):
        """Filters to decided rows only when decided_only=True."""
        sig1 = _sig(["A", "B"])
        sig2 = _sig(["C", "D"])
        rows = [
            {
                "artist_variants_hash": _hash(sig1),
                "artist_variants_text": sig1,
                "canonical_name": "A",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:00:00+00:00",
            },
            {
                "artist_variants_hash": _hash(sig2),
                "artist_variants_text": sig2,
                "canonical_name": "",
                "to_link": pd.NA,
                "comment": "",
                "stamp": "2025-05-02T17:00:00+00:00",
            },
        ]
        _make_avc_df(rows).to_parquet(tmp_pq["avc"], index=False)
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import avc_summary

            decided = avc_summary(decided_only=True)
            assert len(decided) == 1
            assert decided[0]["to_link_display"] == "✓"

    def test_undecided_filter(self, tmp_pq):
        """Filters to undecided rows when undecided_only=True."""
        sig1 = _sig(["A", "B"])
        sig2 = _sig(["C", "D"])
        rows = [
            {
                "artist_variants_hash": _hash(sig1),
                "artist_variants_text": sig1,
                "canonical_name": "A",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:00:00+00:00",
            },
            {
                "artist_variants_hash": _hash(sig2),
                "artist_variants_text": sig2,
                "canonical_name": "",
                "to_link": pd.NA,
                "comment": "",
                "stamp": "2025-05-02T17:00:00+00:00",
            },
        ]
        _make_avc_df(rows).to_parquet(tmp_pq["avc"], index=False)
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import avc_summary

            undecided = avc_summary(undecided_only=True)
            assert len(undecided) == 1
            assert undecided[0]["to_link_display"] == "?"


# ── propagate_avc tests ──────────────────────────────────────────────────────
class TestPropagateAvc:
    """Tests for propagate_avc()."""

    def test_appends_aliases_without_overwriting(self, tmp_pq):
        """Appends variant names to existing aliases."""
        sig = _sig(["Bjork", "Björk"])
        avc_rows = [
            {
                "artist_variants_hash": _hash(sig),
                "artist_variants_text": sig,
                "canonical_name": "Björk",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:52:43+00:00",
            }
        ]
        ai_rows = [
            {"artist_name": "Björk", "mbid": "abc", "country": "IS", "disambiguation_comment": "", "aliases": "Byork"},
        ]
        _make_avc_df(avc_rows).to_parquet(tmp_pq["avc"], index=False)
        _make_ai_df(ai_rows).to_parquet(tmp_pq["ai"], index=False)
        with (
            patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]),
            patch("corefunc.canon.workflow.ARTIST_INFO_PQ", tmp_pq["ai"]),
        ):
            from corefunc.canon.workflow import propagate_avc

            result = propagate_avc()
            assert result["aliases_added"] == 1
            # Verifying the alias was appended, not overwritten
            ai_after = pd.read_parquet(tmp_pq["ai"])
            aliases = ai_after.loc[ai_after["artist_name"] == "Björk", "aliases"].iloc[0]
            assert "Byork" in aliases
            assert "Bjork" in aliases

    def test_renames_variant_row_to_canonical(self, tmp_pq):
        """Renames artist_info rows from variant to canonical name."""
        sig = _sig(["Beatles", "The Beatles"])
        avc_rows = [
            {
                "artist_variants_hash": _hash(sig),
                "artist_variants_text": sig,
                "canonical_name": "The Beatles",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:49:38+00:00",
            }
        ]
        ai_rows = [
            {"artist_name": "Beatles", "mbid": "xyz", "country": "GB", "disambiguation_comment": "", "aliases": ""},
        ]
        _make_avc_df(avc_rows).to_parquet(tmp_pq["avc"], index=False)
        _make_ai_df(ai_rows).to_parquet(tmp_pq["ai"], index=False)
        with (
            patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]),
            patch("corefunc.canon.workflow.ARTIST_INFO_PQ", tmp_pq["ai"]),
        ):
            from corefunc.canon.workflow import propagate_avc

            result = propagate_avc()
            assert result["updated"] >= 1
            ai_after = pd.read_parquet(tmp_pq["ai"])
            assert "The Beatles" in ai_after["artist_name"].values
            assert "Beatles" not in ai_after["artist_name"].values

    def test_skip_rows_are_ignored(self, tmp_pq):
        """Rows with to_link=False are not propagated."""
        sig = _sig(["Battles", "The Beatniks"])
        avc_rows = [
            {
                "artist_variants_hash": _hash(sig),
                "artist_variants_text": sig,
                "canonical_name": "__SKIP__",
                "to_link": False,
                "comment": "",
                "stamp": "2025-05-02T16:49:46+00:00",
            }
        ]
        ai_rows = [
            {"artist_name": "Battles", "mbid": "", "country": "", "disambiguation_comment": "", "aliases": ""},
        ]
        _make_avc_df(avc_rows).to_parquet(tmp_pq["avc"], index=False)
        _make_ai_df(ai_rows).to_parquet(tmp_pq["ai"], index=False)
        with (
            patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]),
            patch("corefunc.canon.workflow.ARTIST_INFO_PQ", tmp_pq["ai"]),
        ):
            from corefunc.canon.workflow import propagate_avc

            result = propagate_avc()
            assert result["updated"] == 0
            assert result["aliases_added"] == 0


# ── undecided_rows tests ─────────────────────────────────────────────────────
class TestUndecidedRows:
    """Tests for undecided_rows()."""

    def test_returns_only_null_to_link(self, tmp_pq):
        """Returns only rows where to_link IS NULL."""
        sig1 = _sig(["A", "B"])
        sig2 = _sig(["C", "D"])
        sig3 = _sig(["E", "F"])
        rows = [
            {
                "artist_variants_hash": _hash(sig1),
                "artist_variants_text": sig1,
                "canonical_name": "A",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:00:00+00:00",
            },
            {
                "artist_variants_hash": _hash(sig2),
                "artist_variants_text": sig2,
                "canonical_name": "__SKIP__",
                "to_link": False,
                "comment": "",
                "stamp": "2025-05-02T17:00:00+00:00",
            },
            {
                "artist_variants_hash": _hash(sig3),
                "artist_variants_text": sig3,
                "canonical_name": "",
                "to_link": pd.NA,
                "comment": "",
                "stamp": "2025-05-02T18:00:00+00:00",
            },
        ]
        _make_avc_df(rows).to_parquet(tmp_pq["avc"], index=False)
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import undecided_rows

            result = undecided_rows()
            assert len(result) == 1
            assert result.iloc[0]["artist_variants_text"] == sig3

    def test_returns_empty_when_all_decided(self, tmp_pq):
        """Returns empty DataFrame when no rows are undecided."""
        sig = _sig(["X", "Y"])
        rows = [
            {
                "artist_variants_hash": _hash(sig),
                "artist_variants_text": sig,
                "canonical_name": "X",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:00:00+00:00",
            }
        ]
        _make_avc_df(rows).to_parquet(tmp_pq["avc"], index=False)
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import undecided_rows

            assert undecided_rows().empty


# ── update_avc_decision tests ────────────────────────────────────────────────
class TestUpdateAvcDecision:
    """Tests for update_avc_decision()."""

    def test_updates_row_in_place(self, tmp_pq):
        """Updates to_link and canonical_name for a given hash."""
        sig = _sig(["M", "N"])
        h = _hash(sig)
        rows = [
            {
                "artist_variants_hash": h,
                "artist_variants_text": sig,
                "canonical_name": "",
                "to_link": pd.NA,
                "comment": "",
                "stamp": "2025-05-02T16:00:00+00:00",
            }
        ]
        _make_avc_df(rows).to_parquet(tmp_pq["avc"], index=False)
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import update_avc_decision

            update_avc_decision(h, True, "M", "test comment")
            after = pd.read_parquet(tmp_pq["avc"])
            row = after.iloc[0]
            assert row["to_link"] is True or row["to_link"] == True  # noqa: E712
            assert row["canonical_name"] == "M"
            assert row["comment"] == "test comment"


# ── _build_exclusion_set tests ───────────────────────────────────────────────
class TestBuildExclusionSet:
    """Tests for _build_exclusion_set()."""

    def test_includes_variants_and_aliases(self, tmp_pq):
        """Collects names from both avc and artist_info."""
        sig = _sig(["Beatles", "The Beatles"])
        avc_rows = [
            {
                "artist_variants_hash": _hash(sig),
                "artist_variants_text": sig,
                "canonical_name": "The Beatles",
                "to_link": True,
                "comment": "",
                "stamp": "2025-05-02T16:00:00+00:00",
            }
        ]
        ai_rows = [
            {"artist_name": "Björk", "mbid": "", "country": "", "disambiguation_comment": "", "aliases": "Bjork{Byork"},
        ]
        _make_avc_df(avc_rows).to_parquet(tmp_pq["avc"], index=False)
        _make_ai_df(ai_rows).to_parquet(tmp_pq["ai"], index=False)
        with (
            patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]),
            patch("corefunc.canon.workflow.ARTIST_INFO_PQ", tmp_pq["ai"]),
        ):
            from corefunc.canon.workflow import _build_exclusion_set

            excl = _build_exclusion_set()
            assert "Beatles" in excl
            assert "The Beatles" in excl
            assert "Björk" in excl
            assert "Bjork" in excl
            assert "Byork" in excl


# ── write_new_candidates tests ───────────────────────────────────────────────
class TestWriteNewCandidates:
    """Tests for write_new_candidates()."""

    def test_writes_with_null_to_link(self, tmp_pq):
        """New candidates are written with to_link=NULL."""
        candidates = [
            {"signature": _sig(["Foo", "Bar"]), "variants": ["Bar", "Foo"], "hash": _hash(_sig(["Foo", "Bar"]))},
        ]
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import write_new_candidates

            n = write_new_candidates(candidates)
            assert n == 1
            df = pd.read_parquet(tmp_pq["avc"])
            assert len(df) == 1
            assert df.iloc[0]["to_link"] is pd.NA

    def test_empty_candidates_writes_nothing(self, tmp_pq):
        """Returns 0 and writes nothing when candidate list is empty."""
        with patch("corefunc.canon.workflow.AVC_PQ", tmp_pq["avc"]):
            from corefunc.canon.workflow import write_new_candidates

            assert write_new_candidates([]) == 0


# ── avc_seed tests ───────────────────────────────────────────────────────────
class TestAvcSeed:
    """Tests for the SQL → Parquet seeder."""

    def test_parse_simple_sql(self, tmp_path):
        """Parses a minimal MySQL dump and produces correct row count."""
        sql = """
INSERT INTO `artist_variants_canonized` VALUES ('aaa','Beatles{The Beatles',1,'The Beatles',NULL,'2025-05-02 16:49:38',NULL),('bbb','Battles{The Beatniks',0,'__SKIP__','diff bands','2025-05-02 16:49:46',NULL);
"""
        sql_file = tmp_path / "test.sql"
        sql_file.write_text(sql, encoding="utf-8")
        avc_pq = tmp_path / "avc.parquet"
        with patch("corefunc.avc_seed.AVC_PQ", avc_pq):
            from corefunc.avc_seed import seed_avc_from_sql

            n = seed_avc_from_sql(sql_file)
            assert n == 2
            df = pd.read_parquet(avc_pq)
            assert len(df) == 2
            assert df.iloc[0]["canonical_name"] == "The Beatles"
            assert df.iloc[1]["to_link"] == False  # noqa: E712

    def test_handles_escaped_quotes(self, tmp_path):
        """Correctly unescapes MySQL single-quote escapes."""
        sql = """
INSERT INTO `artist_variants_canonized` VALUES ('ccc','Trevor Dunn\\'s Trio-Convulsant{Trevor Dunn\\'s Trio Convulsant',1,'Trevor Dunn\\'s Trio-Convulsant',NULL,'2025-05-02 17:05:36',NULL);
"""
        sql_file = tmp_path / "test.sql"
        sql_file.write_text(sql, encoding="utf-8")
        avc_pq = tmp_path / "avc.parquet"
        with patch("corefunc.avc_seed.AVC_PQ", avc_pq):
            from corefunc.avc_seed import seed_avc_from_sql

            n = seed_avc_from_sql(sql_file)
            assert n == 1
            df = pd.read_parquet(avc_pq)
            assert "Trevor Dunn's Trio-Convulsant" in df.iloc[0]["canonical_name"]
