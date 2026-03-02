"""
Tests for the MBDB gold standard augmentation pipeline:
corefunc/canon/augment.py and the augment path in corefunc/canon/model.py.
"""

from __future__ import annotations
from unittest.mock import patch
import pandas as pd
import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def tmp_pq(tmp_path):
    """Provides patched PQ paths pointing to a temporary directory."""
    return {
        "gs_mb": tmp_path / "gs_mb.parquet",
        "avc": tmp_path / "avc.parquet",
        "dir": tmp_path,
    }


def _alias_df(n: int = 5) -> pd.DataFrame:
    """Builds a canned alias→canonical DataFrame as _psql_csv would return."""
    rows = [{"canonical": f"Artist {i}", "alias": f"Artst {i}"} for i in range(n)]
    return pd.DataFrame(rows)


def _exact_neg_df(n: int = 3) -> pd.DataFrame:
    """Builds a canned same-name negative DataFrame."""
    rows = [{"name_a": f"Ambiguous {i}", "name_b": f"Ambiguous {i}"} for i in range(n)]
    return pd.DataFrame(rows)


def _name_pool_df(n: int = 20) -> pd.DataFrame:
    """Builds a canned artist name pool for hard-negative generation."""
    # Including some names that are deliberately similar
    names = [f"Band {chr(65 + i)}" for i in range(n)]
    names[1] = "Band A"  # to duplicate for similarity matching
    return pd.DataFrame({"name": names})


# ── extract_positive_pairs tests ──────────────────────────────────────────────
class TestExtractPositivePairs:
    """Tests for extract_positive_pairs()."""

    @patch("corefunc.canon.augment._psql_csv")
    def test_returns_correct_schema(self, mock_csv):
        """Returns a DataFrame with variant_a, variant_b, to_link, source columns."""
        mock_csv.return_value = _alias_df(5)
        from corefunc.canon.augment import extract_positive_pairs

        result = extract_positive_pairs(limit=5)
        assert list(result.columns) == ["variant_a", "variant_b", "to_link", "source"]
        assert len(result) == 5

    @patch("corefunc.canon.augment._psql_csv")
    def test_all_to_link_true(self, mock_csv):
        """All rows have to_link=True."""
        mock_csv.return_value = _alias_df(3)
        from corefunc.canon.augment import extract_positive_pairs

        result = extract_positive_pairs(limit=3)
        assert result["to_link"].all()
        assert (result["source"] == "mb_alias").all()

    @patch("corefunc.canon.augment._psql_csv")
    def test_maps_alias_and_canonical_correctly(self, mock_csv):
        """Maps alias to variant_a and canonical to variant_b."""
        mock_csv.return_value = pd.DataFrame(
            [
                {"canonical": "The Beatles", "alias": "Beatles"},
            ]
        )
        from corefunc.canon.augment import extract_positive_pairs

        result = extract_positive_pairs(limit=1)
        assert result.iloc[0]["variant_a"] == "Beatles"
        assert result.iloc[0]["variant_b"] == "The Beatles"

    @patch("corefunc.canon.augment._psql_csv")
    def test_returns_empty_on_no_data(self, mock_csv):
        """Returns empty DataFrame with correct columns when MBDB returns nothing."""
        mock_csv.return_value = pd.DataFrame()
        from corefunc.canon.augment import extract_positive_pairs

        result = extract_positive_pairs(limit=5)
        assert result.empty
        assert "variant_a" in result.columns


# ── extract_negative_pairs tests ──────────────────────────────────────────────
class TestExtractNegativePairs:
    """Tests for extract_negative_pairs()."""

    @patch("corefunc.canon.augment._generate_hard_negatives")
    @patch("corefunc.canon.augment._psql_csv")
    def test_combines_exact_and_fuzzy(self, mock_csv, mock_hard):
        """Combines Phase A exact negatives with Phase B hard negatives."""
        mock_csv.return_value = _exact_neg_df(3)
        mock_hard.return_value = pd.DataFrame(
            [
                {"variant_a": "Foo", "variant_b": "Foe", "to_link": False, "source": "mb_neg_fuzzy"},
                {"variant_a": "Bar", "variant_b": "Baz", "to_link": False, "source": "mb_neg_fuzzy"},
            ]
        )
        from corefunc.canon.augment import extract_negative_pairs

        result = extract_negative_pairs(limit=10, similarity_floor=60)
        assert len(result) == 5
        assert (result["to_link"] == False).all()  # noqa: E712
        sources = set(result["source"])
        assert "mb_neg_exact" in sources
        assert "mb_neg_fuzzy" in sources

    @patch("corefunc.canon.augment._generate_hard_negatives")
    @patch("corefunc.canon.augment._psql_csv")
    def test_exact_only_when_limit_small(self, mock_csv, mock_hard):
        """Uses only exact negatives when limit <= _EXACT_NEG_CAP and enough rows."""
        mock_csv.return_value = _exact_neg_df(5)
        mock_hard.return_value = pd.DataFrame(
            columns=["variant_a", "variant_b", "to_link", "source"],
        )
        from corefunc.canon.augment import extract_negative_pairs

        result = extract_negative_pairs(limit=5, similarity_floor=60)
        assert len(result) == 5
        assert (result["source"] == "mb_neg_exact").all()

    @patch("corefunc.canon.augment._psql_csv")
    def test_returns_empty_when_no_data(self, mock_csv):
        """Returns empty DataFrame when MBDB has no same-name pairs."""
        mock_csv.return_value = pd.DataFrame()
        from corefunc.canon.augment import extract_negative_pairs

        result = extract_negative_pairs(limit=5, similarity_floor=60)
        assert result.empty


# ── _generate_hard_negatives tests ────────────────────────────────────────────
class TestGenerateHardNegatives:
    """Tests for _generate_hard_negatives()."""

    @patch("corefunc.canon.augment._psql_csv")
    def test_finds_similar_pairs(self, mock_csv):
        """Finds fuzzy-similar name pairs from the MBID-keyed name pool."""
        # Creating enough names to pass the pool size guard (>= 10),
        # with some deliberately similar ones for RapidFuzz to match
        # and unique MBIDs so pairs are verified as different artists
        pool = [
            {"name": "Beatles", "mbid": "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"},
            {"name": "The Beatles", "mbid": "a2345678-0000-0000-0000-000000000001"},
            {"name": "Beethovens", "mbid": "a2345678-0000-0000-0000-000000000002"},
            {"name": "Radiohead", "mbid": "a7f7df4a-77d8-4f12-8acd-5c60c93f4de8"},
            {"name": "Radioheads", "mbid": "a2345678-0000-0000-0000-000000000004"},
            {"name": "Coldplay", "mbid": "cc197bad-dc9c-440d-a5b5-d52ba2e14234"},
            {"name": "Coldploy", "mbid": "a2345678-0000-0000-0000-000000000006"},
            {"name": "Metallica", "mbid": "65f4f0c5-ef9e-490c-aee3-909e7ae6b2ab"},
            {"name": "Megadeth", "mbid": "a9044915-8be3-4c7e-b11f-9e2d2ea1a91e"},
            {"name": "Nirvana", "mbid": "5b11f4ce-a62d-471e-81fc-a69a8278c7da"},
            {"name": "Pink Floyd", "mbid": "83d91898-7763-47d7-b03b-b92132375c47"},
            {"name": "Fleetwood Mac", "mbid": "a2345678-0000-0000-0000-00000000000b"},
        ]
        mock_csv.return_value = pd.DataFrame(pool)
        from corefunc.canon.augment import _generate_hard_negatives

        result = _generate_hard_negatives(limit=10, similarity_floor=60, neg_limit=10)
        assert not result.empty
        assert (result["to_link"] == False).all()  # noqa: E712
        assert (result["source"] == "mb_neg_fuzzy").all()

    @patch("corefunc.canon.augment._psql_csv")
    def test_returns_empty_on_small_pool(self, mock_csv):
        """Returns empty when the name pool is too small."""
        mock_csv.return_value = pd.DataFrame({"name": ["A", "B"], "mbid": ["m1", "m2"]})
        from corefunc.canon.augment import _generate_hard_negatives

        result = _generate_hard_negatives(limit=10, similarity_floor=60, neg_limit=10)
        assert result.empty

    @patch("corefunc.canon.augment._psql_csv")
    def test_skips_same_mbid_pairs(self, mock_csv):
        """Does not produce pairs where both names share the same MBID."""
        same_mbid = "aaaaaaaa-0000-0000-0000-000000000000"
        pool = [
            {"name": n, "mbid": same_mbid}
            for n in [
                "Beatles",
                "The Beatles",
                "Beethovens",
                "Radiohead",
                "Radioheads",
                "Coldplay",
                "Coldploy",
                "Metallica",
                "Megadeth",
                "Nirvana",
            ]
        ]
        mock_csv.return_value = pd.DataFrame(pool)
        from corefunc.canon.augment import _generate_hard_negatives

        result = _generate_hard_negatives(limit=10, similarity_floor=60, neg_limit=10)
        assert result.empty


# ── augment_gold_standard tests ───────────────────────────────────────────────
class TestAugmentGoldStandard:
    """Tests for augment_gold_standard()."""

    @patch("corefunc.canon.augment.extract_negative_pairs")
    @patch("corefunc.canon.augment.extract_positive_pairs")
    @patch("corefunc.canon.augment.check_local_mb", return_value=True)
    def test_writes_gs_mb_parquet(self, mock_check, mock_pos, mock_neg, tmp_pq):
        """Writes gs_mb.parquet with combined positive and negative pairs."""
        mock_pos.return_value = pd.DataFrame(
            [
                {"variant_a": "A", "variant_b": "B", "to_link": True, "source": "mb_alias"},
            ]
        )
        mock_neg.return_value = pd.DataFrame(
            [
                {"variant_a": "C", "variant_b": "D", "to_link": False, "source": "mb_neg_fuzzy"},
            ]
        )
        with patch("corefunc.canon.augment.GS_MB_PQ", tmp_pq["gs_mb"]):
            from corefunc.canon.augment import augment_gold_standard

            n = augment_gold_standard(pos_limit=1, neg_limit=1)
            assert n == 2
            df = pd.read_parquet(tmp_pq["gs_mb"])
            assert len(df) == 2
            assert set(df["to_link"]) == {True, False}

    @patch("corefunc.canon.augment.check_local_mb", return_value=False)
    def test_raises_when_mbdb_unreachable(self, mock_check):
        """Raises RuntimeError when local MB mirror is unreachable."""
        from corefunc.canon.augment import augment_gold_standard

        with pytest.raises(RuntimeError, match="Cannot reach"):
            augment_gold_standard()


# ── _build_gold_standard augment=True tests ───────────────────────────────────
class TestBuildGoldStandardAugmented:
    """Tests that _build_gold_standard merges MB data when augment=True."""

    @patch("corefunc.canon.model.read_parquet")
    def test_merges_mb_pairs(self, mock_read, tmp_pq):
        """Includes MB pairs alongside AVC pairs when augment=True."""
        # Building a minimal AVC DataFrame
        avc_df = pd.DataFrame(
            [
                {
                    "artist_variants": "Beatles{The Beatles",
                    "canonical_name": "The Beatles",
                    "to_link": True,
                    "comment": "",
                }
            ]
        )
        # Building a minimal gs_mb DataFrame
        mb_df = pd.DataFrame(
            [
                {"variant_a": "Foo", "variant_b": "Foobar", "to_link": True, "source": "mb_alias"},
                {"variant_a": "Bar", "variant_b": "Baz", "to_link": False, "source": "mb_neg_fuzzy"},
            ]
        )
        mb_df.to_parquet(tmp_pq["gs_mb"], index=False)

        def side_effect(path):
            """Returns the right DataFrame based on the file path."""
            if "avc" in str(path):
                return avc_df
            if "gs_mb" in str(path):
                return mb_df
            return None

        mock_read.side_effect = side_effect
        from corefunc.canon.model import _build_gold_standard

        with patch("corefunc.canon.model.GS_MB_PQ", tmp_pq["gs_mb"]):
            gs = _build_gold_standard(augment=True)
            # AVC expands 1 pair + 2 MB pairs = 3 total
            assert len(gs) == 3
            assert "ratio" in gs.columns
            assert "avg_name_len" in gs.columns

    @patch("corefunc.canon.model.read_parquet")
    def test_no_merge_when_augment_false(self, mock_read):
        """Does not load gs_mb.parquet when augment=False."""
        avc_df = pd.DataFrame(
            [
                {
                    "artist_variants": "Beatles{The Beatles",
                    "canonical_name": "The Beatles",
                    "to_link": True,
                    "comment": "",
                }
            ]
        )
        mock_read.return_value = avc_df
        from corefunc.canon.model import _build_gold_standard

        gs = _build_gold_standard(augment=False)
        # Only the AVC pair
        assert len(gs) == 1
        # read_parquet should only be called once (for AVC)
        mock_read.assert_called_once()
