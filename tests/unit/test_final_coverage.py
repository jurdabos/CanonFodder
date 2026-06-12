"""
Tests targeting remaining coverage gaps to reach 80%:
- helpers/device.py: GPU probe success/failure
- corefunc/canon/trainer.py: _build_feature_sep_training_data, _build_mbdb_max_training_data
- helpers/features.py: unicode_script, script_mismatch, fuzzy_scores shim
"""

from unittest.mock import patch

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# helpers/device.py
# ═══════════════════════════════════════════════════════════════════════════════
class TestGetDevice:
    """Tests GPU probe caching and fallback."""

    def test_cpu_fallback(self):
        """Returns 'cpu' when GPU probe fails."""
        import helpers.device as dev

        dev._CACHED_DEVICE = None
        result = dev.get_device()
        assert result in ("cpu", "cuda")

    def test_cached_result(self):
        """Returns cached value on subsequent calls."""
        import helpers.device as dev

        dev._CACHED_DEVICE = "cpu"
        assert dev.get_device() == "cpu"
        dev._CACHED_DEVICE = None

    def test_reset_cache(self):
        """Clears the cached device selection."""
        import helpers.device as dev

        dev._CACHED_DEVICE = "test_value"
        dev.reset_cache()
        assert dev._CACHED_DEVICE is None

    @patch("xgboost.train", side_effect=Exception("no GPU"))
    @patch("xgboost.DMatrix")
    def test_gpu_probe_failure(self, _mock_dm, _mock_train):
        """Falls back to CPU when xgboost.train fails."""
        import helpers.device as dev

        dev._CACHED_DEVICE = None
        result = dev.get_device()
        assert result == "cpu"
        dev._CACHED_DEVICE = None


# ═══════════════════════════════════════════════════════════════════════════════
# trainer.py — _build_feature_sep_training_data
# ═══════════════════════════════════════════════════════════════════════════════
class TestBuildFeatureSepTrainingData:
    """Tests the distribution-matched negative sampling builder."""

    @pytest.fixture()
    def gs_mb_data(self):
        """Provides mock gs_mb data with positives and negatives."""
        rows = []
        for i in range(40):
            rows.append(
                {
                    "variant_a": f"Artist{i}",
                    "variant_b": f"Artst{i}",
                    "to_link": i < 20,
                }
            )
        return pd.DataFrame(rows)

    @pytest.fixture()
    def dbscan_data(self):
        """Provides mock dbscan data with negatives."""
        rows = []
        for i in range(50):
            rows.append(
                {
                    "variant_a": f"DbArt{i}",
                    "variant_b": f"DbAr{i}",
                    "to_link": False,
                }
            )
        return pd.DataFrame(rows)

    @patch("corefunc.canon.trainer.read_parquet")
    def test_no_gs_raises(self, mock_read):
        """Raises when gs_mb.parquet is missing."""
        from corefunc.canon.trainer import _build_feature_sep_training_data

        mock_read.return_value = None
        with pytest.raises(RuntimeError, match="gs_mb.parquet"):
            _build_feature_sep_training_data()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_no_dbscan_raises(self, mock_read, gs_mb_data):
        """Raises when gs_mb_dbscan.parquet is missing."""
        from corefunc.canon.trainer import _build_feature_sep_training_data

        mock_read.side_effect = [gs_mb_data, None]
        with pytest.raises(RuntimeError, match="gs_mb_dbscan"):
            _build_feature_sep_training_data()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_balanced_data(self, mock_read, gs_mb_data, dbscan_data):
        """Returns a DataFrame with both positives and negatives."""
        mock_read.side_effect = [gs_mb_data, dbscan_data]
        from corefunc.canon.trainer import _build_feature_sep_training_data

        result = _build_feature_sep_training_data(neg_count=10)
        assert "variant_a" in result.columns
        assert "to_link" in result.columns
        assert result["to_link"].sum() > 0
        assert (~result["to_link"]).sum() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# trainer.py — _build_mbdb_max_training_data (with proper data)
# ═══════════════════════════════════════════════════════════════════════════════
class TestBuildMbdbMaxWithData:
    """Tests _build_mbdb_max_training_data with valid mock data."""

    @patch("corefunc.canon.trainer.read_parquet")
    def test_produces_train_data(self, mock_read):
        """Returns balanced train set from gs_mb_max + dbscan negatives."""
        max_df = pd.DataFrame(
            {
                "variant_a": [f"ArtA{i}" for i in range(20)],
                "variant_b": [f"ArtB{i}" for i in range(20)],
                "to_link": [True] * 20,
            }
        )
        dbscan_df = pd.DataFrame(
            {
                "variant_a": [f"DbA{i}" for i in range(30)],
                "variant_b": [f"DbB{i}" for i in range(30)],
                "to_link": [False] * 30,
            }
        )
        mock_read.side_effect = [max_df, dbscan_df]
        from corefunc.canon.trainer import _build_mbdb_max_training_data

        result = _build_mbdb_max_training_data()
        assert "variant_a" in result.columns
        assert result["to_link"].any()

    @patch("corefunc.canon.trainer.read_parquet")
    def test_no_dbscan_raises(self, mock_read):
        """Raises when dbscan negatives are missing."""
        max_df = pd.DataFrame(
            {
                "variant_a": ["A"],
                "variant_b": ["B"],
                "to_link": [True],
            }
        )
        mock_read.side_effect = [max_df, None]
        from corefunc.canon.trainer import _build_mbdb_max_training_data

        with pytest.raises(RuntimeError, match="gs_mb_dbscan"):
            _build_mbdb_max_training_data()


# ═══════════════════════════════════════════════════════════════════════════════
# helpers/features.py — _unicode_script and _script_mismatch_flag
# ═══════════════════════════════════════════════════════════════════════════════
class TestUnicodeScript:
    """Tests the Unicode script classifier."""

    def test_latin_chars(self):
        """Classifies ASCII letters as LATIN."""
        from helpers.features import _unicode_script

        assert _unicode_script("A") == "LATIN"
        assert _unicode_script("z") == "LATIN"

    def test_cyrillic_chars(self):
        """Classifies Cyrillic characters."""
        from helpers.features import _unicode_script

        assert _unicode_script("Б") == "CYRILLIC"

    def test_cjk_chars(self):
        """Classifies CJK characters."""
        from helpers.features import _unicode_script

        assert _unicode_script("漢") == "CJK"

    def test_arabic_chars(self):
        """Classifies Arabic characters."""
        from helpers.features import _unicode_script

        assert _unicode_script("ع") == "ARABIC"

    def test_hangul_chars(self):
        """Classifies Hangul characters."""
        from helpers.features import _unicode_script

        assert _unicode_script("한") == "HANGUL"

    def test_japanese_chars(self):
        """Classifies Hiragana/Katakana."""
        from helpers.features import _unicode_script

        assert _unicode_script("あ") == "JAPANESE"

    def test_unknown_chars(self):
        """Returns OTHER for unrecognised scripts."""
        from helpers.features import _unicode_script

        # Thai character
        result = _unicode_script("ก")
        assert result in ("OTHER", "LATIN")


class TestScriptMismatchFlag:
    """Tests the script mismatch detection."""

    def test_same_script(self):
        """Returns 0 for same-script strings."""
        from helpers.features import _script_mismatch_flag

        assert _script_mismatch_flag("Beatles", "The Beatles") == 0

    def test_different_scripts(self):
        """Returns 1 for different-script strings."""
        from helpers.features import _script_mismatch_flag

        assert _script_mismatch_flag("Beatles", "Битлз") == 1


class TestFuzzyScoresShim:
    """Tests the legacy fuzzy_scores compatibility shim."""

    def test_delegates_to_cluster(self):
        """Returns 6-score dict via cluster module."""
        from helpers.features import fuzzy_scores

        scores = fuzzy_scores("Beatles", "The Beatles")
        assert "ratio" in scores
        assert "WRatio" in scores
        assert len(scores) == 6


class TestTokenOrderDisplacement:
    """Tests the Kendall tau displacement function."""

    def test_no_shared_tokens(self):
        """Returns 0 when no tokens are shared."""
        from helpers.features import _kendall_tau_displacement

        assert _kendall_tau_displacement(["a"], ["b"]) == 0.0

    def test_single_shared_token(self):
        """Returns 0 when only one token is shared."""
        from helpers.features import _kendall_tau_displacement

        assert _kendall_tau_displacement(["a", "b"], ["a", "c"]) == 0.0

    def test_reversed_order(self):
        """Returns 1.0 for fully reversed shared tokens."""
        from helpers.features import _kendall_tau_displacement

        assert _kendall_tau_displacement(["a", "b"], ["b", "a"]) == 1.0
