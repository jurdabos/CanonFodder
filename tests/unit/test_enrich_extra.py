"""
Tests for corefunc.enrich and corefunc.canon.workflow extras, and TCN building blocks.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════════════
# corefunc.enrich
# ═══════════════════════════════════════════════════════════════════════════════
class TestBackfillMbids:
    """Tests MBID backfill from artist_info → scrobble."""

    def test_no_scrobbles(self, tmp_pq_dir):
        """Returns 0 when scrobble data is missing."""
        from corefunc.enrich import backfill_mbids
        assert backfill_mbids() == 0

    def test_no_artist_info(self, tmp_pq_dir, sample_scrobble_df):
        """Returns 0 when artist_info is missing."""
        from helpers.io import dump_scrobble_df
        dump_scrobble_df(sample_scrobble_df)
        from corefunc.enrich import backfill_mbids
        assert backfill_mbids() == 0

    def test_backfills_missing(self, tmp_pq_dir, sample_artist_info_df):
        """Fills missing MBIDs from artist_info lookup."""
        scrobbles = pd.DataFrame({
            "artist_name": ["Bohren & der Club of Gore", "Unknown"],
            "album_title": ["X", "Y"],
            "track_title": ["A", "B"],
            "artist_mbid": [None, None],
            "play_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        })
        from helpers.io import dump_scrobble_df, dump_parquet, ARTIST_INFO_PQ, read_scrobble_df
        dump_scrobble_df(scrobbles)
        dump_parquet(sample_artist_info_df, ARTIST_INFO_PQ)
        from corefunc.enrich import backfill_mbids
        n = backfill_mbids()
        assert n == 1
        updated = read_scrobble_df()
        filled = updated[updated["artist_name"] == "Bohren & der Club of Gore"]["artist_mbid"].iloc[0]
        assert filled == "a4074512-87e0-4820-b609-0c4a18142a70"

    def test_nothing_to_backfill(self, tmp_pq_dir, sample_scrobble_df, sample_artist_info_df):
        """Returns 0 when all MBIDs are already present."""
        from helpers.io import dump_scrobble_df, dump_parquet, ARTIST_INFO_PQ
        dump_scrobble_df(sample_scrobble_df)
        dump_parquet(sample_artist_info_df, ARTIST_INFO_PQ)
        from corefunc.enrich import backfill_mbids
        assert backfill_mbids() == 0


class TestEnrichAll:
    """Tests the unified enrichment orchestrator."""

    @patch("corefunc.enrich.backfill_mbids", return_value=5)
    @patch("corefunc.enrich.enrich_artist_country", return_value=10)
    def test_mbapi_backend(self, mock_enrich, mock_backfill):
        """Uses remote MB API when backend='mbapi'."""
        from corefunc.enrich import enrich_all
        result = enrich_all(backend="mbapi")
        assert result["artist_info_rows"] == 10
        assert result["mbids_backfilled"] == 5

    @patch("corefunc.enrich.backfill_mbids", return_value=0)
    def test_local_backend(self, mock_backfill):
        """Uses local MB mirror when backend='local'."""
        with patch("corefunc.mb_local.enrich_from_local_mb", return_value=7) as mock_local:
            from corefunc.enrich import enrich_all
            result = enrich_all(backend="local")
            assert result["artist_info_rows"] == 7
            mock_local.assert_called_once()

    @patch("corefunc.enrich.backfill_mbids", return_value=0)
    @patch("corefunc.enrich.enrich_artist_country", return_value=3)
    def test_lastfmapi_backend(self, mock_enrich, mock_backfill):
        """Uses Last.fm + MB API when backend='lastfmapi'."""
        with patch("HTTP.lfAPI.enrich_artist_mbids", return_value={"message": "ok"}):
            from corefunc.enrich import enrich_all
            result = enrich_all(backend="lastfmapi")
            assert result["artist_info_rows"] == 3

    def test_invalid_backend(self):
        """Raises ValueError for unknown backend."""
        from corefunc.enrich import enrich_all
        with pytest.raises(ValueError, match="Unknown enrichment backend"):
            enrich_all(backend="imaginary")


# ═══════════════════════════════════════════════════════════════════════════════
# corefunc.canon.workflow extras
# ═══════════════════════════════════════════════════════════════════════════════
class TestCanonWorkflowHelpers:
    """Tests helper functions in corefunc.canon.workflow."""

    def test_make_signature(self):
        """Produces sorted, separator-joined signature."""
        from corefunc.canon.workflow import _make_signature
        sig = _make_signature(["Beta", "Alpha", "Gamma"])
        assert sig == "Alpha{Beta{Gamma"

    def test_make_hash(self):
        """Produces a sha256 hex digest."""
        from corefunc.canon.workflow import _make_hash
        h = _make_hash("Alpha{Beta")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_build_exclusion_set(self, tmp_pq_dir, sample_artist_info_df):
        """Gathers names from avc + artist_info."""
        from helpers.io import dump_parquet, AVC_PQ, ARTIST_INFO_PQ
        avc = pd.DataFrame({
            "artist_variants_hash": ["h1"],
            "artist_variants_text": ["Alpha{Alfa"],
            "canonical_name": ["Alpha"],
            "to_link": [True],
            "comment": [""],
            "stamp": pd.to_datetime(["2024-01-01"], utc=True),
        })
        dump_parquet(avc, AVC_PQ)
        dump_parquet(sample_artist_info_df, ARTIST_INFO_PQ)
        from corefunc.canon.workflow import _build_exclusion_set
        exc = _build_exclusion_set()
        assert "Alpha" in exc
        assert "Alfa" in exc
        assert "Bohren & der Club of Gore" in exc

    def test_log_predictions(self, tmp_pq_dir):
        """Appends prediction records to predictions_log.parquet."""
        import json
        from corefunc.canon.workflow import _log_predictions, PREDICTIONS_LOG_PQ
        rows = [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "variant_a": "A",
                "variant_b": "B",
                "probability": 0.9,
                "features_json": json.dumps({"wratio": 0.85, "partial_ratio": 0.9}),
            },
        ]
        _log_predictions(rows)
        assert PREDICTIONS_LOG_PQ.exists()
        result = pd.read_parquet(PREDICTIONS_LOG_PQ)
        assert "features_json" in result.columns
        assert "variant_a" in result.columns

    def test_log_predictions_empty(self, tmp_pq_dir):
        """Does nothing for empty prediction list."""
        from corefunc.canon.workflow import _log_predictions, PREDICTIONS_LOG_PQ
        # Ensuring clean state
        if PREDICTIONS_LOG_PQ.exists():
            PREDICTIONS_LOG_PQ.unlink()
        _log_predictions([])
        assert not PREDICTIONS_LOG_PQ.exists()

    @patch("mlflow.get_experiment_by_name", return_value=None)
    @patch("mlflow.set_tracking_uri")
    def test_list_mlflow_runs_no_experiment(self, _mock_uri, _mock_get):
        """Returns empty list when experiment does not exist."""
        from corefunc.canon.workflow import list_mlflow_runs
        assert list_mlflow_runs("nonexistent") == []

    @patch("mlflow.sklearn.load_model", return_value="fake_model")
    @patch("mlflow.set_tracking_uri")
    def test_load_run_model(self, _mock_uri, mock_load):
        """Delegates to mlflow.sklearn.load_model."""
        from corefunc.canon.workflow import load_run_model
        result = load_run_model("run123")
        mock_load.assert_called_once_with("runs:/run123/model")
        assert result == "fake_model"


# ═══════════════════════════════════════════════════════════════════════════════
# corefunc.canon.tcn_trainer building blocks
# ═══════════════════════════════════════════════════════════════════════════════
class TestCharVocab:
    """Tests the character vocabulary encoder."""

    def test_fit_and_encode(self):
        """Builds vocabulary and encodes strings with padding."""
        from corefunc.canon.tcn_trainer import CharVocab
        vocab = CharVocab()
        vocab.fit(["abc", "xyz"])
        encoded = vocab.encode("abc", max_len=5)
        assert len(encoded) == 5
        assert encoded[3] == 0  # PAD
        assert vocab.size >= 8  # 6 unique chars + PAD + UNK

    def test_unknown_char(self):
        """Maps unseen characters to UNK."""
        from corefunc.canon.tcn_trainer import CharVocab
        vocab = CharVocab()
        vocab.fit(["abc"])
        encoded = vocab.encode("xyz", max_len=3)
        assert all(idx == CharVocab.UNK for idx in encoded)

    def test_truncation(self):
        """Truncates strings longer than max_len."""
        from corefunc.canon.tcn_trainer import CharVocab
        vocab = CharVocab()
        vocab.fit(["abcdef"])
        encoded = vocab.encode("abcdef", max_len=3)
        assert len(encoded) == 3


class TestTemporalBlocks:
    """Tests TCN building blocks with tiny dimensions."""

    def test_chomp1d(self):
        """Removes trailing padding from tensor."""
        import torch
        from corefunc.canon.tcn_trainer import Chomp1d
        chomp = Chomp1d(2)
        x = torch.randn(1, 4, 10)
        out = chomp(x)
        assert out.shape == (1, 4, 8)

    def test_temporal_block_forward(self):
        """Runs a forward pass through a TemporalBlock."""
        import torch
        from corefunc.canon.tcn_trainer import TemporalBlock
        block = TemporalBlock(4, 8, kernel_size=3, stride=1, dilation=1, padding=2, dropout=0.0)
        x = torch.randn(2, 4, 16)
        out = block(x)
        assert out.shape == (2, 8, 16)

    def test_temporal_conv_net(self):
        """Runs a forward pass through a full TemporalConvNet."""
        import torch
        from corefunc.canon.tcn_trainer import TemporalConvNet
        tcn = TemporalConvNet(num_inputs=4, num_channels=[8, 8], kernel_size=3, dropout=0.0)
        x = torch.randn(2, 4, 16)
        out = tcn(x)
        assert out.shape == (2, 8, 16)

    def test_temporal_conv_net_with_layer_norm(self):
        """Runs TCN with layer norm enabled."""
        import torch
        from corefunc.canon.tcn_trainer import TemporalConvNet
        tcn = TemporalConvNet(
            num_inputs=4, num_channels=[8], kernel_size=3,
            dropout=0.0, use_layer_norm=True,
        )
        x = torch.randn(2, 4, 16)
        out = tcn(x)
        assert out.shape == (2, 8, 16)


class TestNamePairDataset:
    """Tests the Siamese TCN dataset wrapper."""

    def test_len_and_getitem(self):
        """Returns correct length and item shapes."""
        import torch
        from corefunc.canon.tcn_trainer import CharVocab, NamePairDataset
        vocab = CharVocab()
        vocab.fit(["abc", "xyz"])
        df = pd.DataFrame({
            "variant_a": ["abc", "xyz"],
            "variant_b": ["xyz", "abc"],
            "to_link": [1, 0],
        })
        ds = NamePairDataset(df, vocab, max_len=8)
        assert len(ds) == 2
        a, b, label = ds[0]
        assert a.shape == (8,)
        assert b.shape == (8,)
        assert label.dtype == torch.float32


class TestHybridDataset:
    """Tests the hybrid TCN + features dataset wrapper."""

    def test_len_and_getitem(self):
        """Returns correct length and item shapes."""
        from corefunc.canon.tcn_trainer import CharVocab, HybridDataset
        vocab = CharVocab()
        vocab.fit(["abc", "xyz"])
        df = pd.DataFrame({
            "variant_a": ["abc", "xyz"],
            "variant_b": ["xyz", "abc"],
            "to_link": [1, 0],
        })
        features = np.array([[0.5, 0.6], [0.7, 0.8]])
        ds = HybridDataset(df, vocab, max_len=8, features=features)
        assert len(ds) == 2
        a, b, feats, label = ds[0]
        assert a.shape == (8,)
        assert feats.shape == (2,)
