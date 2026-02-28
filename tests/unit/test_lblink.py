"""
Unit tests for HTTP.lblink (ListenBrainz helper).
"""
from unittest.mock import patch, MagicMock
import pandas as pd
from HTTP.lblink import export_listens_to_parquet, _load_token, _RequestsBackend


class TestExportListensToParquet:
    """Tests the listen-to-Parquet exporter."""

    def test_empty_listens_skips(self, tmp_path):
        """Does nothing when the listen list is empty."""
        out = str(tmp_path / "out.parquet")
        export_listens_to_parquet([], out)
        assert not (tmp_path / "out.parquet").exists()

    def test_writes_parquet(self, tmp_path):
        """Writes a valid Parquet file from listen dicts."""
        listens = [
            {
                "listened_at": 1700000000,
                "track_metadata": {"artist_name": "A", "release_name": "Al", "track_name": "T"},
            },
            {
                "listened_at": 1700000100,
                "track_metadata": {"artist_name": "B", "release_name": "Al2", "track_name": "T2"},
            },
        ]
        out = str(tmp_path / "out.parquet")
        export_listens_to_parquet(listens, out)
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert "Artist" in df.columns
        assert df.iloc[0]["Artist"] == "A"


class TestLoadToken:
    """Tests the LB_TOKEN resolution helper."""

    @patch.dict("os.environ", {"LB_TOKEN": "test-token"}, clear=False)
    def test_reads_from_env(self):
        """Returns the token from LB_TOKEN env var."""
        assert _load_token() == "test-token"

    @patch.dict("os.environ", {}, clear=True)
    @patch("HTTP.lblink.Path")
    def test_returns_none_when_missing(self, mock_path_cls):
        """Returns None when no token is found."""
        mock_path_instance = MagicMock()
        mock_path_cls.cwd.return_value.parent.__truediv__ = MagicMock(return_value=mock_path_instance)
        mock_path_instance.exists.return_value = False
        result = _load_token()
        assert result is None


class TestRequestsBackend:
    """Tests the _RequestsBackend HTTP wrapper."""

    @patch("HTTP.lblink.requests.Session")
    def test_get_listens(self, mock_session_cls):
        """Fetches listens via the HTTP backend."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"payload": {"listens": [{"track_metadata": {"track_name": "T"}}]}}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        backend = _RequestsBackend()
        listens = backend.get_listens("testuser", count=1)
        assert len(listens) == 1
