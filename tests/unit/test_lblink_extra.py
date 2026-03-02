"""
Additional unit tests for HTTP.lblink — LBClient facade and _cli helper.
"""

from unittest.mock import patch, MagicMock
import pytest
from HTTP.lblink import LBClient, _RequestsBackend, _cli


class TestLBClientInit:
    """Tests the LBClient constructor."""

    @patch("HTTP.lblink.pylistenbrainz", None)
    def test_falls_back_to_http(self):
        """Uses HTTP backend when pylistenbrainz is unavailable."""
        client = LBClient()
        assert client._backend == "http"
        assert isinstance(client._client, _RequestsBackend)

    @patch("HTTP.lblink.pylistenbrainz")
    def test_uses_pylb_when_available(self, mock_pylb):
        """Uses pylistenbrainz backend when the library is present."""
        mock_pylb.ListenBrainz.return_value = MagicMock()
        client = LBClient()
        assert client._backend == "pylb"

    @patch("HTTP.lblink.pylistenbrainz")
    def test_falls_back_on_type_error(self, mock_pylb):
        """Falls back to HTTP when pylistenbrainz constructor fails."""
        mock_pylb.ListenBrainz.side_effect = TypeError("bad args")
        client = LBClient()
        assert client._backend == "http"


class TestLBClientGetListens:
    """Tests the get_listens facade method."""

    @patch("HTTP.lblink.pylistenbrainz", None)
    def test_http_backend_delegates(self):
        """Delegates to _RequestsBackend.get_listens."""
        client = LBClient()
        client._client = MagicMock()
        client._client.get_listens.return_value = [{"track_metadata": {"track_name": "T1"}}]
        result = client.get_listens("user1", count=1)
        assert len(result) == 1
        client._client.get_listens.assert_called_once()

    @patch("HTTP.lblink.pylistenbrainz")
    def test_pylb_backend_converts(self, mock_pylb):
        """Converts pylistenbrainz objects to dicts."""
        mock_listen = MagicMock()
        mock_listen.to_dict.return_value = {"track_metadata": {"track_name": "T1"}}
        mock_pylb.ListenBrainz.return_value.get_listens.return_value = [mock_listen]
        client = LBClient()
        result = client.get_listens("user1", count=1)
        assert result[0]["track_metadata"]["track_name"] == "T1"


class TestLBClientLookupMetadata:
    """Tests the lookup_metadata facade."""

    @patch("HTTP.lblink.pylistenbrainz", None)
    def test_http_backend(self):
        """Delegates to HTTP backend."""
        client = LBClient()
        client._client = MagicMock()
        client._client.lookup_metadata.return_value = {"recording_name": "T1"}
        result = client.lookup_metadata("T1", "A1")
        assert result["recording_name"] == "T1"

    @patch("HTTP.lblink.pylistenbrainz")
    def test_pylb_backend(self, mock_pylb):
        """Delegates to pylistenbrainz."""
        mock_pylb.ListenBrainz.return_value.lookup_metadata.return_value = {"recording_name": "T1"}
        client = LBClient()
        result = client.lookup_metadata("T1", "A1")
        assert result["recording_name"] == "T1"


class TestLBClientSubmitListens:
    """Tests the submit_listens facade."""

    @patch("HTTP.lblink.pylistenbrainz", None)
    def test_http_backend(self):
        """Delegates to HTTP backend."""
        client = LBClient()
        client._client = MagicMock()
        client.submit_listens({"listen_type": "single"})
        client._client.submit_listens.assert_called_once()

    @patch("HTTP.lblink.pylistenbrainz")
    def test_pylb_backend(self, mock_pylb):
        """Delegates to pylistenbrainz."""
        client = LBClient()
        client.submit_listens({"listen_type": "single"})
        mock_pylb.ListenBrainz.return_value.submit_listens.assert_called_once()


class TestRequestsBackendLookup:
    """Tests _RequestsBackend.lookup_metadata."""

    @patch("HTTP.lblink.requests.Session")
    def test_lookup_with_incs(self, mock_sess_cls):
        """Passes metadata=true and incs to the API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"recording_name": "T1"}
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        backend = _RequestsBackend()
        result = backend.lookup_metadata("T1", "A1", incs="artist")
        assert result["recording_name"] == "T1"

    @patch("HTTP.lblink.requests.Session")
    def test_lookup_without_incs(self, mock_sess_cls):
        """Omits metadata param when incs is None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"recording_name": "T1"}
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        backend = _RequestsBackend()
        result = backend.lookup_metadata("T1", "A1")
        assert result["recording_name"] == "T1"


class TestRequestsBackendSubmit:
    """Tests _RequestsBackend.submit_listens."""

    @patch("HTTP.lblink.LB_TOKEN", "fake-token")
    @patch("HTTP.lblink.requests.Session")
    def test_submit_posts(self, mock_sess_cls):
        """Posts the listen document to the API."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.status_code = 200
        mock_sess_cls.return_value.post.return_value = mock_resp
        backend = _RequestsBackend()
        backend.submit_listens({"listen_type": "single"})
        mock_sess_cls.return_value.post.assert_called_once()

    @patch("HTTP.lblink.LB_TOKEN", None)
    def test_submit_raises_without_token(self):
        """Raises RuntimeError when no token is configured."""
        backend = _RequestsBackend()
        with pytest.raises(RuntimeError, match="TOKEN"):
            backend.submit_listens({"listen_type": "single"})


class TestFetchScrobblesSince:
    """Tests the paginated ListenBrainz fetcher."""

    @patch("HTTP.lblink.requests.Session")
    def test_single_page(self, mock_sess_cls):
        """Fetches a single page of listens and normalises columns."""
        from HTTP.lblink import fetch_scrobbles_since

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "payload": {
                "listens": [
                    {
                        "listened_at": 1700000100,
                        "track_metadata": {
                            "artist_name": "Artist A",
                            "track_name": "Song A",
                            "release_name": "Album A",
                            "additional_info": {"artist_mbids": ["abc-123"]},
                        },
                    },
                    {
                        "listened_at": 1700000000,
                        "track_metadata": {
                            "artist_name": "Artist B",
                            "track_name": "Song B",
                            "release_name": "Album B",
                            "additional_info": {"artist_mbids": []},
                        },
                    },
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        df = fetch_scrobbles_since("testuser")
        assert len(df) == 2
        assert list(df.columns) == ["Artist", "Song", "Album", "uts", "artist_mbid"]
        assert df.iloc[0]["Artist"] == "Artist A"
        assert df.iloc[0]["artist_mbid"] == "abc-123"
        assert df.iloc[1]["artist_mbid"] is None

    @patch("HTTP.lblink.requests.Session")
    def test_empty_result(self, mock_sess_cls):
        """Returns empty DataFrame with correct columns when no listens found."""
        from HTTP.lblink import fetch_scrobbles_since

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"payload": {"listens": []}}
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        df = fetch_scrobbles_since("testuser")
        assert df.empty
        assert list(df.columns) == ["Artist", "Song", "Album", "uts", "artist_mbid"]

    @patch("HTTP.lblink.requests.Session")
    def test_with_since_parameter(self, mock_sess_cls):
        """Passes min_ts to the API when since is provided."""
        from HTTP.lblink import fetch_scrobbles_since

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "payload": {
                "listens": [
                    {
                        "listened_at": 1700000100,
                        "track_metadata": {
                            "artist_name": "A",
                            "track_name": "T",
                            "release_name": "Al",
                            "additional_info": {},
                        },
                    },
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        df = fetch_scrobbles_since("testuser", since=1700000000)
        assert len(df) == 1
        # Verifying min_ts was passed in the GET params
        call_kwargs = mock_sess_cls.return_value.get.call_args
        assert call_kwargs[1]["params"]["min_ts"] == 1700000000

    @patch("HTTP.lblink.requests.Session")
    def test_progress_callback_invoked(self, mock_sess_cls):
        """Invokes progress callback during and after fetch."""
        from HTTP.lblink import fetch_scrobbles_since

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "payload": {
                "listens": [
                    {
                        "listened_at": 1700000000,
                        "track_metadata": {
                            "artist_name": "A",
                            "track_name": "T",
                            "release_name": "Al",
                            "additional_info": {},
                        },
                    },
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        callback = MagicMock()
        fetch_scrobbles_since("testuser", progress_callback=callback)
        assert callback.call_count >= 2  # at least page start + final

    @patch("HTTP.lblink.requests.Session")
    def test_missing_additional_info(self, mock_sess_cls):
        """Handles listens with no additional_info gracefully."""
        from HTTP.lblink import fetch_scrobbles_since

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "payload": {
                "listens": [
                    {
                        "listened_at": 1700000000,
                        "track_metadata": {
                            "artist_name": "A",
                            "track_name": "T",
                            "release_name": "Al",
                        },
                    },
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_sess_cls.return_value.get.return_value = mock_resp
        df = fetch_scrobbles_since("testuser")
        assert len(df) == 1
        assert df.iloc[0]["artist_mbid"] is None


class TestCli:
    """Tests the CLI entry point."""

    @patch("HTTP.lblink.LBClient")
    @patch("HTTP.lblink.sys.exit")
    @patch("argparse.ArgumentParser.parse_args")
    def test_no_listens(self, mock_args, mock_exit, mock_client):
        """Exits gracefully when no listens found."""
        mock_args.return_value = MagicMock(user="testuser", count=5, export=None, verbose=False)
        mock_client.return_value.get_listens.return_value = []
        _cli()
        mock_exit.assert_called_once_with(0)

    @patch("HTTP.lblink.export_listens_to_parquet")
    @patch("HTTP.lblink.LBClient")
    @patch("argparse.ArgumentParser.parse_args")
    def test_export_mode(self, mock_args, mock_client, mock_export):
        """Calls export when --export is provided."""
        mock_args.return_value = MagicMock(user="testuser", count=5, export="out.parquet", verbose=False)
        mock_client.return_value.get_listens.return_value = [
            {"listened_at": 100, "track_metadata": {"artist_name": "A", "track_name": "T"}},
        ]
        _cli()
        mock_export.assert_called_once()

    @patch("builtins.print")
    @patch("HTTP.lblink.LBClient")
    @patch("argparse.ArgumentParser.parse_args")
    def test_display_mode(self, mock_args, mock_client, mock_print):
        """Prints listens when no --export flag."""
        mock_args.return_value = MagicMock(user="testuser", count=5, export=None, verbose=False)
        mock_client.return_value.get_listens.return_value = [
            {"listened_at": 1700000000, "track_metadata": {"artist_name": "A", "track_name": "T"}},
        ]
        _cli()
        assert mock_print.call_count >= 1
