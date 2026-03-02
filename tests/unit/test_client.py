"""
Unit tests for HTTP.client (resilient HTTP GET helper).
"""

from unittest.mock import patch, MagicMock
import requests
from HTTP.client import make_request, USER_AGENT


class TestMakeRequest:
    """Tests make_request retry and error handling."""

    @patch("HTTP.client.requests.get")
    def test_success(self, mock_get):
        """Returns the response on a 200 OK."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_get.return_value = mock_resp
        result = make_request("https://example.com")
        assert result is mock_resp
        mock_get.assert_called_once()

    @patch("HTTP.client.requests.get")
    def test_sets_user_agent(self, mock_get):
        """Sets the default User-Agent header."""
        mock_resp = MagicMock(ok=True)
        mock_get.return_value = mock_resp
        make_request("https://example.com")
        call_headers = mock_get.call_args[1]["headers"]
        assert call_headers["User-Agent"] == USER_AGENT

    @patch("HTTP.client.requests.get")
    def test_custom_headers_preserved(self, mock_get):
        """Merges custom headers without overwriting User-Agent."""
        mock_resp = MagicMock(ok=True)
        mock_get.return_value = mock_resp
        make_request("https://example.com", headers={"X-Custom": "val"})
        call_headers = mock_get.call_args[1]["headers"]
        assert call_headers["X-Custom"] == "val"
        assert "User-Agent" in call_headers

    @patch("HTTP.client.sleep")
    @patch("HTTP.client.requests.get")
    def test_retries_on_5xx(self, mock_get, mock_sleep):
        """Retries on 500-series errors and eventually returns None after max_retries."""
        mock_resp = MagicMock(ok=False, status_code=503)
        mock_get.return_value = mock_resp
        result = make_request("https://example.com", max_retries=2)
        assert result is None
        assert mock_get.call_count == 2

    @patch("HTTP.client.requests.get")
    def test_returns_on_4xx(self, mock_get):
        """Returns the response immediately on 4xx (non-retriable)."""
        mock_resp = MagicMock(ok=False, status_code=404, text="Not found")
        mock_get.return_value = mock_resp
        result = make_request("https://example.com", max_retries=3)
        assert result is mock_resp
        mock_get.assert_called_once()

    @patch("HTTP.client.sleep")
    @patch("HTTP.client.requests.get", side_effect=requests.ConnectionError("timeout"))
    def test_retries_on_network_error(self, mock_get, mock_sleep):
        """Retries on network errors and returns None after exhaustion."""
        result = make_request("https://example.com", max_retries=2)
        assert result is None
        assert mock_get.call_count == 2
