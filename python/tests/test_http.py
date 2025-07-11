"""
Tests for the HTTP client module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from tora._http import HttpClient, HttpResponse
from tora._exceptions import ToraNetworkError, ToraTimeoutError


class TestHttpResponse:
    """Tests for HttpResponse class."""

    def test_basic_response(self):
        """Test basic response creation."""
        # Mock raw response
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = [("content-type", "application/json")]

        data = b'{"message": "success"}'
        url = "https://api.example.com/test"

        response = HttpResponse(raw_response, data, url)

        assert response.status_code == 200
        assert response.reason == "OK"
        assert response.headers == {"content-type": "application/json"}
        assert response.text == '{"message": "success"}'

    def test_json_parsing(self):
        """Test JSON parsing."""
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        data = b'{"key": "value", "number": 42}'
        response = HttpResponse(raw_response, data, "https://api.example.com")

        json_data = response.json()
        assert json_data == {"key": "value", "number": 42}

        # Test caching
        json_data2 = response.json()
        assert json_data2 is json_data

    def test_json_parsing_error(self):
        """Test JSON parsing error."""
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        data = b"invalid json"
        response = HttpResponse(raw_response, data, "https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Failed to decode JSON"):
            response.json()

    def test_text_encoding(self):
        """Test text encoding handling."""
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = [
            ("content-type", "text/plain; charset=utf-8")
        ]

        data = "Hello, 世界!".encode("utf-8")
        response = HttpResponse(raw_response, data, "https://api.example.com")

        assert response.text == "Hello, 世界!"

    def test_text_encoding_fallback(self):
        """Test text encoding fallback."""
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        # Invalid UTF-8 bytes
        data = b"\xff\xfe"
        response = HttpResponse(raw_response, data, "https://api.example.com")

        # Should not raise error, should use replacement characters
        text = response.text
        assert isinstance(text, str)

    def test_raise_for_status_success(self):
        """Test raise_for_status for successful response."""
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        response = HttpResponse(raw_response, b"", "https://api.example.com")

        # Should not raise
        response.raise_for_status()

    def test_raise_for_status_error(self):
        """Test raise_for_status for error response."""
        raw_response = Mock()
        raw_response.status = 404
        raw_response.reason = "Not Found"
        raw_response.getheaders.return_value = []

        response = HttpResponse(
            raw_response, b"Not found", "https://api.example.com/test"
        )

        with pytest.raises(Exception) as exc_info:
            response.raise_for_status()

        assert "HTTP 404 Not Found" in str(exc_info.value)
        assert "https://api.example.com/test" in str(exc_info.value)


class TestHttpClient:
    """Tests for HttpClient class."""

    def test_init_success(self):
        """Test successful client initialization."""
        client = HttpClient(
            "https://api.example.com", headers={"Authorization": "Bearer token"}
        )

        assert client.scheme == "https"
        assert client.netloc == "api.example.com"
        assert client.base_path == ""
        assert client.headers == {"Authorization": "Bearer token"}
        assert client.timeout == 30

    def test_init_with_path(self):
        """Test client initialization with base path."""
        client = HttpClient("https://api.example.com/v1/")

        assert client.base_path == "/v1"

    def test_init_invalid_url(self):
        """Test client initialization with invalid URL."""
        with pytest.raises(ToraNetworkError, match="Invalid base URL"):
            HttpClient("invalid-url")

    def test_init_unsupported_scheme(self):
        """Test client initialization with unsupported scheme."""
        with pytest.raises(ToraNetworkError, match="Unsupported URL scheme"):
            HttpClient("ftp://example.com")

    def test_init_custom_timeout(self):
        """Test client initialization with custom timeout."""
        client = HttpClient("https://api.example.com", timeout=60)
        assert client.timeout == 60

    @patch("tora._http.http.client.HTTPSConnection")
    def test_get_request(self, mock_connection_class):
        """Test GET request."""
        # Setup mock connection
        mock_conn = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.reason = "OK"
        mock_response.getheaders.return_value = []
        mock_response.read.return_value = b'{"result": "success"}'
        mock_conn.request.return_value = None
        mock_conn.getresponse.return_value = mock_response
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")
        response = client.get("/test", headers={"Custom": "header"})

        # Verify connection setup
        mock_connection_class.assert_called_once_with("api.example.com", timeout=30)

        # Verify request
        mock_conn.request.assert_called_once_with(
            "GET", "/test", None, headers={"Custom": "header"}
        )

        assert response.status_code == 200

    @patch("tora._http.http.client.HTTPSConnection")
    def test_post_request_json(self, mock_connection_class):
        """Test POST request with JSON data."""
        # Setup mock connection
        mock_conn = Mock()
        mock_response = Mock()
        mock_response.status = 201
        mock_response.reason = "Created"
        mock_response.getheaders.return_value = []
        mock_response.read.return_value = b'{"id": "123"}'
        mock_conn.request.return_value = None
        mock_conn.getresponse.return_value = mock_response
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")
        json_data = {"name": "test", "value": 42}
        response = client.post("/create", json=json_data)

        # Verify request
        expected_body = json.dumps(json_data).encode("utf-8")
        mock_conn.request.assert_called_once_with(
            "POST",
            "/create",
            expected_body,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 201

    @patch("tora._http.http.client.HTTPSConnection")
    def test_post_request_data(self, mock_connection_class):
        """Test POST request with raw data."""
        # Setup mock connection
        mock_conn = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.reason = "OK"
        mock_response.getheaders.return_value = []
        mock_response.read.return_value = b"OK"
        mock_conn.request.return_value = None
        mock_conn.getresponse.return_value = mock_response
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")
        response = client.post("/upload", data=b"binary data")

        # Verify request
        mock_conn.request.assert_called_once_with(
            "POST", "/upload", b"binary data", headers={}
        )

    def test_post_invalid_json(self):
        """Test POST request with invalid JSON data."""
        client = HttpClient("https://api.example.com")

        # Object that can't be JSON serialized
        invalid_data = {"key": object()}

        with pytest.raises(ToraNetworkError, match="Failed to serialize JSON"):
            client.post("/test", json=invalid_data)

    def test_post_invalid_data_type(self):
        """Test POST request with invalid data type."""
        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Invalid data type"):
            client.post("/test", data=123)

    @patch("tora._http.http.client.HTTPSConnection")
    def test_network_error_handling(self, mock_connection_class):
        """Test network error handling."""
        # Setup mock to raise socket error
        mock_conn = Mock()
        mock_conn.request.side_effect = OSError("Connection failed")
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Network error"):
            client.get("/test")

    @patch("tora._http.http.client.HTTPSConnection")
    def test_timeout_error_handling(self, mock_connection_class):
        """Test timeout error handling."""
        import socket

        # Setup mock to raise timeout
        mock_conn = Mock()
        mock_conn.request.side_effect = socket.timeout("Request timed out")
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraTimeoutError, match="timed out"):
            client.get("/test")

    @patch("tora._http.http.client.HTTPSConnection")
    def test_connection_creation_error(self, mock_connection_class):
        """Test connection creation error."""
        import socket

        # Setup mock to raise error on connection creation
        mock_connection_class.side_effect = socket.gaierror("Name resolution failed")

        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Failed to create connection"):
            client.get("/test")

    def test_context_manager(self):
        """Test context manager usage."""
        with HttpClient("https://api.example.com") as client:
            assert client.conn is not None

        # Connection should be closed after context
        assert client.conn is None

    def test_close_method(self):
        """Test close method."""
        client = HttpClient("https://api.example.com")

        # Mock connection
        mock_conn = Mock()
        client.conn = mock_conn

        client.close()

        mock_conn.close.assert_called_once()
        assert client.conn is None

    def test_close_with_error(self):
        """Test close method when connection close raises error."""
        client = HttpClient("https://api.example.com")

        # Mock connection that raises error on close
        mock_conn = Mock()
        mock_conn.close.side_effect = Exception("Close error")
        client.conn = mock_conn

        # Should not raise error
        client.close()
        assert client.conn is None
