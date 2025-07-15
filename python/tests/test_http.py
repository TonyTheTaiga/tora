import json
from unittest.mock import Mock, patch

import pytest

from tora._exceptions import ToraNetworkError, ToraTimeoutError
from tora._http import HttpClient, HttpResponse


class TestHttpResponse:
    def test_basic_response(self):
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
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        data = b'{"key": "value", "number": 42}'
        response = HttpResponse(raw_response, data, "https://api.example.com")

        json_data = response.json()
        assert json_data == {"key": "value", "number": 42}

        json_data2 = response.json()
        assert json_data2 is json_data

    def test_json_parsing_error(self):
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        data = b"invalid json"
        response = HttpResponse(raw_response, data, "https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Failed to decode JSON"):
            response.json()

    def test_text_encoding(self):
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = [("content-type", "text/plain; charset=utf-8")]

        data = "Hello, 世界!".encode()
        response = HttpResponse(raw_response, data, "https://api.example.com")

        assert response.text == "Hello, 世界!"

    def test_text_encoding_fallback(self):
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        data = b"\xff\xfe"
        response = HttpResponse(raw_response, data, "https://api.example.com")

        text = response.text
        assert isinstance(text, str)

    def test_raise_for_status_success(self):
        raw_response = Mock()
        raw_response.status = 200
        raw_response.reason = "OK"
        raw_response.getheaders.return_value = []

        response = HttpResponse(raw_response, b"", "https://api.example.com")

        response.raise_for_status()

    def test_raise_for_status_error(self):
        raw_response = Mock()
        raw_response.status = 404
        raw_response.reason = "Not Found"
        raw_response.getheaders.return_value = []

        response = HttpResponse(raw_response, b"Not found", "https://api.example.com/test")

        with pytest.raises(Exception) as exc_info:
            response.raise_for_status()

        assert "HTTP 404 Not Found" in str(exc_info.value)
        assert "https://api.example.com/test" in str(exc_info.value)


class TestHttpClient:
    def test_init_success(self):
        client = HttpClient("https://api.example.com", headers={"Authorization": "Bearer token"})

        assert client.scheme == "https"
        assert client.netloc == "api.example.com"
        assert client.base_path == ""
        assert client.headers == {"Authorization": "Bearer token"}
        assert client.timeout == 30

    def test_init_with_path(self):
        client = HttpClient("https://api.example.com/v1/")

        assert client.base_path == "/v1"

    def test_init_invalid_url(self):
        with pytest.raises(ToraNetworkError, match="Invalid base URL"):
            HttpClient("invalid-url")

    def test_init_unsupported_scheme(self):
        with pytest.raises(ToraNetworkError, match="Unsupported URL scheme"):
            HttpClient("ftp://example.com")

    def test_init_custom_timeout(self):
        client = HttpClient("https://api.example.com", timeout=60)
        assert client.timeout == 60

    @patch("tora._http.http.client.HTTPSConnection")
    def test_get_request(self, mock_connection_class):
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

        mock_connection_class.assert_called_once_with("api.example.com", timeout=30)
        mock_conn.request.assert_called_once_with("GET", "/test", None, headers={"Custom": "header"})

        assert response.status_code == 200

    @patch("tora._http.http.client.HTTPSConnection")
    def test_post_request_json(self, mock_connection_class):
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
        client.post("/upload", data=b"binary data")

        mock_conn.request.assert_called_once_with("POST", "/upload", b"binary data", headers={})

    def test_post_invalid_json(self):
        client = HttpClient("https://api.example.com")

        invalid_data = {"key": object()}

        with pytest.raises(ToraNetworkError, match="Failed to serialize JSON"):
            client.post("/test", json=invalid_data)

    def test_post_invalid_data_type(self):
        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Invalid data type"):
            client.post("/test", data=123)

    @patch("tora._http.http.client.HTTPSConnection")
    def test_network_error_handling(self, mock_connection_class):
        mock_conn = Mock()
        mock_conn.request.side_effect = OSError("Connection failed")
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Network error"):
            client.get("/test")

    @patch("tora._http.http.client.HTTPSConnection")
    def test_timeout_error_handling(self, mock_connection_class):
        mock_conn = Mock()
        mock_conn.request.side_effect = TimeoutError("Request timed out")
        mock_connection_class.return_value = mock_conn

        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraTimeoutError, match="timed out"):
            client.get("/test")

    @patch("tora._http.http.client.HTTPSConnection")
    def test_connection_creation_error(self, mock_connection_class):
        import socket

        mock_connection_class.side_effect = socket.gaierror("Name resolution failed")

        client = HttpClient("https://api.example.com")

        with pytest.raises(ToraNetworkError, match="Failed to create connection"):
            client.get("/test")

    def test_context_manager(self):
        with HttpClient("https://api.example.com") as client:
            assert client.conn is not None

        assert client.conn is None

    def test_close_method(self):
        client = HttpClient("https://api.example.com")

        mock_conn = Mock()
        client.conn = mock_conn

        client.close()

        mock_conn.close.assert_called_once()
        assert client.conn is None

    def test_close_with_error(self):
        client = HttpClient("https://api.example.com")

        mock_conn = Mock()
        mock_conn.close.side_effect = Exception("Close error")
        client.conn = mock_conn

        client.close()
        assert client.conn is None
