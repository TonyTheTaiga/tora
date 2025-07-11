"""
Tests for the main Tora client.
"""

from unittest.mock import Mock, patch

import pytest

from tora import Tora, create_workspace
from tora._exceptions import (
    HTTPStatusError,
    ToraAuthenticationError,
    ToraConfigurationError,
    ToraExperimentError,
    ToraMetricError,
    ToraValidationError,
)


class TestCreateWorkspace:
    """Tests for create_workspace function."""

    @patch("tora._client.HttpClient")
    def test_create_workspace_success(self, mock_http_client_class, sample_workspace_data):
        """Test successful workspace creation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"status": 201, "data": sample_workspace_data}
        mock_client.post.return_value = mock_response
        mock_http_client_class.return_value.__enter__.return_value = mock_client

        # Test
        result = create_workspace("test-workspace", "Test description", "test-api-key")

        # Assertions
        assert result == sample_workspace_data
        mock_client.post.assert_called_once_with(
            "/workspaces",
            json={"name": "test-workspace", "description": "Test description"},
        )

    def test_create_workspace_no_api_key(self, monkeypatch):
        """Test workspace creation without API key."""
        # Clear any existing API key from environment
        monkeypatch.delenv("TORA_API_KEY", raising=False)

        with pytest.raises(ToraAuthenticationError, match="API key is required"):
            create_workspace("test-workspace")

    def test_create_workspace_invalid_name(self):
        """Test workspace creation with invalid name."""
        with pytest.raises(ToraValidationError, match="must be a non-empty string"):
            create_workspace("", api_key="test-key")

    def test_create_workspace_name_too_long(self):
        """Test workspace creation with name too long."""
        long_name = "a" * 256
        with pytest.raises(ToraValidationError, match="cannot exceed 255 characters"):
            create_workspace(long_name, api_key="test-key")

    @patch("tora._client.HttpClient")
    def test_create_workspace_api_error(self, mock_http_client_class):
        """Test workspace creation API error."""
        # Setup mock to raise HTTP error
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = HTTPStatusError("Bad Request", mock_response)
        mock_client.post.return_value = mock_response
        mock_http_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(ToraValidationError, match="Invalid workspace data"):
            create_workspace("test-workspace", api_key="test-key")


class TestToraInit:
    """Tests for Tora class initialization."""

    def test_init_success(self):
        """Test successful initialization."""
        tora = Tora("exp-123", api_key="test-key")
        assert tora.experiment_id == "exp-123"
        assert tora.max_buffer_len == 25
        assert not tora.is_closed

    def test_init_invalid_experiment_id(self):
        """Test initialization with invalid experiment ID."""
        with pytest.raises(ToraValidationError, match="must be a non-empty string"):
            Tora("")

        with pytest.raises(ToraValidationError, match="must be a non-empty string"):
            Tora(None)

    def test_init_no_server_url(self):
        """Test initialization without server URL."""
        with pytest.raises(ToraConfigurationError, match="Server URL must be provided"):
            Tora("exp-123", server_url="")

    def test_init_custom_buffer_len(self):
        """Test initialization with custom buffer length."""
        tora = Tora("exp-123", max_buffer_len=10, api_key="test-key")
        assert tora.max_buffer_len == 10

    def test_init_buffer_len_minimum(self):
        """Test initialization with buffer length less than 1."""
        tora = Tora("exp-123", max_buffer_len=0, api_key="test-key")
        assert tora.max_buffer_len == 1  # Should be set to minimum of 1


class TestToraCreateExperiment:
    """Tests for Tora.create_experiment class method."""

    @patch("tora._client.HttpClient")
    def test_create_experiment_success(self, mock_http_client_class, sample_experiment_data):
        """Test successful experiment creation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": 201,
            "data": sample_experiment_data,
        }
        mock_client.post.return_value = mock_response
        mock_http_client_class.return_value.__enter__.return_value = mock_client

        # Test
        tora = Tora.create_experiment(
            "test-experiment",
            workspace_id="ws-123",
            description="Test description",
            hyperparams={"lr": 0.01},
            tags=["test"],
            api_key="test-key",
        )

        # Assertions
        assert isinstance(tora, Tora)
        assert tora.experiment_id == "exp-123"
        mock_client.post.assert_called_once()

    def test_create_experiment_invalid_name(self):
        """Test experiment creation with invalid name."""
        with pytest.raises(ToraValidationError):
            Tora.create_experiment("", api_key="test-key")

    def test_create_experiment_invalid_hyperparams(self):
        """Test experiment creation with invalid hyperparams."""
        with pytest.raises(ToraValidationError):
            Tora.create_experiment("test", hyperparams={"": "value"}, api_key="test-key")

    @patch("tora._client.HttpClient")
    def test_create_experiment_auth_error(self, mock_http_client_class):
        """Test experiment creation with authentication error."""
        # Setup mock to raise 401 error
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = HTTPStatusError("Unauthorized", mock_response)
        mock_client.post.return_value = mock_response
        mock_http_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(ToraAuthenticationError, match="Invalid API key"):
            Tora.create_experiment("test", api_key="invalid-key")


class TestToraLoadExperiment:
    """Tests for Tora.load_experiment class method."""

    @patch("tora._client.HttpClient")
    def test_load_experiment_success(self, mock_http_client_class, sample_experiment_data):
        """Test successful experiment loading."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": 200,
            "data": sample_experiment_data,
        }
        mock_client.get.return_value = mock_response
        mock_http_client_class.return_value.__enter__.return_value = mock_client

        # Test
        tora = Tora.load_experiment("exp-123", api_key="test-key")

        # Assertions
        assert isinstance(tora, Tora)
        assert tora.experiment_id == "exp-123"
        mock_client.get.assert_called_once_with("/experiments/exp-123")

    def test_load_experiment_invalid_id(self):
        """Test loading experiment with invalid ID."""
        with pytest.raises(ToraValidationError, match="must be a non-empty string"):
            Tora.load_experiment("")

    @patch("tora._client.HttpClient")
    def test_load_experiment_not_found(self, mock_http_client_class):
        """Test loading non-existent experiment."""
        # Setup mock to raise 404 error
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPStatusError("Not Found", mock_response)
        mock_client.get.return_value = mock_response
        mock_http_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(ToraExperimentError, match="not found"):
            Tora.load_experiment("nonexistent", api_key="test-key")


class TestToraLogging:
    """Tests for Tora metric logging."""

    def test_log_metric_success(self):
        """Test successful metric logging."""
        tora = Tora("exp-123", max_buffer_len=5, api_key="test-key")

        # Log a metric
        tora.log("accuracy", 0.95, step=100)

        # Check buffer
        assert tora.buffer_size == 1
        assert tora._buffer[0]["name"] == "accuracy"
        assert tora._buffer[0]["value"] == 0.95
        assert tora._buffer[0]["step"] == 100

    def test_log_metric_validation(self):
        """Test metric logging validation."""
        tora = Tora("exp-123", api_key="test-key")

        # Invalid metric name
        with pytest.raises(ToraValidationError):
            tora.log("", 0.95)

        # Invalid metric value
        with pytest.raises(ToraValidationError):
            tora.log("accuracy", float("nan"))

        # Invalid step
        with pytest.raises(ToraValidationError):
            tora.log("accuracy", 0.95, step=-1)

    def test_log_metric_closed_client(self):
        """Test logging on closed client."""
        tora = Tora("exp-123", api_key="test-key")
        tora.shutdown()

        with pytest.raises(ToraMetricError, match="closed Tora client"):
            tora.log("accuracy", 0.95)

    @patch("tora._client.Tora._write_logs")
    def test_log_metric_auto_flush(self, mock_write_logs):
        """Test automatic flushing when buffer is full."""
        tora = Tora("exp-123", max_buffer_len=2, api_key="test-key")

        # Log metrics to fill buffer
        tora.log("metric1", 1.0)
        assert mock_write_logs.call_count == 0

        tora.log("metric2", 2.0)
        assert mock_write_logs.call_count == 1  # Should auto-flush

    def test_log_metric_with_metadata(self):
        """Test logging metric with metadata."""
        tora = Tora("exp-123", api_key="test-key")
        metadata = {"epoch": 1, "batch": 10}

        tora.log("loss", 0.5, metadata=metadata)

        assert tora._buffer[0]["metadata"] == metadata

    def test_log_metric_invalid_metadata(self):
        """Test logging with invalid metadata."""
        tora = Tora("exp-123", api_key="test-key")

        # Non-dict metadata
        with pytest.raises(ToraValidationError, match="must be a dictionary"):
            tora.log("loss", 0.5, metadata="invalid")


class TestToraFlushAndShutdown:
    """Tests for Tora flush and shutdown methods."""

    @patch("tora._client.Tora._write_logs")
    def test_flush(self, mock_write_logs):
        """Test manual flush."""
        tora = Tora("exp-123", api_key="test-key")
        tora.log("accuracy", 0.95)

        tora.flush()
        mock_write_logs.assert_called_once()

    def test_flush_closed_client(self):
        """Test flush on closed client."""
        tora = Tora("exp-123", api_key="test-key")
        tora.shutdown()

        with pytest.raises(ToraMetricError, match="closed Tora client"):
            tora.flush()

    @patch("tora._client.Tora._write_logs")
    def test_shutdown(self, mock_write_logs):
        """Test shutdown."""
        tora = Tora("exp-123", api_key="test-key")
        tora.log("accuracy", 0.95)

        tora.shutdown()

        assert tora.is_closed
        mock_write_logs.assert_called_once()

    def test_shutdown_idempotent(self):
        """Test that shutdown can be called multiple times."""
        tora = Tora("exp-123", api_key="test-key")

        tora.shutdown()
        tora.shutdown()  # Should not raise error

        assert tora.is_closed

    @patch("tora._client.Tora._write_logs")
    def test_context_manager(self, mock_write_logs):
        """Test context manager usage."""
        with Tora("exp-123", api_key="test-key") as tora:
            tora.log("accuracy", 0.95)

        # Should auto-shutdown
        assert tora.is_closed
        mock_write_logs.assert_called_once()


class TestToraProperties:
    """Tests for Tora properties."""

    def test_max_buffer_len_property(self):
        """Test max_buffer_len property."""
        tora = Tora("exp-123", api_key="test-key")

        assert tora.max_buffer_len == 25

        tora.max_buffer_len = 10
        assert tora.max_buffer_len == 10

    def test_max_buffer_len_validation(self):
        """Test max_buffer_len validation."""
        tora = Tora("exp-123", api_key="test-key")

        with pytest.raises(ToraValidationError, match="positive integer"):
            tora.max_buffer_len = 0

        with pytest.raises(ToraValidationError, match="positive integer"):
            tora.max_buffer_len = "invalid"

    def test_experiment_id_property(self):
        """Test experiment_id property."""
        tora = Tora("exp-123", api_key="test-key")
        assert tora.experiment_id == "exp-123"

    def test_buffer_size_property(self):
        """Test buffer_size property."""
        tora = Tora("exp-123", api_key="test-key")
        assert tora.buffer_size == 0

        tora.log("metric", 1.0)
        assert tora.buffer_size == 1

    def test_is_closed_property(self):
        """Test is_closed property."""
        tora = Tora("exp-123", api_key="test-key")
        assert not tora.is_closed

        tora.shutdown()
        assert tora.is_closed
