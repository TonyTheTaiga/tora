"""
Tests for the wrapper module (global client functions).
"""

from unittest.mock import Mock, patch

import pytest

import tora._wrapper
from tora import flush, get_experiment_id, get_experiment_url, is_initialized, setup, shutdown, tlog
from tora._exceptions import ToraError, ToraValidationError


class TestSetup:
    """Tests for the setup function."""

    @patch("tora._wrapper.Tora")
    def test_setup_success(self, mock_tora_class):
        """Test successful setup."""
        # Setup mock
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_tora_class.create_experiment.return_value = mock_client

        # Test
        experiment_id = setup(
            "test-experiment",
            workspace_id="ws-123",
            description="Test description",
            hyperparams={"lr": 0.01},
            tags=["test"],
            api_key="test-key",
        )

        # Assertions
        assert experiment_id == "exp-123"
        assert tora._wrapper._INSTANCE is mock_client
        mock_tora_class.create_experiment.assert_called_once_with(
            name="test-experiment",
            workspace_id="ws-123",
            description="Test description",
            hyperparams={"lr": 0.01},
            tags=["test"],
            api_key="test-key",
            server_url=None,
            max_buffer_len=1,
        )

    def test_setup_already_initialized(self):
        """Test setup when client is already initialized."""
        # Setup existing client
        tora._wrapper._INSTANCE = Mock()

        with pytest.raises(ToraError, match="already initialized"):
            setup("test-experiment", api_key="test-key")

    @patch("tora._wrapper.Tora")
    def test_setup_failure_resets_client(self, mock_tora_class):
        """Test that client is reset on setup failure."""
        # Setup mock to raise error
        mock_tora_class.create_experiment.side_effect = ToraValidationError("Invalid name")

        with pytest.raises(ToraValidationError):
            setup("", api_key="test-key")

        # Client should be reset to None
        assert tora._wrapper._INSTANCE is None

    @patch("tora._wrapper.Tora")
    @patch("tora._wrapper.atexit.register")
    def test_setup_registers_cleanup(self, mock_atexit, mock_tora_class):
        """Test that setup registers cleanup function."""
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_tora_class.create_experiment.return_value = mock_client

        setup("test-experiment", api_key="test-key")

        mock_atexit.assert_called_once_with(shutdown)


class TestTlog:
    """Tests for the tlog function."""

    def test_tlog_success(self):
        """Test successful metric logging."""
        # Setup mock client
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        # Test
        tlog("accuracy", 0.95, step=100, metadata={"epoch": 1})

        # Assertions
        mock_client.log.assert_called_once_with("accuracy", 0.95, 100, {"epoch": 1})

    def test_tlog_no_client(self):
        """Test tlog without initialized client."""
        tora._wrapper._INSTANCE = None

        with pytest.raises(ToraError, match="not initialized"):
            tlog("accuracy", 0.95)

    def test_tlog_minimal_args(self):
        """Test tlog with minimal arguments."""
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        tlog("loss", 0.5)

        mock_client.log.assert_called_once_with("loss", 0.5, None, None)


class TestFlush:
    """Tests for the flush function."""

    def test_flush_success(self):
        """Test successful flush."""
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        flush()

        mock_client.flush.assert_called_once()

    def test_flush_no_client(self):
        """Test flush without client (should not raise error)."""
        tora._wrapper._INSTANCE = None

        # Should not raise error
        flush()


class TestShutdown:
    """Tests for the shutdown function."""

    def test_shutdown_success(self):
        """Test successful shutdown."""
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        shutdown()

        mock_client.shutdown.assert_called_once()
        assert tora._wrapper._INSTANCE is None

    def test_shutdown_no_client(self):
        """Test shutdown without client."""
        tora._wrapper._INSTANCE = None

        # Should not raise error
        shutdown()

    def test_shutdown_with_error(self):
        """Test shutdown when client raises error."""
        mock_client = Mock()
        mock_client.shutdown.side_effect = Exception("Shutdown error")
        tora._wrapper._INSTANCE = mock_client

        # Should not raise error, but should reset client
        shutdown()
        assert tora._wrapper._INSTANCE is None


class TestIsInitialized:
    """Tests for the is_initialized function."""

    def test_is_initialized_true(self):
        """Test is_initialized when client exists and is not closed."""
        mock_client = Mock()
        mock_client.is_closed = False
        tora._wrapper._INSTANCE = mock_client

        assert is_initialized() is True

    def test_is_initialized_false_no_client(self):
        """Test is_initialized when no client exists."""
        tora._wrapper._INSTANCE = None

        assert is_initialized() is False

    def test_is_initialized_false_closed_client(self):
        """Test is_initialized when client is closed."""
        mock_client = Mock()
        mock_client.is_closed = True
        tora._wrapper._INSTANCE = mock_client

        assert is_initialized() is False


class TestGetExperimentId:
    """Tests for the get_experiment_id function."""

    def test_get_experiment_id_success(self):
        """Test getting experiment ID when client is initialized."""
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_client.is_closed = False
        tora._wrapper._INSTANCE = mock_client

        assert get_experiment_id() == "exp-123"

    def test_get_experiment_id_no_client(self):
        """Test getting experiment ID when no client exists."""
        tora._wrapper._INSTANCE = None

        assert get_experiment_id() is None

    def test_get_experiment_id_closed_client(self):
        """Test getting experiment ID when client is closed."""
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_client.is_closed = True
        tora._wrapper._INSTANCE = mock_client

        assert get_experiment_id() is None


class TestGetExperimentUrl:
    """Tests for the get_experiment_url function."""

    def test_get_experiment_url_success(self):
        """Test getting experiment URL when client is initialized."""
        mock_client = Mock()
        mock_client.url = "https://test-frontend.example.com/experiments/exp-123"
        mock_client.is_closed = False
        tora._wrapper._INSTANCE = mock_client

        assert get_experiment_url() == "https://test-frontend.example.com/experiments/exp-123"

    def test_get_experiment_url_no_client(self):
        """Test getting experiment URL when no client exists."""
        tora._wrapper._INSTANCE = None

        assert get_experiment_url() is None

    def test_get_experiment_url_closed_client(self):
        """Test getting experiment URL when client is closed."""
        mock_client = Mock()
        mock_client.url = "https://test-frontend.example.com/experiments/exp-123"
        mock_client.is_closed = True
        tora._wrapper._INSTANCE = mock_client

        assert get_experiment_url() is None


class TestIntegration:
    """Integration tests for wrapper functions."""

    @patch("tora._wrapper.Tora")
    def test_full_workflow(self, mock_tora_class):
        """Test complete workflow with wrapper functions."""
        # Setup mock
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_client.is_closed = False
        mock_tora_class.create_experiment.return_value = mock_client

        # Test workflow
        assert not is_initialized()

        experiment_id = setup("test-experiment", api_key="test-key")
        assert experiment_id == "exp-123"
        assert is_initialized()

        tlog("accuracy", 0.95, step=1)
        tlog("loss", 0.05, step=1)

        flush()

        shutdown()
        assert not is_initialized()

        # Verify calls
        mock_client.log.assert_any_call("accuracy", 0.95, 1, None)
        mock_client.log.assert_any_call("loss", 0.05, 1, None)
        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()

    @patch("tora._wrapper.Tora")
    def test_setup_twice_error(self, mock_tora_class):
        """Test that setup twice raises error."""
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_tora_class.create_experiment.return_value = mock_client

        # First setup should succeed
        setup("test-experiment-1", api_key="test-key")

        # Second setup should fail
        with pytest.raises(ToraError, match="already initialized"):
            setup("test-experiment-2", api_key="test-key")

        # Cleanup
        shutdown()

        # Now second setup should succeed
        setup("test-experiment-2", api_key="test-key")

    def test_tlog_before_setup(self):
        """Test tlog before setup raises error."""
        assert not is_initialized()

        with pytest.raises(ToraError, match="not initialized"):
            tlog("accuracy", 0.95)
