from unittest.mock import Mock, patch

import pytest

import tora._wrapper
from tora import flush, get_experiment_id, is_initialized, setup, shutdown, tmetric
from tora._wrapper import tresult
from tora._exceptions import ToraError, ToraValidationError


class TestSetup:
    @patch("tora._wrapper.Tora")
    def test_setup_success(self, mock_tora_class):
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_tora_class.create_experiment.return_value = mock_client

        experiment_id = setup(
            "test-experiment",
            workspace_id="ws-123",
            description="Test description",
            hyperparams={"lr": 0.01},
            tags=["test"],
            api_key="test-key",
        )

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
        tora._wrapper._INSTANCE = Mock()

        with pytest.raises(ToraError, match="already initialized"):
            setup("test-experiment", api_key="test-key")

    @patch("tora._wrapper.Tora")
    def test_setup_failure_resets_client(self, mock_tora_class):
        mock_tora_class.create_experiment.side_effect = ToraValidationError("Invalid name")

        with pytest.raises(ToraValidationError):
            setup("", api_key="test-key")

        assert tora._wrapper._INSTANCE is None

    @patch("tora._wrapper.Tora")
    @patch("tora._wrapper.atexit.register")
    def test_setup_registers_cleanup(self, mock_atexit, mock_tora_class):
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_tora_class.create_experiment.return_value = mock_client

        setup("test-experiment", api_key="test-key")

        mock_atexit.assert_called_once_with(shutdown)


class TestTmetric:
    def test_tmetric_success(self):
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        tmetric("accuracy", 0.95, step=100)

        mock_client.metric.assert_called_once_with("accuracy", 0.95, 100)

    def test_tmetric_no_client(self):
        tora._wrapper._INSTANCE = None

        with pytest.raises(ToraError, match="not initialized"):
            tmetric("accuracy", 0.95, step=1)


class TestFlush:
    def test_flush_success(self):
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        flush()

        mock_client.flush.assert_called_once()

    def test_flush_no_client(self):
        tora._wrapper._INSTANCE = None

        flush()


class TestShutdown:
    def test_shutdown_success(self):
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        shutdown()

        mock_client.shutdown.assert_called_once()
        assert tora._wrapper._INSTANCE is None

    def test_shutdown_no_client(self):
        tora._wrapper._INSTANCE = None

        shutdown()

    def test_shutdown_with_error(self):
        mock_client = Mock()
        mock_client.shutdown.side_effect = Exception("Shutdown error")
        tora._wrapper._INSTANCE = mock_client

        shutdown()
        assert tora._wrapper._INSTANCE is None


class TestIsInitialized:
    def test_is_initialized_true(self):
        mock_client = Mock()
        mock_client.is_closed = False
        tora._wrapper._INSTANCE = mock_client

        assert is_initialized() is True

    def test_is_initialized_false_no_client(self):
        tora._wrapper._INSTANCE = None

        assert is_initialized() is False

    def test_is_initialized_false_closed_client(self):
        mock_client = Mock()
        mock_client.is_closed = True
        tora._wrapper._INSTANCE = mock_client

        assert is_initialized() is False


class TestTresult:
    def test_tresult_success(self):
        mock_client = Mock()
        tora._wrapper._INSTANCE = mock_client

        tresult("best_acc", 0.99)

        mock_client.result.assert_called_once_with("best_acc", 0.99)

    def test_tresult_no_client(self):
        tora._wrapper._INSTANCE = None

        with pytest.raises(ToraError, match="not initialized"):
            tresult("score", 1.0)


class TestGetExperimentId:
    def test_get_experiment_id_success(self):
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_client.is_closed = False
        tora._wrapper._INSTANCE = mock_client

        assert get_experiment_id() == "exp-123"

    def test_get_experiment_id_no_client(self):
        tora._wrapper._INSTANCE = None

        assert get_experiment_id() is None

    def test_get_experiment_id_closed_client(self):
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_client.is_closed = True
        tora._wrapper._INSTANCE = mock_client

        assert get_experiment_id() is None


class TestIntegration:
    @patch("tora._wrapper.Tora")
    def test_full_workflow(self, mock_tora_class):
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_client.is_closed = False
        mock_tora_class.create_experiment.return_value = mock_client

        assert not is_initialized()

        experiment_id = setup("test-experiment", api_key="test-key")
        assert experiment_id == "exp-123"
        assert is_initialized()

        tmetric("accuracy", 0.95, step=1)
        tmetric("loss", 0.05, step=1)

        flush()

        shutdown()
        assert not is_initialized()

        mock_client.metric.assert_any_call("accuracy", 0.95, 1)
        mock_client.metric.assert_any_call("loss", 0.05, 1)
        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()

    @patch("tora._wrapper.Tora")
    def test_setup_twice_error(self, mock_tora_class):
        mock_client = Mock()
        mock_client.experiment_id = "exp-123"
        mock_tora_class.create_experiment.return_value = mock_client

        setup("test-experiment-1", api_key="test-key")

        with pytest.raises(ToraError, match="already initialized"):
            setup("test-experiment-2", api_key="test-key")

        shutdown()

        setup("test-experiment-2", api_key="test-key")

    def test_tmetric_before_setup(self):
        assert not is_initialized()

        with pytest.raises(ToraError, match="not initialized"):
            tmetric("accuracy", 0.95, step=1)
