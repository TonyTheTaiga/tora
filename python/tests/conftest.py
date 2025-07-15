import contextlib
from unittest.mock import Mock

import pytest

from tora._http import HttpClient, HttpResponse


@pytest.fixture
def mock_http_client():
    client = Mock(spec=HttpClient)
    return client


@pytest.fixture
def mock_response():
    response = Mock(spec=HttpResponse)
    response.status_code = 200
    response.text = '{"status": 200, "data": {}}'
    response.json.return_value = {"status": 200, "data": {}}
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def sample_experiment_data():
    return {
        "id": "exp-123",
        "name": "test-experiment",
        "description": "Test experiment",
        "hyperparams": [
            {"key": "learning_rate", "value": 0.01},
            {"key": "batch_size", "value": 32},
        ],
        "tags": ["test", "ml"],
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "available_metrics": ["accuracy", "loss"],
        "workspace_id": "ws-123",
        "url": "https://test-frontend.example.com/experiments/exp-123",
    }


@pytest.fixture
def sample_workspace_data():
    return {
        "id": "ws-123",
        "name": "test-workspace",
        "description": "Test workspace",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_hyperparams():
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 10,
        "model_name": "resnet50",
    }


@pytest.fixture
def sample_tags():
    return ["test", "ml", "experiment"]


@pytest.fixture
def mock_successful_response(sample_experiment_data):
    response = Mock(spec=HttpResponse)
    response.status_code = 200
    response.json.return_value = {"status": 200, "data": sample_experiment_data}
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_error_response():
    response = Mock(spec=HttpResponse)
    response.status_code = 400
    response.text = "Bad Request"
    response.json.return_value = {"status": 400, "error": "Bad Request"}

    def raise_error():
        from tora._exceptions import HTTPStatusError

        raise HTTPStatusError("HTTP 400 Bad Request", response)

    response.raise_for_status.side_effect = raise_error
    return response


@pytest.fixture(autouse=True)
def reset_global_client():
    import tora._wrapper

    tora._wrapper._INSTANCE = None
    yield
    if tora._wrapper._INSTANCE:
        with contextlib.suppress(Exception):
            tora._wrapper._INSTANCE.shutdown()
        tora._wrapper._INSTANCE = None


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.delenv("TORA_API_KEY", raising=False)
    monkeypatch.delenv("TORA_BASE_URL", raising=False)

    monkeypatch.setenv("TORA_API_KEY", "test-api-key-isolated")
    monkeypatch.setenv("TORA_BASE_URL", "https://test-isolated.example.com/api")


@pytest.fixture
def env_vars(monkeypatch):
    monkeypatch.setenv("TORA_API_KEY", "test-api-key")
    monkeypatch.setenv("TORA_BASE_URL", "https://test.tora.dev/api")
    return {
        "TORA_API_KEY": "test-api-key",
        "TORA_BASE_URL": "https://test.tora.dev/api",
    }
