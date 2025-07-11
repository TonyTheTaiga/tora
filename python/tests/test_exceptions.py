"""
Tests for the exceptions module.
"""

import pytest

from tora._exceptions import (
    HTTPStatusError,
    ToraAPIError,
    ToraAuthenticationError,
    ToraConfigurationError,
    ToraError,
    ToraExperimentError,
    ToraMetricError,
    ToraNetworkError,
    ToraTimeoutError,
    ToraValidationError,
    ToraWorkspaceError,
)


class TestToraError:
    """Tests for the base ToraError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ToraError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error with details."""
        details = {"code": 123, "field": "name"}
        error = ToraError("Test error", details)
        assert error.message == "Test error"
        assert error.details == details
        assert "Details: {'code': 123, 'field': 'name'}" in str(error)

    def test_inheritance(self):
        """Test that ToraError inherits from Exception."""
        error = ToraError("Test")
        assert isinstance(error, Exception)


class TestSpecificErrors:
    """Tests for specific error types."""

    def test_configuration_error(self):
        """Test ToraConfigurationError."""
        error = ToraConfigurationError("Config error")
        assert isinstance(error, ToraError)
        assert str(error) == "Config error"

    def test_authentication_error(self):
        """Test ToraAuthenticationError."""
        error = ToraAuthenticationError("Auth error")
        assert isinstance(error, ToraError)
        assert str(error) == "Auth error"

    def test_validation_error(self):
        """Test ToraValidationError."""
        error = ToraValidationError("Validation error")
        assert isinstance(error, ToraError)
        assert str(error) == "Validation error"

    def test_experiment_error(self):
        """Test ToraExperimentError."""
        error = ToraExperimentError("Experiment error")
        assert isinstance(error, ToraError)
        assert str(error) == "Experiment error"

    def test_metric_error(self):
        """Test ToraMetricError."""
        error = ToraMetricError("Metric error")
        assert isinstance(error, ToraError)
        assert str(error) == "Metric error"

    def test_workspace_error(self):
        """Test ToraWorkspaceError."""
        error = ToraWorkspaceError("Workspace error")
        assert isinstance(error, ToraError)
        assert str(error) == "Workspace error"


class TestToraNetworkError:
    """Tests for ToraNetworkError."""

    def test_basic_network_error(self):
        """Test basic network error."""
        error = ToraNetworkError("Network error")
        assert isinstance(error, ToraError)
        assert str(error) == "Network error"
        assert error.status_code is None
        assert error.response_text is None

    def test_network_error_with_status(self):
        """Test network error with status code."""
        error = ToraNetworkError("Network error", status_code=500)
        assert error.status_code == 500
        assert "Status: 500" in str(error)

    def test_network_error_with_response(self):
        """Test network error with response text."""
        error = ToraNetworkError("Network error", response_text="Server error")
        assert error.response_text == "Server error"
        assert "Response: Server error" in str(error)

    def test_network_error_full(self):
        """Test network error with all parameters."""
        details = {"url": "https://api.example.com"}
        error = ToraNetworkError(
            "Network error", status_code=404, response_text="Not found", details=details
        )

        error_str = str(error)
        assert "Network error" in error_str
        assert "Status: 404" in error_str
        assert "Response: Not found" in error_str
        assert "Details: {'url': 'https://api.example.com'}" in error_str


class TestToraAPIError:
    """Tests for ToraAPIError."""

    def test_api_error(self):
        """Test API error."""
        error = ToraAPIError("API error", status_code=400)
        assert isinstance(error, ToraNetworkError)
        assert isinstance(error, ToraError)
        assert error.status_code == 400


class TestToraTimeoutError:
    """Tests for ToraTimeoutError."""

    def test_timeout_error(self):
        """Test timeout error."""
        error = ToraTimeoutError("Timeout error")
        assert isinstance(error, ToraNetworkError)
        assert isinstance(error, ToraError)


class TestHTTPStatusError:
    """Tests for legacy HTTPStatusError."""

    def test_legacy_http_error(self):
        """Test legacy HTTP error for backward compatibility."""

        # Mock response object
        class MockResponse:
            status_code = 404
            text = "Not found"

        response = MockResponse()
        error = HTTPStatusError("HTTP error", response)

        assert isinstance(error, ToraNetworkError)
        assert error.status_code == 404
        assert error.response_text == "Not found"
        assert error.response is response

    def test_legacy_http_error_no_attributes(self):
        """Test legacy HTTP error with response without attributes."""
        response = "simple response"
        error = HTTPStatusError("HTTP error", response)

        assert isinstance(error, ToraNetworkError)
        assert error.status_code is None
        assert error.response_text is None
        assert error.response == response
