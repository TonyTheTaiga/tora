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
    def test_basic_error(self):
        error = ToraError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_with_details(self):
        details = {"code": 123, "field": "name"}
        error = ToraError("Test error", details)
        assert error.message == "Test error"
        assert error.details == details
        assert "Details: {'code': 123, 'field': 'name'}" in str(error)

    def test_inheritance(self):
        error = ToraError("Test")
        assert isinstance(error, Exception)


class TestSpecificErrors:
    def test_configuration_error(self):
        error = ToraConfigurationError("Config error")
        assert isinstance(error, ToraError)
        assert str(error) == "Config error"

    def test_authentication_error(self):
        error = ToraAuthenticationError("Auth error")
        assert isinstance(error, ToraError)
        assert str(error) == "Auth error"

    def test_validation_error(self):
        error = ToraValidationError("Validation error")
        assert isinstance(error, ToraError)
        assert str(error) == "Validation error"

    def test_experiment_error(self):
        error = ToraExperimentError("Experiment error")
        assert isinstance(error, ToraError)
        assert str(error) == "Experiment error"

    def test_metric_error(self):
        error = ToraMetricError("Metric error")
        assert isinstance(error, ToraError)
        assert str(error) == "Metric error"

    def test_workspace_error(self):
        error = ToraWorkspaceError("Workspace error")
        assert isinstance(error, ToraError)
        assert str(error) == "Workspace error"


class TestToraNetworkError:
    def test_basic_network_error(self):
        error = ToraNetworkError("Network error")
        assert isinstance(error, ToraError)
        assert str(error) == "Network error"
        assert error.status_code is None
        assert error.response_text is None

    def test_network_error_with_status(self):
        error = ToraNetworkError("Network error", status_code=500)
        assert error.status_code == 500
        assert "Status: 500" in str(error)

    def test_network_error_with_response(self):
        error = ToraNetworkError("Network error", response_text="Server error")
        assert error.response_text == "Server error"
        assert "Response: Server error" in str(error)

    def test_network_error_full(self):
        details = {"url": "https://api.example.com"}
        error = ToraNetworkError("Network error", status_code=404, response_text="Not found", details=details)

        error_str = str(error)
        assert "Network error" in error_str
        assert "Status: 404" in error_str
        assert "Response: Not found" in error_str
        assert "Details: {'url': 'https://api.example.com'}" in error_str


class TestToraAPIError:
    def test_api_error(self):
        error = ToraAPIError("API error", status_code=400)
        assert isinstance(error, ToraNetworkError)
        assert isinstance(error, ToraError)
        assert error.status_code == 400


class TestToraTimeoutError:
    def test_timeout_error(self):
        error = ToraTimeoutError("Timeout error")
        assert isinstance(error, ToraNetworkError)
        assert isinstance(error, ToraError)


class TestHTTPStatusError:
    def test_legacy_http_error(self):
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
        response = "simple response"
        error = HTTPStatusError("HTTP error", response)

        assert isinstance(error, ToraNetworkError)
        assert error.status_code is None
        assert error.response_text is None
        assert error.response == response
