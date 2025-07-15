import pytest

from tora._exceptions import ToraValidationError
from tora._validation import (
    validate_experiment_name,
    validate_hyperparams,
    validate_metric_name,
    validate_metric_value,
    validate_step,
    validate_tags,
    validate_workspace_id,
)


class TestValidateExperimentName:
    def test_valid_name(self):
        assert validate_experiment_name("test-experiment") == "test-experiment"
        assert validate_experiment_name("  test  ") == "test"
        assert validate_experiment_name("My Experiment 123") == "My Experiment 123"

    def test_invalid_name_type(self):
        with pytest.raises(ToraValidationError, match="must be a string"):
            validate_experiment_name(123)

        with pytest.raises(ToraValidationError, match="must be a string"):
            validate_experiment_name(None)

    def test_empty_name(self):
        with pytest.raises(ToraValidationError, match="cannot be empty"):
            validate_experiment_name("")

        with pytest.raises(ToraValidationError, match="cannot be empty"):
            validate_experiment_name("   ")

    def test_name_too_long(self):
        long_name = "a" * 256
        with pytest.raises(ToraValidationError, match="cannot exceed 255 characters"):
            validate_experiment_name(long_name)

    def test_invalid_characters(self):
        invalid_names = [
            "test<name",
            "test>name",
            "test:name",
            'test"name',
            "test/name",
        ]
        for name in invalid_names:
            with pytest.raises(ToraValidationError, match="invalid characters"):
                validate_experiment_name(name)


class TestValidateWorkspaceId:
    def test_valid_workspace_id(self):
        assert validate_workspace_id("12345678-1234-1234-1234-123456789012") == "12345678-1234-1234-1234-123456789012"
        assert validate_workspace_id("12345678123412341234123456789012") == "12345678123412341234123456789012"
        assert validate_workspace_id(None) is None

    def test_invalid_workspace_id_type(self):
        with pytest.raises(ToraValidationError, match="must be a string"):
            validate_workspace_id(123)

    def test_empty_workspace_id(self):
        with pytest.raises(ToraValidationError, match="cannot be empty"):
            validate_workspace_id("")

        with pytest.raises(ToraValidationError, match="cannot be empty"):
            validate_workspace_id("   ")

    def test_invalid_workspace_id_format(self):
        with pytest.raises(ToraValidationError, match="must contain only letters, numbers, and hyphens"):
            validate_workspace_id("invalid@workspace")


class TestValidateHyperparams:
    def test_valid_hyperparams(self):
        params = {"lr": 0.01, "batch_size": 32, "model": "resnet"}
        result = validate_hyperparams(params)
        assert result == params

        assert validate_hyperparams(None) is None
        assert validate_hyperparams({}) == {}

    def test_invalid_hyperparams_type(self):
        with pytest.raises(ToraValidationError, match="must be a mapping"):
            validate_hyperparams("invalid")

    def test_invalid_key_type(self):
        with pytest.raises(ToraValidationError, match="key must be string"):
            validate_hyperparams({123: "value"})

    def test_empty_key(self):
        with pytest.raises(ToraValidationError, match="key cannot be empty"):
            validate_hyperparams({"": "value"})

    def test_key_too_long(self):
        long_key = "a" * 101
        with pytest.raises(ToraValidationError, match="exceeds 100 characters"):
            validate_hyperparams({long_key: "value"})

    def test_invalid_value_type(self):
        with pytest.raises(ToraValidationError, match="invalid type"):
            validate_hyperparams({"key": []})

    def test_nan_float_value(self):
        with pytest.raises(ToraValidationError, match="cannot be NaN"):
            validate_hyperparams({"key": float("nan")})

    def test_infinite_float_value(self):
        with pytest.raises(ToraValidationError, match="cannot be infinite"):
            validate_hyperparams({"key": float("inf")})

    def test_string_too_long(self):
        long_string = "a" * 1001
        with pytest.raises(ToraValidationError, match="exceeds 1000 characters"):
            validate_hyperparams({"key": long_string})


class TestValidateTags:
    def test_valid_tags(self):
        tags = ["ml", "test", "experiment"]
        assert validate_tags(tags) == tags
        assert validate_tags(None) is None
        assert validate_tags([]) == []

    def test_invalid_tags_type(self):
        with pytest.raises(ToraValidationError, match="must be a list"):
            validate_tags("invalid")

    def test_too_many_tags(self):
        tags = [f"tag{i}" for i in range(51)]
        with pytest.raises(ToraValidationError, match="more than 50 tags"):
            validate_tags(tags)

    def test_invalid_tag_type(self):
        with pytest.raises(ToraValidationError, match="must be a string"):
            validate_tags([123])

    def test_empty_tag(self):
        with pytest.raises(ToraValidationError, match="cannot be empty"):
            validate_tags([""])

    def test_tag_too_long(self):
        long_tag = "a" * 51
        with pytest.raises(ToraValidationError, match="exceeds 50 characters"):
            validate_tags([long_tag])

    def test_invalid_tag_characters(self):
        invalid_tags = ["tag<name", "tag>name", "tag:name", 'tag"name', "tag,name"]
        for tag in invalid_tags:
            with pytest.raises(ToraValidationError, match="invalid characters"):
                validate_tags([tag])

    def test_duplicate_tags(self):
        tags = ["ml", "ML", "test", "Test"]
        result = validate_tags(tags)
        assert len(result) == 2


class TestValidateMetricName:
    def test_valid_metric_name(self):
        assert validate_metric_name("accuracy") == "accuracy"
        assert validate_metric_name("train/loss") == "train/loss"
        assert validate_metric_name("val_accuracy") == "val_accuracy"
        assert validate_metric_name("metric-1") == "metric-1"

    def test_invalid_metric_name_type(self):
        with pytest.raises(ToraValidationError, match="must be a string"):
            validate_metric_name(123)

    def test_empty_metric_name(self):
        with pytest.raises(ToraValidationError, match="cannot be empty"):
            validate_metric_name("")

    def test_metric_name_too_long(self):
        long_name = "a" * 101
        with pytest.raises(ToraValidationError, match="cannot exceed 100 characters"):
            validate_metric_name(long_name)

    def test_invalid_metric_name_characters(self):
        with pytest.raises(ToraValidationError, match="can only contain"):
            validate_metric_name("metric name")


class TestValidateMetricValue:
    def test_valid_metric_values(self):
        assert validate_metric_value(1) == 1
        assert validate_metric_value(1.5) == 1.5
        assert validate_metric_value(True) == 1
        assert validate_metric_value(False) == 0

    def test_string_to_float_conversion(self):
        assert validate_metric_value("1.5") == 1.5
        assert validate_metric_value("42") == 42.0

    def test_invalid_metric_value_type(self):
        with pytest.raises(ToraValidationError, match="must be numeric"):
            validate_metric_value("invalid")

        with pytest.raises(ToraValidationError, match="must be numeric"):
            validate_metric_value([])

    def test_nan_metric_value(self):
        with pytest.raises(ToraValidationError, match="cannot be NaN"):
            validate_metric_value(float("nan"))

    def test_infinite_metric_value(self):
        with pytest.raises(ToraValidationError, match="cannot be infinite"):
            validate_metric_value(float("inf"))


class TestValidateStep:
    def test_valid_steps(self):
        assert validate_step(1) == 1
        assert validate_step(0) == 0
        assert validate_step(None) is None

    def test_string_to_int_conversion(self):
        assert validate_step("42") == 42

    def test_invalid_step_type(self):
        with pytest.raises(ToraValidationError, match="must be an integer"):
            validate_step("invalid")

    def test_negative_step(self):
        with pytest.raises(ToraValidationError, match="must be non-negative"):
            validate_step(-1)

    def test_step_too_large(self):
        with pytest.raises(ToraValidationError, match="too large"):
            validate_step(2**63)
