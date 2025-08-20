#!/usr/bin/env python3
"""Basic usage example for the Tora Python SDK.

This example demonstrates the core functionality of the Tora SDK:
- Creating experiments
- Logging metrics
- Using context managers
- Error handling
"""

import random
import time

import tora
from tora import ToraError


def simulate_training():
    """Simulate a machine learning training process."""
    # Simulate training metrics
    for epoch in range(10):
        # Simulate some training
        time.sleep(0.1)

        # Generate fake metrics that improve over time
        train_loss = 1.0 - (epoch * 0.08) + random.uniform(-0.05, 0.05)
        train_acc = 0.5 + (epoch * 0.04) + random.uniform(-0.02, 0.02)
        val_loss = 1.1 - (epoch * 0.07) + random.uniform(-0.08, 0.08)
        val_acc = 0.45 + (epoch * 0.045) + random.uniform(-0.03, 0.03)

        yield (
            epoch,
            {
                "train_loss": max(0.01, train_loss),
                "train_accuracy": min(0.99, max(0.0, train_acc)),
                "val_loss": max(0.01, val_loss),
                "val_accuracy": min(0.99, max(0.0, val_acc)),
            },
        )


def example_basic_usage():
    """Example of basic Tora usage."""
    print("=== Basic Usage Example ===")

    try:
        # Create an experiment
        client = tora.Tora.create_experiment(
            name="basic-example",
            description="Basic usage example for Tora SDK",
            hyperparams={
                "learning_rate": 0.001,
                "batch_size": 32,
                "optimizer": "adam",
                "model": "simple_cnn",
            },
            tags=["example", "tutorial", "basic"],
        )

        print(f"Created experiment: {client.experiment_id}")

        # Log metrics during "training"
        print("Starting training simulation...")
        for epoch, metrics in simulate_training():
            for metric_name, value in metrics.items():
                client._log(metric_name, value, step=epoch)

            print(f"Epoch {epoch}: {metrics}")

        # Log some final metrics with metadata
        client._log(
            "final_score",
            0.95,
            metadata={
                "model_size": "2.3MB",
                "inference_time": "12ms",
                "dataset": "custom_dataset_v1",
            },
        )

        print("Training completed!")

        # Ensure all metrics are sent
        client.shutdown()
        print("Experiment finished and data sent to Tora")

    except ToraError as e:
        print(f"Tora error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_context_manager():
    """Example using context manager for automatic cleanup."""
    print("\n=== Context Manager Example ===")

    try:
        # Using context manager ensures automatic cleanup
        with tora.Tora.create_experiment(
            name="context-manager-example",
            description="Example using context manager",
            hyperparams={"epochs": 5, "lr": 0.01},
            tags=["example", "context-manager"],
        ) as client:
            print(f"Created experiment: {client.experiment_id}")

            # Log some metrics
            for i in range(5):
                accuracy = 0.6 + (i * 0.08) + random.uniform(-0.02, 0.02)
                loss = 0.8 - (i * 0.12) + random.uniform(-0.05, 0.05)

                client._log("accuracy", accuracy, step=i)
                client._log("loss", max(0.01, loss), step=i)

                print(f"Step {i}: accuracy={accuracy:.3f}, loss={loss:.3f}")
                time.sleep(0.1)

            print("Training completed!")
            # No need to call shutdown() - context manager handles it

        print("Context manager automatically cleaned up")

    except ToraError as e:
        print(f"Tora error: {e}")


def example_global_api():
    """Example using the global API for simpler usage."""
    print("\n=== Global API Example ===")

    try:
        # Set up global experiment
        experiment_id = tora.setup(
            name="global-api-example",
            description="Example using global API",
            hyperparams={
                "model": "resnet18",
                "dataset": "cifar10",
                "batch_size": 64,
            },
            tags=["example", "global-api", "resnet"],
        )

        print(f"Set up global experiment: {experiment_id}")

        # Now you can log metrics from anywhere in your code
        for epoch in range(3):
            # Simulate training
            train_loss = 1.5 - (epoch * 0.3) + random.uniform(-0.1, 0.1)
            val_acc = 0.3 + (epoch * 0.2) + random.uniform(-0.05, 0.05)

            tora.tlog("train_loss", max(0.01, train_loss), step=epoch)
            tora.tlog("val_accuracy", min(0.99, val_acc), step=epoch)

            print(f"Epoch {epoch}: train_loss={train_loss:.3f}, val_acc={val_acc:.3f}")
            time.sleep(0.1)

        # Log final results
        tora.tlog("final_accuracy", 0.87)
        tora.tlog("training_time", 125.5, metadata={"units": "seconds"})

        print("Training completed!")

        # Clean up (optional - happens automatically at program exit)
        tora.shutdown()
        print("Global experiment cleaned up")

    except ToraError as e:
        print(f"Tora error: {e}")


def example_error_handling():
    """Example demonstrating error handling."""
    print("\n=== Error Handling Example ===")

    # Example 1: Invalid experiment name
    try:
        tora.Tora.create_experiment("")  # Empty name should fail
    except tora.ToraValidationError as e:
        print(f"Validation error (expected): {e}")

    # Example 2: Invalid metric value
    try:
        client = tora.Tora.create_experiment("error-example")
        client._log("invalid_metric", float("nan"))  # NaN should fail
    except tora.ToraValidationError as e:
        print(f"Validation error (expected): {e}")
        client.shutdown()

    # Example 3: Network error simulation
    try:
        # This will fail if no API key is set
        client = tora.Tora.create_experiment("network-test", api_key="invalid-key")
    except tora.ToraAuthenticationError as e:
        print(f"Authentication error (expected): {e}")
    except tora.ToraNetworkError as e:
        print(f"Network error: {e}")
    except ToraError as e:
        print(f"General Tora error: {e}")


def main():
    """Run all examples."""
    print("Tora Python SDK Examples")
    print("=" * 50)

    # Check if we're configured
    if not tora._config.TORA_API_KEY:
        print("⚠️  Warning: TORA_API_KEY not set. Some examples may fail.")
        print("   Set your API key: export TORA_API_KEY='your-key-here'")
        print()

    # Run examples
    example_basic_usage()
    example_context_manager()
    example_global_api()
    example_error_handling()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check your Tora dashboard to see the logged experiments.")


if __name__ == "__main__":
    main()
