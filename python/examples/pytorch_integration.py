#!/usr/bin/env python3
"""PyTorch integration example for the Tora Python SDK.

This example demonstrates how to integrate Tora with PyTorch training:
- Logging training and validation metrics
- Tracking hyperparameters
- Model checkpointing integration
- Custom callbacks
"""

import random
import time
from typing import Any

import tora


class MockModel:
    """Mock PyTorch model for demonstration."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.parameters_count = input_size * hidden_size + hidden_size * num_classes

    def train(self):
        """Set model to training mode."""

    def eval(self):
        """Set model to evaluation mode."""


class MockOptimizer:
    """Mock PyTorch optimizer for demonstration."""

    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def zero_grad(self):
        """Zero gradients."""

    def step(self):
        """Optimizer step."""


class MockDataLoader:
    """Mock PyTorch DataLoader for demonstration."""

    def __init__(self, dataset_size: int = 1000, batch_size: int = 32):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_batches = dataset_size // batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            # Simulate batch data
            yield i, {"data": f"batch_{i}", "target": f"target_{i}"}

    def __len__(self):
        return self.num_batches


def simulate_forward_pass() -> dict[str, float]:
    """Simulate a forward pass and return loss and accuracy."""
    # Simulate training that improves over time
    base_loss = random.uniform(0.1, 0.8)
    base_acc = random.uniform(0.7, 0.95)

    return {"loss": base_loss, "accuracy": base_acc}


def train_epoch(model, train_loader, optimizer, tora_client, epoch: int) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(train_loader)

    for batch_idx, _batch in enumerate(train_loader):
        # Simulate training step
        optimizer.zero_grad()

        # Simulate forward pass
        metrics = simulate_forward_pass()
        loss = metrics["loss"]
        accuracy = metrics["accuracy"]

        # Simulate backward pass
        # loss.backward()  # In real PyTorch
        optimizer.step()

        total_loss += loss
        total_accuracy += accuracy

        # Log batch-level metrics every 10 batches
        if batch_idx % 10 == 0:
            global_step = epoch * num_batches + batch_idx
            tora_client.log("batch_loss", loss, step=global_step)
            tora_client.log("batch_accuracy", accuracy, step=global_step)

            print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}: loss={loss:.4f}, acc={accuracy:.4f}")

        # Simulate training time
        time.sleep(0.01)

    # Return epoch averages
    return {
        "train_loss": total_loss / num_batches,
        "train_accuracy": total_accuracy / num_batches,
    }


def validate_epoch(model, val_loader, tora_client, epoch: int) -> dict[str, float]:
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(val_loader)

    # Simulate validation (no gradients)
    for _batch_idx, _batch in enumerate(val_loader):
        # Simulate forward pass
        metrics = simulate_forward_pass()
        # Validation typically has slightly different metrics
        loss = metrics["loss"] * 1.1  # Slightly higher loss
        accuracy = metrics["accuracy"] * 0.95  # Slightly lower accuracy

        total_loss += loss
        total_accuracy += accuracy

        time.sleep(0.005)  # Validation is typically faster

    return {
        "val_loss": total_loss / num_batches,
        "val_accuracy": total_accuracy / num_batches,
    }


def pytorch_training_example():
    """Complete PyTorch training example with Tora integration."""
    print("=== PyTorch Training Example ===")

    # Hyperparameters
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
        "hidden_size": 128,
        "optimizer": "adam",
        "model_architecture": "simple_mlp",
        "dataset": "mnist",
    }

    try:
        # Create Tora experiment
        with tora.Tora.create_experiment(
            name="pytorch-mnist-training",
            description="MNIST classification with PyTorch and Tora tracking",
            hyperparams=hyperparams,
            tags=["pytorch", "mnist", "classification", "mlp"],
        ) as tora_client:
            print(f"Created experiment: {tora_client.experiment_id}")

            # Initialize model, optimizer, and data loaders
            model = MockModel(input_size=784, hidden_size=hyperparams["hidden_size"], num_classes=10)
            optimizer = MockOptimizer(lr=hyperparams["learning_rate"])

            train_loader = MockDataLoader(dataset_size=50000, batch_size=hyperparams["batch_size"])
            val_loader = MockDataLoader(dataset_size=10000, batch_size=hyperparams["batch_size"])

            # Log model information
            tora_client._log(
                "model_parameters",
                model.parameters_count,
                metadata={
                    "input_size": model.input_size,
                    "hidden_size": model.hidden_size,
                    "num_classes": model.num_classes,
                },
            )

            print(f"Model has {model.parameters_count:,} parameters")
            print(f"Training batches: {len(train_loader)}")
            print(f"Validation batches: {len(val_loader)}")

            # Training loop
            best_val_acc = 0.0

            for epoch in range(hyperparams["epochs"]):
                print(f"\nEpoch {epoch + 1}/{hyperparams['epochs']}")
                print("-" * 40)

                # Training
                train_metrics = train_epoch(model, train_loader, optimizer, tora_client, epoch)

                # Validation
                val_metrics = validate_epoch(model, val_loader, tora_client, epoch)

                # Log epoch-level metrics
                all_metrics = {**train_metrics, **val_metrics}
                for metric_name, value in all_metrics.items():
                    tora_client._log(metric_name, value, step=epoch)

                # Log learning rate (in real PyTorch, you might have a scheduler)
                current_lr = optimizer.lr * (0.95**epoch)  # Simulate decay
                tora_client._log("learning_rate", current_lr, step=epoch)

                # Check for best model
                val_acc = val_metrics["val_accuracy"]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    tora_client._log("best_val_accuracy", best_val_acc, step=epoch)
                    print(f"New best validation accuracy: {best_val_acc:.4f}")

                    # In real PyTorch, you would save the model here
                    # torch.save(model.state_dict(), 'best_model.pth')

                print(
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']:.4f}",
                )
                print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}")

            # Log final results
            tora_client._log(
                "final_results",
                best_val_acc,
                metadata={
                    "total_epochs": hyperparams["epochs"],
                    "final_train_loss": train_metrics["train_loss"],
                    "final_val_loss": val_metrics["val_loss"],
                    "training_completed": True,
                },
            )

            print("\nTraining completed!")
            print(f"Best validation accuracy: {best_val_acc:.4f}")

    except tora.ToraError as e:
        print(f"Tora error: {e}")
    except Exception as e:
        print(f"Training error: {e}")


def pytorch_callback_example():
    """Example of creating a reusable PyTorch callback for Tora."""
    print("\n=== PyTorch Callback Example ===")

    class ToraCallback:
        """Reusable callback for PyTorch training with Tora logging."""

        def __init__(self, tora_client: tora.Tora, log_frequency: int = 10):
            self.tora = tora_client
            self.log_frequency = log_frequency
            self.step = 0

        def on_batch_end(self, metrics: dict[str, float]):
            """Called at the end of each batch."""
            if self.step % self.log_frequency == 0:
                for name, value in metrics.items():
                    self.tora._log(f"batch_{name}", value, step=self.step)
            self.step += 1

        def on_epoch_end(self, epoch: int, metrics: dict[str, float]):
            """Called at the end of each epoch."""
            for name, value in metrics.items():
                self.tora._log(name, value, step=epoch)

        def on_training_end(self, final_metrics: dict[str, Any]):
            """Called when training is complete."""
            for name, value in final_metrics.items():
                if isinstance(value, int | float):
                    self.tora._log(f"final_{name}", value)
                else:
                    # Log as metadata for non-numeric values
                    self.tora._log("training_summary", 1.0, metadata={name: value})

    # Example usage of the callback
    try:
        with tora.Tora.create_experiment(
            name="pytorch-callback-example",
            description="Example using PyTorch callback pattern",
            hyperparams={"lr": 0.01, "epochs": 3},
            tags=["pytorch", "callback", "example"],
        ) as tora_client:
            print(f"Created experiment: {tora_client.experiment_id}")

            # Initialize callback
            callback = ToraCallback(tora_client, log_frequency=5)

            # Simulate training with callback
            for epoch in range(3):
                print(f"Epoch {epoch}")

                # Simulate batches
                for _batch in range(20):
                    batch_metrics = {
                        "loss": random.uniform(0.1, 0.5),
                        "accuracy": random.uniform(0.8, 0.95),
                    }
                    callback.on_batch_end(batch_metrics)

                # Epoch metrics
                epoch_metrics = {
                    "train_loss": random.uniform(0.1, 0.3),
                    "val_loss": random.uniform(0.15, 0.35),
                    "train_accuracy": random.uniform(0.85, 0.95),
                    "val_accuracy": random.uniform(0.80, 0.90),
                }
                callback.on_epoch_end(epoch, epoch_metrics)

                print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
                print(f"  Val Accuracy: {epoch_metrics['val_accuracy']:.4f}")

            # Training complete
            final_metrics = {
                "best_accuracy": 0.92,
                "total_training_time": 150.5,
                "model_size_mb": 2.3,
                "convergence_epoch": 2,
            }
            callback.on_training_end(final_metrics)

            print("Training with callback completed!")

    except tora.ToraError as e:
        print(f"Tora error: {e}")


def main():
    """Run PyTorch integration examples."""
    print("Tora PyTorch Integration Examples")
    print("=" * 50)

    if not tora._config.TORA_API_KEY:
        print("⚠️  Warning: TORA_API_KEY not set. Examples may fail.")
        print("   Set your API key: export TORA_API_KEY='your-key-here'")
        print()

    pytorch_training_example()
    pytorch_callback_example()

    print("\n" + "=" * 50)
    print("PyTorch examples completed!")
    print("In real usage, replace mock classes with actual PyTorch components.")


if __name__ == "__main__":
    main()
