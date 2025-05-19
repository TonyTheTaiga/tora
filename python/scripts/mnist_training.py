import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tora import Tora
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def safe_value(value):
    """
    Convert various types to a safe numeric type, handling NaN, inf, and non-numeric values.
    """
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def log_metric(client, name, value, step):
    """
    Log a metric to Tora if the value is valid.
    """
    val = safe_value(value)
    if val is not None:
        client.log(name=name, value=val, step=step)


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.
    """

    def __init__(self, dropout_rate=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_epoch(model, device, loader, optimizer, criterion, epoch, tora):
    """
    Train the model for one epoch and log training metrics.
    Returns:
        epoch_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        try:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}"
                )
        except Exception as e:
            print(f"Error batch {batch_idx}: {e}")

    epoch_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    epoch_time = time.time() - start_time

    log_metric(tora, "train_loss", epoch_loss, epoch)
    log_metric(tora, "train_accuracy", accuracy, epoch)
    log_metric(tora, "epoch_time", epoch_time, epoch)

    return epoch_loss, accuracy


def validate(model, device, loader, criterion, epoch, tora, split="val"):
    """
    Evaluate the model on validation or test set and log metrics.
    Returns:
        loss, accuracy, precision, recall, f1
    """
    model.eval()
    total_loss = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)
            preds = output.argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    size = len(loader.dataset)
    avg_loss = total_loss / max(size, 1)

    try:
        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )
    except Exception:
        accuracy = precision = recall = f1 = 0.0

    prefix = split
    log_metric(tora, f"{prefix}_loss", avg_loss, epoch)
    log_metric(tora, f"{prefix}_accuracy", accuracy, epoch)
    log_metric(tora, f"{prefix}_precision", precision * 100, epoch)
    log_metric(tora, f"{prefix}_recall", recall * 100, epoch)
    log_metric(tora, f"{prefix}_f1", f1 * 100, epoch)

    print(
        f"{split.title()} set: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, F1={f1 * 100:.2f}%"
    )
    return avg_loss, accuracy, precision, recall, f1


def main():
    # Hyperparameters
    hyperparams = {
        "batch_size": 64,
        "epochs": 10,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "dropout_rate": 0.20,
        "scheduler": "cosine",
        "optimizer": "SGD",
        "nesterov": True,
        "dampening": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    }

    # Device setup
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    hyperparams["device"] = str(device)

    # Data loading and transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )
    test_set = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(test_set, batch_size=hyperparams["batch_size"])

    # Tora experiment setup
    model = SimpleCNN(dropout_rate=hyperparams["dropout_rate"]).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    hyperparams.update(
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "input_shape": "1x28x28",
            "num_classes": 10,
            "model_parameters": model_params,
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": len(test_set),
            "criterion": "CrossEntropyLoss",
        }
    )
    tora = Tora.create_experiment(
        name="MNIST_CNN",
        description="CNN for MNIST classification with Tora logging",
        hyperparams=hyperparams,
        tags=["mnist", "cnn", "image-classification"],
    )

    # Optimizer and scheduler
    if hyperparams["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams["lr"],
            momentum=hyperparams["momentum"],
            weight_decay=hyperparams["weight_decay"],
            nesterov=hyperparams["nesterov"],
            dampening=hyperparams["dampening"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            betas=(hyperparams["beta1"], hyperparams["beta2"]),
            eps=hyperparams["eps"],
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparams["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, hyperparams["epochs"] + 1):
        log_metric(tora, "learning_rate", optimizer.param_groups[0]["lr"], epoch)
        train_epoch(model, device, train_loader, optimizer, criterion, epoch, tora)
        validate(model, device, val_loader, criterion, epoch, tora, split="val")
        scheduler.step()

    # Final evaluation
    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(
        model, device, test_loader, criterion, hyperparams["epochs"], tora, split="test"
    )
    log_metric(tora, "final_test_accuracy", test_acc, hyperparams["epochs"])
    log_metric(tora, "final_test_precision", test_prec * 100, hyperparams["epochs"])
    log_metric(tora, "final_test_recall", test_rec * 100, hyperparams["epochs"])
    log_metric(tora, "final_test_f1", test_f1 * 100, hyperparams["epochs"])

    # Per-class metrics
    all_targets, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    try:
        cm = confusion_matrix(all_targets, all_preds)
        for class_idx in range(10):
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            log_metric(
                tora, f"class_{class_idx}_precision", prec * 100, hyperparams["epochs"]
            )
            log_metric(
                tora, f"class_{class_idx}_recall", rec * 100, hyperparams["epochs"]
            )
            log_metric(tora, f"class_{class_idx}_f1", f1 * 100, hyperparams["epochs"])
    except Exception as e:
        print(f"Per-class metrics error: {e}")

    tora.shutdown()


if __name__ == "__main__":
    main()
