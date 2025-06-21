import os
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from tora import Tora


def safe_value(value):
    """
    Convert a value to a safe float or int, handling NaN, inf, bools, and non-numeric.
    """
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, bool):
        return int(value)
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


def download_and_extract_imagenette(data_dir: str, url: str):
    """
    Download and extract the Imagenette dataset if not present.
    """
    if not os.path.exists(data_dir):
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        archive = f"{data_dir}.tgz"
        print(f"Downloading Imagenette dataset to {archive}...")

        def progress_hook(count, block_size, total_size):
            pct = int(count * block_size * 100 / total_size)
            if pct % 5 == 0:
                print(f"\rDownloading: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, archive, progress_hook)
        print("\nDownload complete. Extracting...")

        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(data_dir))
        os.remove(archive)
        print("Extraction complete.")
    else:
        print(f"Dataset directory {data_dir} exists. Skipping download.")


def get_data_loaders(data_dir: str, batch_size: int):
    """
    Prepare train, validation, and test DataLoaders for Imagenette and return class names.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Load full train dataset and test dataset
    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Split full train into train and validation
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Return loaders and class names from the training folder
    return train_loader, val_loader, test_loader, full_train.classes


def train_epoch(model, device, loader, optimizer, criterion, epoch, tora):
    """
    Train model for one epoch and log training metrics.
    """
    model.train()
    running_loss = 0.0
    correct = total = 0
    start = time.time()

    for idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if idx % 20 == 0:
            print(f"Epoch {epoch} [{idx}/{len(loader)}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    elapsed = time.time() - start

    log_metric(tora, "train_loss", epoch_loss, epoch)
    log_metric(tora, "train_accuracy", accuracy, epoch)
    log_metric(tora, "epoch_time", elapsed, epoch)

    return epoch_loss, accuracy


def validate(model, device, loader, criterion, epoch, tora, split="val"):
    """
    Evaluate model on validation or test set and log metrics.
    """
    model.eval()
    total_loss = 0.0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    try:
        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="weighted", zero_division=0
        )
    except Exception:
        accuracy = precision = recall = f1 = 0.0

    log_metric(tora, f"{split}_loss", avg_loss, epoch)
    log_metric(tora, f"{split}_accuracy", accuracy, epoch)
    log_metric(tora, f"{split}_precision", precision * 100, epoch)
    log_metric(tora, f"{split}_recall", recall * 100, epoch)
    log_metric(tora, f"{split}_f1", f1 * 100, epoch)

    print(
        f"{split.title()} set: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, F1={f1 * 100:.2f}%"
    )
    return avg_loss, accuracy, precision, recall, f1


def main():
    # Hyperparameters
    hyperparams = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "SGD",
        "batch_size": 32,
        "epochs": 5,
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    hyperparams["device"] = str(device)

    data_dir = "data/imagenette2-320"
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    download_and_extract_imagenette(data_dir, url)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir, hyperparams["batch_size"]
    )
    num_classes = len(class_names)

    # Model setup
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    hyperparams.update({"model": "ResNet34", "num_classes": num_classes})
    tora = Tora.create_experiment(
        workspace_id="84679cc3-dae5-40c7-91f3-10de46123765",
        name="Imagenette_ResNet34",
        description="ResNet34 on Imagenette-320",
        hyperparams=hyperparams,
        tags=["imagenette", "resnet", "transfer-learning"],
    )

    criterion = nn.CrossEntropyLoss()
    if hyperparams["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams["lr"],
            momentum=0.9,
            weight_decay=hyperparams["weight_decay"],
            nesterov=True,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparams["epochs"]
    )

    # Training loop
    for epoch in range(1, hyperparams["epochs"] + 1):
        log_metric(tora, "learning_rate", optimizer.param_groups[0]["lr"], epoch)
        train_epoch(model, device, train_loader, optimizer, criterion, epoch, tora)
        validate(model, device, val_loader, criterion, epoch, tora, split="val")
        scheduler.step()

    # Final evaluation
    validate(
        model, device, test_loader, criterion, hyperparams["epochs"], tora, split="test"
    )

    # Per-class metrics
    all_targets, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    try:
        cm = confusion_matrix(all_targets, all_preds)
        for idx, cname in enumerate(class_names):
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            log_metric(
                tora, f"class_{cname}_precision", prec * 100, hyperparams["epochs"]
            )
            log_metric(tora, f"class_{cname}_recall", rec * 100, hyperparams["epochs"])
            log_metric(tora, f"class_{cname}_f1", f1 * 100, hyperparams["epochs"])
    except Exception as e:
        print(f"Error computing per-class metrics: {e}")

    tora.shutdown()


if __name__ == "__main__":
    main()
