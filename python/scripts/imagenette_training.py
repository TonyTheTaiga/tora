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

from tora import Tora

from torch.utils.data import DataLoader

from torchvision import datasets, models, transforms


def safe_value(value):
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0

        return float(value)

    elif isinstance(value, bool):
        return int(value)

    elif isinstance(value, str):
        return None

    else:
        try:
            return float(value)

        except (ValueError, TypeError):
            return None


def log_metric(client, name, value, step):
    value = safe_value(value)

    if value is not None:
        client.log(name=name, value=value, step=step)


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, tora):
    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        try:
            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * data.size(0)

            _, predicted = output.max(1)

            total += target.size(0)

            correct += predicted.eq(target).sum().item()

            if batch_idx % 20 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")

    epoch_loss = running_loss / max(total, 1)

    accuracy = 100.0 * correct / max(total, 1)

    epoch_time = time.time() - start_time

    log_metric(tora, "train_loss", epoch_loss, epoch)

    log_metric(tora, "train_accuracy", accuracy, epoch)

    log_metric(tora, "epoch_time", epoch_time, epoch)

    return epoch_loss, accuracy


def validate(model, device, test_loader, criterion, epoch, tora, split="val"):
    model.eval()

    test_loss = 0

    all_targets = []

    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item() * data.size(0)

            pred = output.argmax(dim=1)

            all_targets.extend(target.cpu().numpy())

            all_predictions.extend(pred.cpu().numpy())

    dataset_size = len(test_loader.dataset)

    test_loss = test_loss / max(dataset_size, 1)

    try:
        accuracy = accuracy_score(all_targets, all_predictions) * 100

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted", zero_division=0
        )

    except:
        accuracy, precision, recall, f1 = 0, 0, 0, 0

    prefix = "val" if split == "val" else "test"

    log_metric(tora, f"{prefix}_loss", test_loss, epoch)

    log_metric(tora, f"{prefix}_accuracy", accuracy, epoch)

    log_metric(tora, f"{prefix}_precision", precision * 100, epoch)

    log_metric(tora, f"{prefix}_recall", recall * 100, epoch)

    log_metric(tora, f"{prefix}_f1", f1 * 100, epoch)

    print(
        f"\n{split.capitalize()} set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1 * 100:.2f}%\n"
    )

    return test_loss, accuracy, precision, recall, f1


if __name__ == "__main__":
    hyperparams = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "optimizer": "SGD",
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

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

    data_dir = "data/imagenette2-320"

    if not os.path.exists(data_dir):
        print("Downloading Imagenette dataset...")

        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

        os.makedirs("data", exist_ok=True)

        filename = url.split("/")[-1]

        filepath = os.path.join("data", filename)

        if not os.path.exists(filepath):

            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)

                if percent % 5 == 0:
                    print(f"\rDownloading: {percent}%", end="", flush=True)

            urllib.request.urlretrieve(url, filepath, progress_hook)

            print("\nDownload complete.")

        print("Extracting dataset...")

        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path="data")

        print("Extraction complete.")

    else:
        print(f"Dataset directory {data_dir} already exists. Skipping download.")

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)

    test_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)

    train_size = int(0.8 * len(train_dataset))

    val_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    batch_size = 32

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    class_names = train_dataset.classes

    num_classes = len(class_names)

    model = models.resnet34(weights="IMAGENET1K_V1")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    hyperparams.update(
        {
            "model": "ResNet34",
        }
    )

    tora = Tora.create_experiment(
        name="Imagenette_ResNet34",
        description="ResNet34 model for Imagenette classification with tracked metrics",
        hyperparams=hyperparams,
        tags=["imagenette", "resnet", "image-classification", "transfer-learning"],
    )

    epochs = 5

    lr = hyperparams["lr"]

    weight_decay = hyperparams["weight_decay"]

    momentum = 0.9

    nesterov = True

    dampening = 0

    beta1 = 0.9

    beta2 = 0.999

    eps = 1e-8

    criterion = nn.CrossEntropyLoss()

    if hyperparams["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dampening=dampening,
        )

    elif hyperparams["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
        )

    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        log_metric(tora, "learning_rate", optimizer.param_groups[0]["lr"], epoch)

        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, tora
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, device, val_loader, criterion, epoch, tora, split="val"
        )

        scheduler.step()

    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(
        model, device, test_loader, criterion, epochs, tora, split="test"
    )

    log_metric(tora, "final_test_accuracy", test_acc, epochs)

    log_metric(tora, "final_test_precision", test_prec * 100, epochs)

    log_metric(tora, "final_test_recall", test_rec * 100, epochs)

    log_metric(tora, "final_test_f1", test_f1 * 100, epochs)

    all_targets = []

    all_predictions = []

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1)

            all_targets.extend(target.cpu().numpy())

            all_predictions.extend(pred.cpu().numpy())

    try:
        cm = confusion_matrix(all_targets, all_predictions)

        for class_idx in range(num_classes):
            true_positives = cm[class_idx, class_idx]

            false_positives = cm[:, class_idx].sum() - true_positives

            false_negatives = cm[class_idx, :].sum() - true_positives

            denominator_p = max(true_positives + false_positives, 1)

            denominator_r = max(true_positives + false_negatives, 1)

            class_precision = true_positives / denominator_p

            class_recall = true_positives / denominator_r

            if class_precision + class_recall > 0:
                class_f1 = (
                    2
                    * (class_precision * class_recall)
                    / (class_precision + class_recall)
                )

            else:
                class_f1 = 0

            class_name = class_names[class_idx]

            log_metric(
                tora, f"class_{class_name}_precision", class_precision * 100, epochs
            )

            log_metric(tora, f"class_{class_name}_recall", class_recall * 100, epochs)

            log_metric(tora, f"class_{class_name}_f1", class_f1 * 100, epochs)

    except Exception as e:
        print(f"Error calculating per-class metrics: {str(e)}")

    tora.shutdown()
