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


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        self.relu3 = nn.ReLU()

        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))

        x = self.dropout1(x)

        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.dropout2(x)

        x = x.view(-1, 64 * 7 * 7)

        x = self.relu3(self.fc1(x))

        x = self.dropout3(x)

        x = self.fc2(x)

        return x


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

            if batch_idx % 100 == 0:
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

    model_params = sum(p.numel() for p in SimpleCNN().parameters())

    train_size = int(0.8 * 60000)

    val_size = int(0.2 * 60000)

    test_size = 10000

    hyperparams.update(
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "input_shape": "1x28x28",
            "num_classes": 10,
            "model_parameters": model_params,
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
            "criterion": "CrossEntropyLoss",
        }
    )

    tora = Tora.create_experiment(
        name="MNIST_CNN",
        description="CNN model for MNIST digit classification with tracked metrics",
        hyperparams=hyperparams,
        tags=["mnist", "cnn", "image-classification"],
    )

    batch_size = hyperparams["batch_size"]

    epochs = hyperparams["epochs"]

    lr = hyperparams["lr"]

    momentum = hyperparams["momentum"]

    weight_decay = hyperparams["weight_decay"]

    dropout_rate = hyperparams["dropout_rate"]

    nesterov = hyperparams["nesterov"]

    dampening = hyperparams["dampening"]

    beta1 = hyperparams["beta1"]

    beta2 = hyperparams["beta2"]

    eps = hyperparams["eps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    hyperparams["device"] = str(device)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )

    train_size = int(0.8 * len(train_dataset))

    val_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=batch_size)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleCNN(dropout_rate=dropout_rate).to(device)

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

    model.load_state_dict(torch.load("best_mnist_model.pt"))

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

        for class_idx in range(10):
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

            log_metric(
                tora, f"class_{class_idx}_precision", class_precision * 100, epochs
            )

            log_metric(tora, f"class_{class_idx}_recall", class_recall * 100, epochs)

            log_metric(tora, f"class_{class_idx}_f1", class_f1 * 100, epochs)

    except Exception as e:
        print(f"Error calculating per-class metrics: {str(e)}")

    tora.shutdown()
