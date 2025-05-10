import time

import datasets

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

from torch.utils.data import DataLoader, Dataset

from transformers import BertModel, BertTokenizer


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


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])

        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1, freeze_bert=False):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, tora):
    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(input_ids)}/{len(train_loader.dataset)}"
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
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            test_loss += criterion(outputs, labels).item() * input_ids.size(0)

            pred = outputs.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())

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
        "batch_size": 16,
        "epochs": 5,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "dropout_rate": 0.1,
        "max_length": 128,
        "freeze_bert": True,
        "warmup_steps": 0,
        "scheduler": "linear",
        "optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    hyperparams["device"] = str(device)

    print("Loading dataset...")

    sst2 = datasets.load_dataset("glue", "sst2")

    train_dataset = sst2["train"]

    validation_dataset = sst2["validation"]

    num_classes = 2

    class_names = ["negative", "positive"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_texts = train_dataset["sentence"]

    train_labels = train_dataset["label"]

    val_texts = validation_dataset["sentence"]

    val_labels = validation_dataset["label"]

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length=hyperparams["max_length"]
    )

    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length=hyperparams["max_length"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])

    model = BertClassifier(
        num_classes=num_classes,
        dropout_rate=hyperparams["dropout_rate"],
        freeze_bert=hyperparams["freeze_bert"],
    )

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hyperparams.update(
        {
            "dataset": "SST-2",
            "model": "BertClassifier",
            "num_classes": num_classes,
            "model_parameters": model_params,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "criterion": "CrossEntropyLoss",
        }
    )

    tora = Tora.create_experiment(
        name="SST2_BERT",
        description="BERT model for SST-2 sentiment classification with tracked metrics",
        hyperparams=hyperparams,
        tags=["nlp", "bert", "sentiment-analysis", "text-classification"],
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if hyperparams["optimizer"] == "AdamW":
        from transformers import AdamW

        optimizer = AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            betas=(hyperparams["beta1"], hyperparams["beta2"]),
            eps=hyperparams["eps"],
        )

    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )

    from transformers import get_linear_schedule_with_warmup

    num_training_steps = len(train_loader) * hyperparams["epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hyperparams["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    best_val_acc = 0

    best_model_path = "best_sst2_model.pt"

    for epoch in range(1, hyperparams["epochs"] + 1):
        log_metric(tora, "learning_rate", optimizer.param_groups[0]["lr"], epoch)

        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, tora
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, device, val_loader, criterion, epoch, tora, split="val"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(), best_model_path)

            print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

        scheduler.step()

    print(f"Loading best model with validation accuracy: {best_val_acc:.2f}%")

    model.load_state_dict(torch.load(best_model_path))

    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(
        model, device, val_loader, criterion, hyperparams["epochs"], tora, split="test"
    )

    log_metric(tora, "final_test_accuracy", test_acc, hyperparams["epochs"])

    log_metric(tora, "final_test_precision", test_prec * 100, hyperparams["epochs"])

    log_metric(tora, "final_test_recall", test_rec * 100, hyperparams["epochs"])

    log_metric(tora, "final_test_f1", test_f1 * 100, hyperparams["epochs"])

    all_targets = []

    all_predictions = []

    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            pred = outputs.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())

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
                tora,
                f"class_{class_name}_precision",
                class_precision * 100,
                hyperparams["epochs"],
            )

            log_metric(
                tora,
                f"class_{class_name}_recall",
                class_recall * 100,
                hyperparams["epochs"],
            )

            log_metric(
                tora, f"class_{class_name}_f1", class_f1 * 100, hyperparams["epochs"]
            )

    except Exception as e:
        print(f"Error calculating per-class metrics: {str(e)}")

    tora.shutdown()
