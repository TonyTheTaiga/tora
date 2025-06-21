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
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from tora import Tora


def safe_value(value):
    """
    Convert various types to safe numeric values, handling NaN, inf, and non-numeric.
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
    Safely log a metric to the Tora client if value is valid.
    """
    val = safe_value(value)
    if val is not None:
        client.log(name=name, value=val, step=step)


class TextClassificationDataset(Dataset):
    """
    PyTorch dataset for text classification tasks.
    Tokenizes texts and pairs with labels.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier(nn.Module):
    """
    BERT-based classifier for text classification.
    """

    def __init__(self, num_classes, dropout_rate=0.1, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits


def train_epoch(model, device, loader, optimizer, criterion, epoch, tora):
    """
    Train the model for one epoch and log training metrics.
    Returns:
        epoch_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        try:
            logits = model(input_ids=input_ids, attention_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}"
                )
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")

    epoch_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    elapsed = time.time() - start

    log_metric(tora, "train_loss", epoch_loss, epoch)
    log_metric(tora, "train_accuracy", accuracy, epoch)
    log_metric(tora, "epoch_time", elapsed, epoch)

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
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=mask)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            preds = logits.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    size = len(loader.dataset)
    avg_loss = total_loss / max(size, 1)

    try:
        accuracy = accuracy_score(all_targets, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets,
            all_preds,
            average="weighted",
            zero_division=0,
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
        f"{split.title()} set: Loss={avg_loss:.4f}, "
        f"Acc={accuracy:.2f}%, F1={f1 * 100:.2f}%"
    )

    return avg_loss, accuracy, precision, recall, f1


if __name__ == "__main__":
    # Hyperparameters
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
    }

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    hyperparams["device"] = str(device)

    # Load SST-2 dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("glue", "sst2")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds = TextClassificationDataset(
        train_data["sentence"],
        train_data["label"],
        tokenizer,
        max_length=hyperparams["max_length"],
    )
    val_ds = TextClassificationDataset(
        val_data["sentence"],
        val_data["label"],
        tokenizer,
        max_length=hyperparams["max_length"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=hyperparams["batch_size"])

    # Initialize model and Tora experiment
    num_classes = 2
    model = BertClassifier(
        num_classes=num_classes,
        dropout_rate=hyperparams["dropout_rate"],
        freeze_bert=hyperparams["freeze_bert"],
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hyperparams.update(
        {
            "model": "BertClassifier",
            "num_classes": num_classes,
            "model_parameters": model_params,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "criterion": "CrossEntropyLoss",
        }
    )

    tora = Tora.create_experiment(
        name="SST2_BERT",
        description="BERT sentiment classification on SST-2",
        hyperparams=hyperparams,
        tags=["nlp", "bert", "sentiment", "classification"],
    )

    # Optimizer and scheduler
    if hyperparams["optimizer"] == "AdamW":
        from transformers import AdamW, get_linear_schedule_with_warmup

        optimizer = AdamW(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            betas=(hyperparams["beta1"], hyperparams["beta2"]),
            eps=hyperparams["eps"],
        )
        total_steps = len(train_loader) * hyperparams["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=hyperparams["warmup_steps"],
            num_training_steps=total_steps,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    best_path = "best_sst2_model.pt"
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
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: val_acc={best_val_acc:.2f}%")

        if scheduler:
            scheduler.step()

    # Final evaluation
    print(f"Loading best model (acc={best_val_acc:.2f}%)...")
    model.load_state_dict(torch.load(best_path))
    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(
        model, device, val_loader, criterion, hyperparams["epochs"], tora, split="test"
    )

    # Log final metrics
    log_metric(tora, "final_test_accuracy", test_acc, hyperparams["epochs"])
    log_metric(tora, "final_test_precision", test_prec * 100, hyperparams["epochs"])
    log_metric(tora, "final_test_recall", test_rec * 100, hyperparams["epochs"])
    log_metric(tora, "final_test_f1", test_f1 * 100, hyperparams["epochs"])

    # Per-class metrics
    all_t, all_p = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)
            out = model(input_ids=inp, attention_mask=mask)
            preds = out.argmax(dim=1)
            all_t.extend(lbl.cpu().numpy())
            all_p.extend(preds.cpu().numpy())

    try:
        cm = confusion_matrix(all_t, all_p)
        class_names = ["negative", "positive"]
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
