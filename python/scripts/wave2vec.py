"""
Speech-to-Text Training Script with Tora Tracking
This script fine-tunes a Wav2Vec2 model on the Common Voice English dataset
using Hugging Face Transformers (v4+) and Datasets (>=2.0), with experiment tracking via Tora.
"""

import time

import numpy as np

import torch

import torch.optim as optim

from tora import Tora

from torch.utils.data import DataLoader

from datasets import load_dataset

import evaluate

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from dataclasses import dataclass

from typing import List, Dict, Union


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


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor

    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]

        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def load_data(processor, dataset_name="mozilla-foundation/common_voice_11_0"):
    train_dataset = load_dataset(dataset_name, "en", split="train")

    val_dataset = load_dataset(dataset_name, "en", split="validation")

    test_dataset = load_dataset(dataset_name, "en", split="test")

    for ds in (train_dataset, val_dataset, test_dataset):
        ds.filter(lambda x: len(x.get("audio", {}).get("array", [])) > 0)

    wer_metric = evaluate.load("wer")

    def prepare_batch(batch):
        audio = batch["audio"]

        inputs = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )

        with processor.as_target_processor():
            labels = processor(batch["sentence"], return_tensors="pt").input_ids

        batch["input_values"] = inputs.input_values.squeeze(0)

        batch["attention_mask"] = inputs.attention_mask.squeeze(0)

        batch["labels"] = labels.squeeze(0)

        return batch

    for ds in (train_dataset, val_dataset, test_dataset):
        ds = ds.map(prepare_batch, remove_columns=ds.column_names)

    collator = DataCollatorCTCWithPadding(processor=processor)

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collator
    )

    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collator
    )

    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collator
    )

    return train_loader, val_loader, test_loader, wer_metric


def train_epoch(model, device, loader, optimizer, processor, wer_metric, tora, epoch):
    model.train()

    total_loss = 0.0

    total_wer = 0.0

    start = time.time()

    for i, batch in enumerate(loader):
        inputs = batch["input_values"].to(device)

        masks = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        out = model(inputs, attention_mask=masks, labels=labels)

        loss = out.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        logits = out.logits.detach().cpu().numpy()

        pred_ids = np.argmax(logits, axis=-1)

        preds = processor.batch_decode(pred_ids)

        refs = processor.batch_decode(labels.cpu().numpy(), group_tokens=False)

        batch_wer = wer_metric.compute(predictions=preds, references=refs)

        total_wer += batch_wer * inputs.size(0)

        if i % 50 == 0:
            print(
                f"Epoch {epoch} Batch {i}: loss={loss.item():.4f}, wer={batch_wer:.3f}"
            )

    avg_loss = total_loss / len(loader.dataset)

    avg_wer = total_wer / len(loader.dataset)

    log_metric(tora, "train_loss", avg_loss, epoch)

    log_metric(tora, "train_wer", avg_wer * 100, epoch)

    log_metric(tora, "epoch_time", time.time() - start, epoch)

    return avg_loss, avg_wer


def evaluate_split(
    model, device, loader, processor, wer_metric, tora, epoch, prefix="val"
):
    model.eval()

    total_loss = 0.0

    total_wer = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_values"].to(device)

            masks = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            out = model(inputs, attention_mask=masks, labels=labels)

            total_loss += out.loss.item() * inputs.size(0)

            logits = out.logits.detach().cpu().numpy()

            pred_ids = np.argmax(logits, axis=-1)

            preds = processor.batch_decode(pred_ids)

            refs = processor.batch_decode(labels.cpu().numpy(), group_tokens=False)

            total_wer += wer_metric.compute(
                predictions=preds, references=refs
            ) * inputs.size(0)

    avg_loss = total_loss / len(loader.dataset)

    avg_wer = total_wer / len(loader.dataset)

    log_metric(tora, f"{prefix}_loss", avg_loss, epoch)

    log_metric(tora, f"{prefix}_wer", avg_wer * 100, epoch)

    print(
        f"{prefix.capitalize()} Epoch {epoch}: loss={avg_loss:.4f}, wer={avg_wer:.3f}"
    )

    return avg_loss, avg_wer


def main():
    hyperparams = {
        "batch_size": 8,
        "epochs": 5,
        "lr": 3e-5,
        "model_name": "facebook/wav2vec2-base-960h",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperparams["device"] = str(device)

    processor = Wav2Vec2Processor.from_pretrained(hyperparams["model_name"])

    model = Wav2Vec2ForCTC.from_pretrained(hyperparams["model_name"]).to(device)

    train_loader, val_loader, test_loader, wer_metric = load_data(processor)

    hyperparams.update(
        {
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        }
    )

    tora = Tora.create_experiment(
        name="wav2vec2_finetune_common_voice",
        description="Fine-tune Wav2Vec2 on Common Voice EN",
        hyperparams=hyperparams,
        tags=["speech_to_text", "wav2vec2", "common_voice"],
    )

    optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"])

    best_wer = float("inf")

    for epoch in range(1, hyperparams["epochs"] + 1):
        train_epoch(
            model, device, train_loader, optimizer, processor, wer_metric, tora, epoch
        )

        val_loss, val_wer = evaluate_split(
            model, device, val_loader, processor, wer_metric, tora, epoch, prefix="val"
        )

        if val_wer < best_wer:
            best_wer = val_wer

            torch.save(model.state_dict(), "best_wav2vec2.pt")

    model.load_state_dict(torch.load("best_wav2vec2.pt"))

    evaluate_split(
        model,
        device,
        test_loader,
        processor,
        wer_metric,
        tora,
        hyperparams["epochs"],
        prefix="test",
    )

    tora.shutdown()


if __name__ == "__main__":
    main()
