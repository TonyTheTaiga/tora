#!/usr/bin/env python3
"""
Fine‑tune Wav2Vec2 on Common Voice English for ASR.

Tested with:
    • python 3.11
    • torch 2.3.0
    • transformers 4.41.2
    • datasets 2.20.0
    • evaluate 0.5.0
    • torchaudio 2.3.0
    macOS 14 (Apple Silicon) – works on CPU, CUDA or MPS
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

import evaluate

try:
    from tora import Tora  # optional experiment tracker
except ImportError:
    Tora = None

# -------------------------- utility --------------------------------- #


def safe_value(value):
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, bool):
        return int(value)
    return None


def log_metric(client, name, value, step):
    if client is None:
        return
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


# --------------------- data collator -------------------------------- #


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )

        # Replace padding with -100 so they are ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


# -------------------------- data ------------------------------------ #


def load_data(
    processor, dataset_name="mozilla-foundation/common_voice_17_0", lang="en"
):
    """Load Common Voice splits and convert to model‑ready tensors."""
    train_ds = load_dataset(dataset_name, lang, split="train")
    val_ds = load_dataset(dataset_name, lang, split="validation")
    test_ds = load_dataset(dataset_name, lang, split="test")

    # Drop examples without audio
    for split, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        ds = ds.filter(
            lambda ex: ex["audio"] is not None and len(ex["audio"]["array"]) > 0,
            num_proc=4,
        )
        if split == "train":
            train_ds = ds
        elif split == "val":
            val_ds = ds
        else:
            test_ds = ds

    wer_metric = evaluate.load("wer")

    def prepare(example):
        audio = example["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )
        example["input_values"] = inputs.input_values.squeeze(0)
        example["attention_mask"] = inputs.attention_mask.squeeze(0)
        with processor.as_target_processor():
            example["labels"] = processor(
                example["sentence"], return_tensors="pt"
            ).input_ids.squeeze(0)
        return example

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, num_proc=4)
    val_ds = val_ds.map(prepare, remove_columns=val_ds.column_names, num_proc=4)
    test_ds = test_ds.map(prepare, remove_columns=test_ds.column_names, num_proc=4)

    collator = DataCollatorCTCWithPadding(processor)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collator)
    return train_loader, val_loader, test_loader, wer_metric


# ----------------------- training utils ----------------------------- #


def train_epoch(
    model,
    device,
    loader,
    optimizer,
    processor,
    wer_metric,
    tracker,
    epoch,
):
    model.train()
    total_loss, total_wer = 0.0, 0.0
    start = time.time()

    for step, batch in enumerate(loader, 1):
        inputs = batch["input_values"].to(device)
        masks = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        pred_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = labels.cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(label_ids, group_tokens=False)

        batch_wer = wer_metric.compute(predictions=preds, references=refs)
        total_wer += batch_wer * inputs.size(0)

        if step % 50 == 0:
            print(
                f"Epoch {epoch} | Step {step}/{len(loader)} "
                f"loss={loss.item():.4f} wer={batch_wer:.3f}"
            )

    avg_loss = total_loss / len(loader.dataset)
    avg_wer = total_wer / len(loader.dataset)

    log_metric(tracker, "train_loss", avg_loss, epoch)
    log_metric(tracker, "train_wer", avg_wer * 100, epoch)
    log_metric(tracker, "epoch_time", time.time() - start, epoch)
    return avg_loss, avg_wer


@torch.no_grad()
def evaluate_split(
    model,
    device,
    loader,
    processor,
    wer_metric,
    tracker,
    epoch,
    prefix="val",
):
    model.eval()
    total_loss, total_wer = 0.0, 0.0

    for batch in loader:
        inputs = batch["input_values"].to(device)
        masks = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(inputs, attention_mask=masks, labels=labels)
        total_loss += outputs.loss.item() * inputs.size(0)

        pred_ids = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = labels.cpu().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(label_ids, group_tokens=False)

        total_wer += wer_metric.compute(
            predictions=preds, references=refs
        ) * inputs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    avg_wer = total_wer / len(loader.dataset)

    log_metric(tracker, f"{prefix}_loss", avg_loss, epoch)
    log_metric(tracker, f"{prefix}_wer", avg_wer * 100, epoch)
    print(f"{prefix.capitalize()} epoch {epoch}: loss={avg_loss:.4f} wer={avg_wer:.3f}")
    return avg_loss, avg_wer


# ------------------------- main ------------------------------------- #


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    hyperparams = dict(
        batch_size=8,
        epochs=5,
        lr=3e-5,
        model_name="facebook/wav2vec2-base-960h",
    )

    device = choose_device()
    hyperparams["device"] = str(device)

    processor = Wav2Vec2Processor.from_pretrained(hyperparams["model_name"])
    model = Wav2Vec2ForCTC.from_pretrained(hyperparams["model_name"]).to(device)

    loaders = load_data(processor)
    train_loader, val_loader, test_loader, wer_metric = loaders

    hyperparams.update(
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
        test_samples=len(test_loader.dataset),
    )

    tracker = None
    if Tora is not None:
        tracker = Tora.create_experiment(
            name="wav2vec2_finetune_common_voice",
            description="Fine‑tune Wav2Vec2 on Common Voice EN",
            hyperparams=hyperparams,
            tags=["speech_to_text", "wav2vec2", "common_voice"],
        )

    optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"])

    best_wer = float("inf")
    for epoch in range(1, hyperparams["epochs"] + 1):
        train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            processor,
            wer_metric,
            tracker,
            epoch,
        )
        _, val_wer = evaluate_split(
            model,
            device,
            val_loader,
            processor,
            wer_metric,
            tracker,
            epoch,
            prefix="val",
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
        tracker,
        hyperparams["epochs"],
        prefix="test",
    )

    if tracker is not None:
        tracker.shutdown()


if __name__ == "__main__":
    main()
