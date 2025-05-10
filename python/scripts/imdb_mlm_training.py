"""
IMDB Masked Language Model Training Script
This script trains a BERT model (bert-base-uncased) on the IMDB dataset
using Masked Language Modeling (MLM). The trained model can then be used
for downstream tasks like sentiment analysis.
The script uses Tora for experiment tracking and HuggingFace's transformers
and datasets libraries.
"""

import os

import time

import numpy as np

import torch

from tora import Tora

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from datasets import load_dataset

import evaluate


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


def load_and_prepare_dataset(tokenizer, max_length=512):
    dataset = load_dataset("imdb")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

        return tokenized

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],
        desc="Tokenizing dataset",
    )

    train_size = int(0.9 * len(tokenized_datasets["train"]))

    val_size = len(tokenized_datasets["train"]) - train_size

    splits = tokenized_datasets["train"].train_test_split(
        train_size=train_size, test_size=val_size, seed=42
    )

    train_dataset = splits["train"]

    val_dataset = splits["test"]

    test_dataset = tokenized_datasets["test"]

    return train_dataset, val_dataset, test_dataset


class ToraCallback(TrainerCallback):
    def __init__(self, tora_client):
        self.tora = tora_client

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        loss = logs.get("loss")

        if loss is not None:
            log_metric(self.tora, "train_loss", loss, state.global_step)

        lr = logs.get("learning_rate")

        if lr is not None:
            log_metric(self.tora, "learning_rate", lr, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        for key, value in metrics.items():
            if key.startswith("eval_"):
                log_metric(self.tora, key, value, state.global_step)


def main():
    hyperparams = {
        "model_name": "google-bert/bert-base-uncased",
        "batch_size": 4,
        "eval_batch_size": 16,
        "epochs": 3,
        "lr": 5e-5,
        "weight_decay": 0.01,
        "max_length": 256,
        "mlm_probability": 0.15,
        "dataset": "imdb",
        "seed": 42,
        "logging_steps": 500,
        "eval_steps": 1000,
        "save_steps": 2000,
        "warmup_steps": 500,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    hyperparams["device"] = str(device)

    torch.manual_seed(hyperparams["seed"])

    np.random.seed(hyperparams["seed"])

    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"])

    model = AutoModelForMaskedLM.from_pretrained(hyperparams["model_name"])

    model_params = sum(p.numel() for p in model.parameters())

    hyperparams["model_parameters"] = model_params

    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(
        tokenizer, max_length=hyperparams["max_length"]
    )

    hyperparams.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
        }
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=hyperparams["mlm_probability"],
    )

    results_dir = "results/imdb_mlm"

    os.makedirs(results_dir, exist_ok=True)

    tora = Tora.create_experiment(
        name="IMDB_MLM_BERT",
        description="BERT-base-uncased trained on IMDB with Masked Language Modeling",
        hyperparams=hyperparams,
        tags=["nlp", "bert", "mlm", "imdb"],
    )

    perplexity = evaluate.load("perplexity")

    def compute_metrics(eval_pred):
        result = {}

        try:
            result["perplexity"] = perplexity.compute(
                predictions=eval_pred.predictions, model_id=hyperparams["model_name"]
            )["perplexity"]

        except Exception as e:
            print(f"Error computing perplexity: {str(e)}")

            result["perplexity"] = 0.0

        return result

    training_args = TrainingArguments(
        output_dir=results_dir,
        overwrite_output_dir=True,
        num_train_epochs=hyperparams["epochs"],
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["eval_batch_size"],
        eval_strategy="steps",
        eval_steps=hyperparams["eval_steps"],
        logging_dir=f"{results_dir}/logs",
        logging_steps=hyperparams["logging_steps"],
        save_steps=hyperparams["save_steps"],
        save_total_limit=2,
        seed=hyperparams["seed"],
        data_seed=hyperparams["seed"],
        learning_rate=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
        warmup_steps=hyperparams["warmup_steps"],
        report_to="none",
    )

    tora_callback = ToraCallback(tora_client=tora)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[tora_callback],
    )

    start_time = time.time()

    trainer.train()

    training_time = time.time() - start_time

    log_metric(tora, "total_training_time", training_time, trainer.state.global_step)

    final_model_dir = f"{results_dir}/final_model"

    trainer.save_model(final_model_dir)

    tokenizer.save_pretrained(final_model_dir)

    eval_results = trainer.evaluate(test_dataset)

    for key, value in eval_results.items():
        log_metric(tora, f"final_{key}", value, trainer.state.global_step)

    print("\nTraining completed!")

    print(f"Model saved to: {final_model_dir}")

    print(f"Final perplexity: {eval_results.get('eval_perplexity', 'N/A')}")

    print(f"Total training time: {training_time:.2f} seconds")

    tora.shutdown()


if __name__ == "__main__":
    main()
