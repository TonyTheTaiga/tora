"""
Hugging Face Training Template with Tora Integration
This template provides a structured approach to training Hugging Face models
with Tora experiment tracking. It's organized into three main functions:
1. load_dataset: Handles dataset loading and preprocessing
2. load_model: Loads and configures the model and tokenizer
3. train: Handles the training process with Tora integration
The template can be adapted for various NLP tasks by modifying the task-specific parts.
"""

import os

import time

import argparse

import numpy as np

import torch

from tora import Tora

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from datasets import load_dataset

import evaluate

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def load_dataset(config):
    print(f"Loading dataset: {config['dataset_name']}")

    dataset = load_dataset(config["dataset_name"])

    text_field, label_field = config.get("text_field"), config.get("label_field")

    if not text_field:
        for field in ["text", "sentence", "content"]:
            if field in dataset["train"].features:
                text_field = field

                break

    if config["task_type"] == "classification" and not label_field:
        for field in ["label", "sentiment", "class"]:
            if field in dataset["train"].features:
                label_field = field

                break

    if not text_field:
        raise ValueError("Could not identify text field. Please specify it manually.")

    if config["task_type"] == "classification" and not label_field:
        raise ValueError("Could not identify label field. Please specify it manually.")

    print(f"Using text field: {text_field}")

    if label_field:
        print(f"Using label field: {label_field}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if config["task_type"] == "mlm":

        def tokenize_function(examples):
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=config["max_length"],
                padding="max_length",
                return_special_tokens_mask=True,
            )

        remove_columns = [text_field]

        if label_field:
            remove_columns.append(label_field)

    else:

        def tokenize_function(examples):
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=config["max_length"],
                padding="max_length",
            )

        remove_columns = [text_field]

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
    )

    if "train" in tokenized_datasets and "validation" in tokenized_datasets:
        train_dataset = tokenized_datasets["train"]

        val_dataset = tokenized_datasets["validation"]

    elif "train" in tokenized_datasets and "test" in tokenized_datasets:
        train_size = int(0.9 * len(tokenized_datasets["train"]))

        val_size = len(tokenized_datasets["train"]) - train_size

        splits = tokenized_datasets["train"].train_test_split(
            train_size=train_size, test_size=val_size, seed=config["seed"]
        )

        train_dataset = splits["train"]

        val_dataset = splits["test"]

    else:
        train_size = int(0.9 * len(tokenized_datasets["train"]))

        val_size = len(tokenized_datasets["train"]) - train_size

        splits = tokenized_datasets["train"].train_test_split(
            train_size=train_size, test_size=val_size, seed=config["seed"]
        )

        train_dataset = splits["train"]

        val_dataset = splits["test"]

    test_dataset = tokenized_datasets.get("test", val_dataset)

    num_labels = None

    if config["task_type"] == "classification" and label_field:
        label_names = getattr(dataset["train"].features[label_field], "names", None)

        if label_names:
            num_labels = len(label_names)

        else:
            num_labels = int(max(dataset["train"][label_field])) + 1

    return train_dataset, val_dataset, test_dataset, num_labels, text_field, label_field


def load_model(config, num_labels=None):
    print(f"Loading model: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if config["task_type"] == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(config["model_name"])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config.get("mlm_probability", 0.15),
        )

    elif config["task_type"] == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=num_labels,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    else:
        raise ValueError(f"Unsupported task type: {config['task_type']}")

    model_params = sum(p.numel() for p in model.parameters())

    print(f"Model has {model_params:,} parameters")

    return model, tokenizer, data_collator


def train(
    config,
    model,
    tokenizer,
    data_collator,
    train_dataset,
    val_dataset,
    test_dataset=None,
):
    os.makedirs(config["output_dir"], exist_ok=True)

    config.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset)
            if test_dataset is not None
            else len(val_dataset),
            "model_parameters": sum(p.numel() for p in model.parameters()),
        }
    )

    tora = Tora.create_experiment(
        name=config["experiment_name"],
        description=config["experiment_description"],
        hyperparams=config,
        tags=config.get("tags", []),
    )

    if config["task_type"] == "classification":

        def compute_metrics(pred):
            labels = pred.label_ids

            preds = pred.predictions.argmax(-1)

            accuracy = accuracy_score(labels, preds)

            precision = precision_score(
                labels, preds, average="weighted", zero_division=0
            )

            recall = recall_score(labels, preds, average="weighted", zero_division=0)

            f1 = f1_score(labels, preds, average="weighted", zero_division=0)

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    elif config["task_type"] == "mlm":
        perplexity = evaluate.load("perplexity")

        def compute_metrics(eval_pred):
            result = {}

            try:
                result["perplexity"] = perplexity.compute(
                    predictions=eval_pred.predictions, model_id=config["model_name"]
                )["perplexity"]

            except Exception as e:
                print(f"Error computing perplexity: {str(e)}")

                result["perplexity"] = 0.0

            return result

    else:
        compute_metrics = None

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
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

    print(f"Starting training with {config['epochs']} epochs...")

    start_time = time.time()

    train_result = trainer.train()

    training_time = time.time() - start_time

    train_metrics = train_result.metrics

    trainer.log_metrics("train", train_metrics)

    trainer.save_metrics("train", train_metrics)

    trainer.save_state()

    log_metric(tora, "total_training_time", training_time, trainer.state.global_step)

    print("Performing final evaluation...")

    if test_dataset is not None:
        eval_results = trainer.evaluate(test_dataset)

    else:
        eval_results = trainer.evaluate(val_dataset)

    trainer.log_metrics("eval", eval_results)

    trainer.save_metrics("eval", eval_results)

    for key, value in eval_results.items():
        log_metric(tora, f"final_{key}", value, trainer.state.global_step)

    final_model_dir = f"{config['output_dir']}/final_model"

    trainer.save_model(final_model_dir)

    tokenizer.save_pretrained(final_model_dir)

    print("\nTraining completed!")

    print(f"Model saved to: {final_model_dir}")

    if config["task_type"] == "classification":
        print(f"Final accuracy: {eval_results.get('eval_accuracy', 'N/A')}")

        print(f"Final F1 score: {eval_results.get('eval_f1', 'N/A')}")

    elif config["task_type"] == "mlm":
        print(f"Final perplexity: {eval_results.get('eval_perplexity', 'N/A')}")

    print(f"Total training time: {training_time:.2f} seconds")

    tora.shutdown()

    return trainer, model, tokenizer


def setup_device_and_seed(seed):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        torch.cuda.manual_seed_all(seed)

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)

    np.random.seed(seed)

    return device


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Face training with Tora integration"
    )

    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=["classification", "mlm"],
        help="Task type: classification or mlm",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="imdb",
        help="Dataset name (from Hugging Face datasets)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name (from Hugging Face models)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/model",
        help="Output directory for model and logs",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )

    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "task_type": args.task_type,
        "text_field": None,
        "label_field": None,
        "max_length": 512,
        "batch_size": args.batch_size,
        "eval_batch_size": args.batch_size * 2,
        "epochs": args.epochs,
        "lr": 5e-5,
        "weight_decay": 0.01,
        "seed": args.seed,
        "logging_steps": 500,
        "eval_steps": 1000,
        "save_steps": 2000,
        "warmup_steps": 500,
        "mlm_probability": 0.15,
        "experiment_name": f"{args.task_type.upper()}_{args.model_name.split('/')[-1]}_{args.dataset_name}",
        "experiment_description": f"Training {args.model_name} on {args.dataset_name} for {args.task_type}",
        "tags": ["huggingface", args.task_type, "nlp"],
        "output_dir": args.output_dir,
    }

    device = setup_device_and_seed(config["seed"])

    config["device"] = str(device)

    train_dataset, val_dataset, test_dataset, num_labels, text_field, label_field = (
        load_dataset(config)
    )

    model, tokenizer, data_collator = load_model(config, num_labels)

    trainer, model, tokenizer = train(
        config,
        model,
        tokenizer,
        data_collator,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    return trainer, model, tokenizer


if __name__ == "__main__":
    main()
