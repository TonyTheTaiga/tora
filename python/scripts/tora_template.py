import argparse
import os
import time

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from tora import Tora


def safe_value(value):
    """Convert a value to a safe float or int, handling NaN, inf, bool, and non-numeric values."""
    if isinstance(value, int | float):
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
    """Safely log a metric value to the Tora client."""
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


class ToraCallback(TrainerCallback):
    """Hugging Face Trainer callback to log metrics to Tora."""

    def __init__(self, tora_client):
        self.tora = tora_client

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training loss and learning rate on each step."""
        if not logs:
            return
        if "loss" in logs:
            log_metric(self.tora, "train_loss", logs["loss"], state.global_step)
        if "learning_rate" in logs:
            log_metric(self.tora, "learning_rate", logs["learning_rate"], state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics prefixed with 'eval_'."""
        if not metrics:
            return
        for key, value in metrics.items():
            if key.startswith("eval_"):
                log_metric(self.tora, key, value, state.global_step)


def load_and_tokenize_dataset(config):
    """Load a Hugging Face dataset and tokenize based on task type.

    Returns:
        train_dataset, val_dataset, test_dataset, num_labels, text_field, label_field

    """
    print(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config["dataset_name"])

    text_field = config.get("text_field")
    label_field = config.get("label_field")
    if not text_field:
        for field in ("text", "sentence", "content"):
            if field in dataset["train"].features:
                text_field = field
                break
    if config["task_type"] == "classification" and not label_field:
        for field in ("label", "sentiment", "class"):
            if field in dataset["train"].features:
                label_field = field
                break

    if not text_field:
        raise ValueError("Text field not found; specify with --text_field")
    if config["task_type"] == "classification" and not label_field:
        raise ValueError("Label field not found; specify with --label_field")

    print(f"Using text field: {text_field}")
    if label_field:
        print(f"Using label field: {label_field}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize_mlm(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
            return_special_tokens_mask=True,
        )

    def tokenize_clf(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
        )

    tokenize_fn = tokenize_mlm if config["task_type"] == "mlm" else tokenize_clf
    remove_cols = [text_field] + ([label_field] if label_field else [])

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing dataset",
    )

    if "validation" in tokenized:
        train_ds = tokenized["train"]
        val_ds = tokenized["validation"]
    else:
        splits = tokenized["train"].train_test_split(test_size=0.1, seed=config["seed"])
        train_ds, val_ds = splits["train"], splits["test"]

    test_ds = tokenized.get("test", val_ds)

    num_labels = None
    if config["task_type"] == "classification":
        feat = dataset["train"].features[label_field]
        label_names = getattr(feat, "names", None)
        num_labels = len(label_names) if label_names else max(dataset["train"][label_field]) + 1

    return train_ds, val_ds, test_ds, num_labels, text_field, label_field


def load_model_and_collator(config, num_labels=None):
    """Load a Hugging Face model and appropriate data collator.

    Returns:
        model, tokenizer, data_collator

    """
    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if config["task_type"] == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(config["model_name"])
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config.get("mlm_probability", 0.15),
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=num_labels)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    return model, tokenizer, collator


def train_and_evaluate(
    config,
    model,
    tokenizer,
    data_collator,
    train_dataset,
    val_dataset,
    test_dataset=None,
):
    """Train and evaluate the model using Hugging Face Trainer with Tora logging.

    Returns:
        trainer, model, tokenizer

    """
    os.makedirs(config["output_dir"], exist_ok=True)
    config.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset) if test_dataset else len(val_dataset),
            "model_parameters": sum(p.numel() for p in model.parameters()),
        },
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
            return {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, average="weighted", zero_division=0),
                "recall": recall_score(labels, preds, average="weighted", zero_division=0),
                "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            }

    else:
        perplexity = evaluate.load("perplexity")

        def compute_metrics(pred):
            try:
                res = perplexity.compute(predictions=pred.predictions, model_id=config["model_name"])
                return {"perplexity": res.get("perplexity", 0.0)}
            except Exception as e:
                print(f"Perplexity error: {e}")
                return {"perplexity": 0.0}

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        seed=config["seed"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ToraCallback(tora_client=tora)],
    )

    print(f"Training for {config['epochs']} epochs...")
    start = time.time()
    result = trainer.train()
    training_time = time.time() - start

    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    log_metric(tora, "total_training_time", training_time, trainer.state.global_step)

    print("Final evaluation...")
    eval_ds = test_dataset or val_dataset
    eval_results = trainer.evaluate(eval_ds)
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    for key, val in eval_results.items():
        log_metric(tora, f"final_{key}", val, trainer.state.global_step)

    final_dir = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Training complete!")
    print(f"Model saved to {final_dir}")
    if config["task_type"] == "classification":
        print(f"Accuracy: {eval_results.get('eval_accuracy')}  F1: {eval_results.get('eval_f1')}")
    else:
        print(f"Perplexity: {eval_results.get('eval_perplexity')}")
    print(f"Total time: {training_time:.2f}s")
    tora.shutdown()
    return trainer, model, tokenizer


def setup_device_and_seed(seed):
    """Configure device (GPU/TPU/CPU) and set random seeds."""
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
    parser = argparse.ArgumentParser(description="Train transformer models with Tora integration")
    parser.add_argument("--task_type", choices=["classification", "mlm"], default="classification")
    parser.add_argument("--dataset_name", default="imdb")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="results/model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "task_type": args.task_type,
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
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
        "experiment_description": f"Training {args.model_name} on {args.dataset_name} ({args.task_type})",
        "tags": ["huggingface", args.task_type, "nlp"],
        "output_dir": args.output_dir,
    }

    device = setup_device_and_seed(config["seed"])
    config["device"] = str(device)

    train_ds, val_ds, test_ds, num_labels, _, _ = load_and_tokenize_dataset(config)
    model, tokenizer, collator = load_model_and_collator(config, num_labels)
    train_and_evaluate(config, model, tokenizer, collator, train_ds, val_ds, test_ds)


if __name__ == "__main__":
    main()
