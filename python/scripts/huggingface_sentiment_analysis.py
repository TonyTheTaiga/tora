import argparse

import os

import numpy as np

import torch

from datasets import load_dataset

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tora import Tora

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


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


def compute_metrics(pred):
    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)

    precision = precision_score(labels, preds, average="weighted", zero_division=0)

    recall = recall_score(labels, preds, average="weighted", zero_division=0)

    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class ToraCallback(TrainerCallback):
    def __init__(self, tora_client):
        self.tora = tora_client

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        loss = logs.get("loss")

        if loss is not None:
            log_metric(self.tora, "train_loss", loss, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        for key, value in metrics.items():
            if key.startswith("eval_"):
                log_metric(self.tora, key, value, state.global_step)


def train_sentiment_model(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hyperparams = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "device": str(device),
    }

    tora = Tora.create_experiment(
        name=f"Sentiment_Analysis_{args.model_name.split('/')[-1]}",
        description=f"Fine-tuning {args.model_name} on {args.dataset_name} for sentiment analysis",
        hyperparams=hyperparams,
        tags=["nlp", "sentiment-analysis", "transformer", "huggingface"],
    )

    print(f"Loading dataset: {args.dataset_name}")

    if args.dataset_name == "imdb":
        dataset = load_dataset("imdb")

    elif args.dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2")

    else:
        dataset = load_dataset(args.dataset_name)

    if "train" in dataset and "validation" in dataset:
        train_dataset = dataset["train"]

        eval_dataset = dataset["validation"]

    elif "train" in dataset and "test" in dataset:
        train_dataset = dataset["train"]

        eval_dataset = dataset["test"]

    else:
        split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=args.seed)

        train_dataset = split_dataset["train"]

        eval_dataset = split_dataset["test"]

    text_field, label_field = None, None

    for field in ["text", "sentence", "content"]:
        if field in train_dataset.features:
            text_field = field

            break

    for field in ["label", "sentiment", "class"]:
        if field in train_dataset.features:
            label_field = field

            break

    if not text_field or not label_field:
        print("Could not identify text and label fields. Please specify them manually.")

        return

    print(f"Using text field: {text_field}, label field: {label_field}")

    label_names = getattr(train_dataset.features[label_field], "names", None)

    if label_names:
        num_labels = len(label_names)

    else:
        num_labels = int(max(train_dataset[label_field])) + 1

    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    model_size = sum(p.numel() for p in model.parameters())

    hyperparams["model_parameters"] = model_size

    print(f"Model size: {model_size} parameters")

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)

    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    loss_callback = ToraCallback(tora)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback],
    )

    print("Starting training...")

    train_result = trainer.train()

    train_metrics = train_result.metrics

    trainer.log_metrics("train", train_metrics)

    trainer.save_metrics("train", train_metrics)

    trainer.save_state()

    print("Evaluating model...")

    eval_metrics = trainer.evaluate()

    trainer.log_metrics("eval", eval_metrics)

    trainer.save_metrics("eval", eval_metrics)

    trainer.save_model(os.path.join(args.output_dir, "final_model"))

    if label_names:
        predictions = trainer.predict(tokenized_eval)

        preds = np.argmax(predictions.predictions, axis=1)

        labels = predictions.label_ids

        for i, class_name in enumerate(label_names):
            class_precision = precision_score(labels == i, preds == i, zero_division=0)

            class_recall = recall_score(labels == i, preds == i, zero_division=0)

            class_f1 = f1_score(labels == i, preds == i, zero_division=0)

            log_metric(
                tora,
                f"class_{class_name}_precision",
                class_precision * 100,
                args.epochs,
            )

            log_metric(
                tora, f"class_{class_name}_recall", class_recall * 100, args.epochs
            )

            log_metric(tora, f"class_{class_name}_f1", class_f1 * 100, args.epochs)

            print(
                f"Class {class_name}: Precision={class_precision:.4f}, Recall={class_recall:.4f}, F1={class_f1:.4f}"
            )

    print(
        f"Training complete. Model saved to {os.path.join(args.output_dir, 'final_model')}"
    )

    tora.shutdown()

    return trainer, model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a sentiment analysis model using Hugging Face"
    )

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")

    parser.add_argument("--dataset_name", type=str, default="sst2")

    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=2e-5)

    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="results/sentiment_model")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_sentiment_model(args)
