import os
import time
import argparse
import numpy as np
import torch
from tora import Tora
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------- Utility helpers --------------------


def safe_value(value):
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, bool):
        return int(value)
    else:
        return None


def log_metric(client, name, value, step):
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


class ToraCallback(TrainerCallback):
    """Logs loss / lr / eval metrics to Tora."""

    def __init__(self, tora):
        self.tora = tora

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            log_metric(self.tora, "train_loss", logs["loss"], state.global_step)
        if "learning_rate" in logs:
            log_metric(self.tora, "lr", logs["learning_rate"], state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        for k, v in metrics.items():
            if k.startswith("eval_"):
                log_metric(self.tora, k, v, state.global_step)


# -------------------- Data prep --------------------


def load_imdb(tokenizer, max_len, seed):
    raw = load_dataset("imdb")

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=max_len)

    tokenised = raw.map(tok_fn, batched=True, remove_columns=["text"])

    split = tokenised["train"].train_test_split(test_size=0.1, seed=seed)
    train_ds, val_ds = split["train"], split["test"]
    return train_ds, val_ds, tokenised["test"]


# -------------------- Metrics --------------------


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
    }


# -------------------- Main training routine --------------------


def train(config):
    os.makedirs(config["output_dir"], exist_ok=True)

    # set seeds + device
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load MLM‑fine‑tuned encoder weights + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["weights_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["weights_dir"], num_labels=2, ignore_mismatched_sizes=True
    )

    # datasets
    train_ds, val_ds, test_ds = load_imdb(
        tokenizer, config["max_length"], config["seed"]
    )
    collator = DataCollatorWithPadding(tokenizer)

    # experiment tracker
    tora = Tora.create_experiment(
        name=config["experiment_name"],
        description=config["experiment_description"],
        hyperparams=config,
        tags=["sentiment", "imdb", "nlp"],
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        seed=config["seed"],
        data_seed=config["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[ToraCallback(tora)],
    )

    # training
    print("Starting fine‑tuning for sentiment analysis…")
    start = time.time()
    trainer.train()
    duration = time.time() - start

    # evaluation on test
    results = trainer.evaluate(test_ds)
    trainer.log_metrics("test", results)
    trainer.save_metrics("test", results)
    for k, v in results.items():
        log_metric(tora, f"final_{k}", v, trainer.state.global_step)

    # save artefacts
    final_dir = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(
        f"\nDone! Test accuracy: {results['eval_accuracy']:.4f} | time: {duration / 60:.1f} min"
    )
    tora.shutdown()


# -------------------- Argparse / entry --------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine‑tune MLM‑pretrained encoder for IMDb sentiment classification"
    )
    parser.add_argument(
        "--weights_dir",
        required=True,
        help="Path to directory with MLM‑trained weights + tokenizer",
    )
    parser.add_argument(
        "--output_dir", default="results/imdb_sa", help="Where to store checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "weights_dir": args.weights_dir,
        "output_dir": args.output_dir,
        "max_length": 256,
        "batch_size": args.batch_size,
        "eval_batch_size": args.batch_size * 2,
        "epochs": args.epochs,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "eval_steps": 500,
        "logging_steps": 200,
        "save_steps": 1000,
        "seed": args.seed,
        "experiment_name": "SA_from_MLM_IMDB",
        "experiment_description": "Sentiment fine‑tuning using MLM‑adapted encoder on IMDb",
    }
    train(config)


if __name__ == "__main__":
    main()
