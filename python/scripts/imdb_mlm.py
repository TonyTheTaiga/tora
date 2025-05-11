import os
import time
import math
import argparse
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
    """Minimal Tora callback that logs training/eval loss and learning‑rate."""

    def __init__(self, tora_client):
        self.tora = tora_client

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            log_metric(self.tora, "train_loss", logs["loss"], state.global_step)
        if "learning_rate" in logs:
            log_metric(
                self.tora, "learning_rate", logs["learning_rate"], state.global_step
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        for key, value in metrics.items():
            if key.startswith("eval_"):
                log_metric(self.tora, key, value, state.global_step)


def prepare_imdb_dataset(tokenizer, max_length, seed):
    """Loads IMDb as plain text and tokenises for MLM."""
    raw = load_dataset("imdb")  # splits: train/test

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenised = raw.map(
        tokenize, batched=True, remove_columns=["text", "label"], desc="Tokenising IMDb"
    )

    # create validation split from training set (90/10)
    split = tokenised["train"].train_test_split(test_size=0.1, seed=seed)
    train_ds, val_ds = split["train"], split["test"]
    test_ds = tokenised["test"]
    return train_ds, val_ds, test_ds


def load_model_and_collator(model_name, mlm_prob):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )
    print(
        f"Loaded {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model, tokenizer, collator


def train(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # model + data
    model, tokenizer, collator = load_model_and_collator(
        config["model_name"], config["mlm_probability"]
    )
    train_ds, val_ds, test_ds = prepare_imdb_dataset(
        tokenizer, config["max_length"], config["seed"]
    )

    # where we log everything
    tora = Tora.create_experiment(
        name=config["experiment_name"],
        description=config["experiment_description"],
        hyperparams=config,
        tags=["mlm", "imdb", "nlp"],
    )
    tora_callback = ToraCallback(tora)

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
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        seed=config["seed"],
        data_seed=config["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # disable HF default reporting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[tora_callback],
    )

    print("Starting training…")
    start = time.time()
    trainer.train()
    duration = time.time() - start

    # final evaluation on test
    print("Evaluating on test set…")
    results = trainer.evaluate(test_ds)
    results["perplexity"] = math.exp(results["eval_loss"])

    trainer.log_metrics("test", results)
    trainer.save_metrics("test", results)

    for k, v in results.items():
        log_metric(tora, f"final_{k}", v, trainer.state.global_step)

    # save artefacts
    final_dir = os.path.join(config["output_dir"], "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(
        f"\nTraining finished in {duration / 60:.1f}min  |  test perplexity: {results['perplexity']:.2f}"
    )
    tora.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Train a small transformer encoder on IMDb for MLM"
    )
    parser.add_argument(
        "--model_name", default="prajjwal1/bert-tiny", help="HF model identifier"
    )
    parser.add_argument(
        "--output_dir", default="results/imdb_mlm", help="Where to store checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "dataset_name": "imdb",
        "max_length": 256,
        "mlm_probability": 0.15,
        "batch_size": args.batch_size * 2,
        "eval_batch_size": args.batch_size * 2,
        "epochs": args.epochs,
        "lr": 7e-5,
        "weight_decay": 0.01,
        "warmup_steps": 300,
        "eval_steps": 1000,
        "logging_steps": 200,
        "save_steps": 2000,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "experiment_name": f"MLM_{args.model_name.split('/')[-1]}_IMDB",
        "experiment_description": "Masked‑language‑model fine‑tuning on IMDb",
    }
    train(config)


if __name__ == "__main__":
    main()
