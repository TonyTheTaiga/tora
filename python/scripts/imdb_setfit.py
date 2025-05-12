import os
import time
import random
import argparse
import numpy as np
import torch
from tora import Tora
from datasets import load_dataset, Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Utility helpers (template-compatible)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def prepare_imdb_fewshot(shots_per_class: int, seed: int):
    """Return (train_ds, val_ds, test_ds) with balanced few‑shot sampling."""
    raw = load_dataset("imdb")
    rng = random.Random(seed)

    # Few‑shot subset from training split
    fewshot_idx = []
    for label in [0, 1]:
        idx = [i for i, l in enumerate(raw["train"]["label"]) if l == label]
        rng.shuffle(idx)
        fewshot_idx.extend(idx[:shots_per_class])
    train_ds = raw["train"].select(fewshot_idx)

    # 2 000‑example validation (1 000 / class)
    leftover = [i for i in range(len(raw["train"])) if i not in fewshot_idx]
    val_idx, counts = [], {0: 0, 1: 0}
    for i in leftover:
        lbl = raw["train"]["label"][i]
        if counts[lbl] < 1000:
            val_idx.append(i)
            counts[lbl] += 1
        if counts[0] == counts[1] == 1000:
            break
    val_ds = raw["train"].select(val_idx)

    test_ds = raw["test"]
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Metric computation helpers (device‑safe)
# ---------------------------------------------------------------------------


def _to_numpy(x):
    """Accept list / numpy / torch tensor and return 1‑D numpy array on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(model: SetFitModel, dataset: Dataset):
    # preds may be list[int] or torch.Tensor depending on backend/device
    preds = model.predict(dataset["text"])
    preds = _to_numpy(preds)
    labels = _to_numpy(dataset["label"])
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


# ---------------------------------------------------------------------------
# Training routine with per‑epoch logging
# ---------------------------------------------------------------------------


def train(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Datasets
    train_ds, val_ds, test_ds = prepare_imdb_fewshot(
        config["shots_per_class"], config["seed"]
    )

    # Model – sentence‑transformer encoder + linear head
    model = SetFitModel.from_pretrained(
        config["model_name"], use_differentiable_head=True
    )

    # Experiment tracker
    tora = Tora.create_experiment(
        name=config["experiment_name"],
        description=config["experiment_description"],
        hyperparams=config,
        tags=["setfit", "imdb", "fewshot"],
    )

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric="accuracy",
        batch_size=config["batch_size"],
        column_mapping={"text": "text", "label": "label"},
        num_iterations=config["contrastive_iterations"],
        num_epochs=1,  # manual loop for logging
    )

    print("Starting SetFit training…")
    global_start = time.time()

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        epoch_start = time.time()

        trainer.train()  # one epoch

        # Train‑set metrics (few‑shot; cheap to evaluate)
        train_metrics = compute_metrics(model, train_ds)
        for k, v in train_metrics.items():
            log_metric(tora, f"train_{k}", v, epoch + 1)

        # Validation metrics
        val_metrics = trainer.evaluate()
        for k, v in val_metrics.items():
            log_metric(tora, f"val_{k}", v, epoch + 1)

        epoch_time = time.time() - epoch_start
        log_metric(tora, "epoch_time", epoch_time, epoch + 1)
        print(
            f"  val_accuracy={val_metrics['accuracy']:.4f} | epoch_time={epoch_time:.1f}s"
        )

    total_time = time.time() - global_start

    # Final test metrics
    test_metrics = compute_metrics(model, test_ds)
    for k, v in test_metrics.items():
        log_metric(tora, f"test_{k}", v, config["epochs"])

    # Save model
    final_dir = os.path.join(config["output_dir"], "final_model")
    model.save_pretrained(final_dir)

    print("\nTraining complete!")
    print(
        f"Total time: {total_time / 60:.1f} min | Test accuracy: {test_metrics['accuracy']:.4f}"
    )
    tora.shutdown()


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Few‑shot sentiment classification on IMDb with SetFit (per‑epoch logging)"
    )
    parser.add_argument(
        "--model_name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--output_dir", default="results/imdb_setfit")
    parser.add_argument("--shots_per_class", type=int, default=8)
    parser.add_argument("--contrastive_iterations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "dataset_name": "imdb",
        "shots_per_class": args.shots_per_class,
        "contrastive_iterations": args.contrastive_iterations,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "experiment_name": f"SetFit_{args.model_name.split('/')[-1]}_{args.shots_per_class}shot_IMDB",
        "experiment_description": "Few‑shot SetFit fine‑tuning on IMDb sentiment with per‑epoch logging",
    }
    train(config)


if __name__ == "__main__":
    main()
