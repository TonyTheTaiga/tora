import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ndcg_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split

from tora import Tora


def safe_value(value):
    """
    Convert a value to a safe numeric type, handling NaN, inf, and non-numeric values.
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
    Log a single metric to Tora if the value is valid.
    """
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


def download_movielens_dataset(data_dir="data/ml-100k"):
    """
    Download and extract the MovieLens 100K dataset if not already present.
    """
    if not os.path.exists(data_dir):
        print("Downloading MovieLens 100K dataset...")
        import urllib.request
        import zipfile

        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = f"{data_dir}.zip"

        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(data_dir))
        os.remove(zip_path)
        print("Download complete!")
    else:
        print("MovieLens 100K dataset already exists. Skipping download.")


def load_movielens_data():
    """
    Load ratings and movie metadata from the MovieLens 100K dataset.
    Returns:
        ratings (DataFrame), movies (DataFrame), n_users (int), n_items (int)
    """
    download_movielens_dataset()

    ratings = pd.read_csv(
        "data/ml-100k/u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    movies = pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "action",
            "adventure",
            "animation",
            "children",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film-noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci-fi",
            "thriller",
            "war",
            "western",
        ],
        encoding="latin-1",
        engine="python",
    )

    # Binarize ratings
    threshold = 4.0
    ratings["rating_binary"] = (ratings.rating >= threshold).astype(int)

    n_users = ratings.user_id.nunique()
    n_items = ratings.item_id.nunique()

    print(
        f"Loaded MovieLens 100K dataset: Users={n_users}, Items={n_items}, Ratings={len(ratings)}"
    )
    return ratings, movies, n_users, n_items


class RecommenderDataset(Dataset):
    """
    PyTorch Dataset for user-item interactions, supports binary or continuous ratings.
    """

    def __init__(self, ratings_df, n_users, n_items, binary=True):
        self.ratings_df = ratings_df
        self.n_users = n_users
        self.n_items = n_items
        self.binary = binary
        self.rating_col = "rating_binary" if binary else "rating"

        self.user_ids = (ratings_df.user_id.values - 1).astype(np.int64)
        self.item_ids = (ratings_df.item_id.values - 1).astype(np.int64)
        self.ratings = ratings_df[self.rating_col].values

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "rating": torch.tensor(
                self.ratings[idx],
                dtype=torch.float if not self.binary else torch.long,
            ),
        }


class MatrixFactorization(nn.Module):
    """
    Simple matrix factorization model with optional binary output.
    """

    def __init__(self, n_users, n_items, n_factors, dropout=0.2, binary=True):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.dropout = nn.Dropout(dropout)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.binary = binary
        if binary:
            self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_id, item_id):
        u = self.dropout(self.user_embedding(user_id))
        v = self.dropout(self.item_embedding(item_id))
        bias_u = self.user_bias(user_id).squeeze()
        bias_v = self.item_bias(item_id).squeeze()
        dot = (u * v).sum(dim=1)
        pred = dot + bias_u + bias_v + self.global_bias
        return self.sigmoid(pred) if self.binary else pred


def create_negative_samples(ratings_df, n_items, neg_ratio=1):
    """
    Generate negative samples for each user by sampling unobserved items.
    """
    user_item = set(zip(ratings_df.user_id, ratings_df.item_id))
    negatives = []
    for user in ratings_df.user_id.unique():
        pos_items = set(ratings_df[ratings_df.user_id == user].item_id)
        candidates = set(range(1, n_items + 1)) - pos_items
        n_neg = min(len(pos_items) * neg_ratio, len(candidates))
        for item in random.sample(candidates, int(n_neg)):
            negatives.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "rating": 0,
                    "timestamp": 0,
                    "rating_binary": 0,
                }
            )
    neg_df = pd.DataFrame(negatives)
    return pd.concat([ratings_df, neg_df], ignore_index=True)


def train_epoch(
    model, device, train_loader, optimizer, criterion, epoch, tora, binary=True
):
    """
    Train model for one epoch and log metrics to Tora.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    start = time.time()

    for idx, batch in enumerate(train_loader):
        u = batch["user_id"].to(device)
        v = batch["item_id"].to(device)
        r = batch["rating"].to(device).float()

        optimizer.zero_grad()
        try:
            out = model(u, v)
            if binary:
                loss = criterion(out, r)
                preds = (out >= 0.5).float()
                correct += (preds == r).sum().item()
            else:
                loss = criterion(out, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * u.size(0)
            count += u.size(0)
            if idx % 50 == 0:
                print(
                    f"Epoch {epoch} [{idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )
        except Exception as e:
            print(f"Error batch {idx}: {e}")

    avg_loss = total_loss / max(count, 1)
    duration = time.time() - start
    log_metric(tora, "train_loss", avg_loss, epoch)
    log_metric(tora, "epoch_time", duration, epoch)
    if binary:
        acc = 100.0 * correct / max(count, 1)
        log_metric(tora, "train_accuracy", acc, epoch)
        return avg_loss, acc
    return avg_loss


def validate(model, device, val_loader, criterion, epoch, tora, binary=True, k=10):
    """
    Evaluate model on validation set, logging loss and ranking metrics.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    all_targets, all_scores, all_preds = [], [], []
    user_preds = {}

    with torch.no_grad():
        for batch in val_loader:
            u = batch["user_id"].to(device)
            v = batch["item_id"].to(device)
            r = batch["rating"].to(device).float()
            out = model(u, v)
            loss = criterion(out, r)
            total_loss += loss.item() * u.size(0)
            count += u.size(0)

            if binary:
                all_targets.extend(r.cpu().numpy())
                all_scores.extend(out.cpu().numpy())
                preds = (out >= 0.5).float().cpu().numpy()
                all_preds.extend(preds)

                for i in range(len(u)):
                    user = u[i].item()
                    user_preds.setdefault(user, []).append(
                        (v[i].item(), out[i].item(), r[i].item())
                    )

    val_loss = total_loss / max(count, 1)
    log_metric(tora, "val_loss", val_loss, epoch)

    if binary and all_targets:
        acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))
        prec = 100.0 * precision_score(all_targets, all_preds, zero_division=0)
        rec = 100.0 * recall_score(all_targets, all_preds, zero_division=0)
        log_metric(tora, "val_accuracy", acc, epoch)
        log_metric(tora, "val_precision", prec, epoch)
        log_metric(tora, "val_recall", rec, epoch)
        print(
            f"Validation Loss={val_loss:.4f}, Acc={acc:.2f}%, "
            f"Prec={prec:.2f}%, Rec={rec:.2f}%"
        )

        ndcg_vals, p_at_k, r_at_k = [], [], []
        for user, items in user_preds.items():
            items.sort(key=lambda x: x[1], reverse=True)
            top = items[:k]
            tp = sum(1 for _, _, rate in top if rate == 1)
            if tp > 0:
                p_at_k.append(tp / len(top))
                r_at_k.append(tp / sum(1 for _, _, rate in items if rate == 1))
            try:
                y_true = np.array([rate for _, _, rate in items])
                y_score = np.array([score for _, score, _ in items])
                ndcg_vals.append(
                    ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)
                )
            except Exception:
                pass

        if p_at_k:
            avg_p = 100.0 * np.mean(p_at_k)
            log_metric(tora, f"val_precision@{k}", avg_p, epoch)
            print(f"Precision@{k}={avg_p:.2f}%")
        if r_at_k:
            avg_r = 100.0 * np.mean(r_at_k)
            log_metric(tora, f"val_recall@{k}", avg_r, epoch)
            print(f"Recall@{k}={avg_r:.2f}%")
        if ndcg_vals:
            avg_n = 100.0 * np.mean(ndcg_vals)
            log_metric(tora, f"val_ndcg@{k}", avg_n, epoch)
            print(f"NDCG@{k}={avg_n:.2f}%")

        return val_loss, acc, prec, rec

    print(f"Validation Loss={val_loss:.4f}")
    return (val_loss,)  # continuous rating


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Hyperparameters
    params = {
        "batch_size": 256,
        "epochs": 30,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "n_factors": 50,
        "dropout": 0.2,
        "negative_ratio": 1,
        "binary": True,
        "patience": 5,
        "k": 10,
    }

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    params["device"] = str(device)

    # Load data and create loaders
    ratings, movies, n_users, n_items = load_movielens_data()
    print("Creating negative samples...")
    combined = create_negative_samples(
        ratings, n_items, neg_ratio=params["negative_ratio"]
    )
    dataset = RecommenderDataset(combined, n_users, n_items, binary=params["binary"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"])

    # Initialize Tora experiment
    params.update(
        {
            "n_users": n_users,
            "n_items": n_items,
            "train_samples": train_size,
            "val_samples": val_size,
            "total_samples": len(dataset),
            "model": "MatrixFactorization",
        }
    )
    tora = Tora.create_experiment(
        name="MovieLens_Recommender",
        description="Matrix factorization for movie recommendations",
        hyperparams=params,
        tags=[
            "recommender-system",
            "collaborative-filtering",
            "matrix-factorization",
        ],
    )

    # Model, loss, optimizer
    model = MatrixFactorization(
        n_users,
        n_items,
        params["n_factors"],
        dropout=params["dropout"],
        binary=params["binary"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_metric(tora, "model_parameters", n_params, 0)
    criterion = nn.BCELoss() if params["binary"] else nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    # Training loop with early stopping
    best_loss = float("inf")
    counter = 0
    for epoch in range(1, params["epochs"] + 1):
        train_loss, train_acc = (
            train_epoch(
                model,
                device,
                train_loader,
                optimizer,
                criterion,
                epoch,
                tora,
                binary=params["binary"],
            )
            if params["binary"]
            else (
                train_epoch(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    criterion,
                    epoch,
                    tora,
                    binary=False,
                ),
                None,
            )
        )
        val_results = validate(
            model,
            device,
            val_loader,
            criterion,
            epoch,
            tora,
            binary=params["binary"],
            k=params["k"],
        )
        val_loss = val_results[0]

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_recommender_model.pt")
            print(f"Saved best model, val_loss={best_loss:.4f}")
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping {counter}/{params['patience']}")
            if counter >= params["patience"]:
                print(f"Stopping early at epoch {epoch}")
                break

    # Final evaluation
    print(f"Loading best model with loss={best_loss:.4f}")
    model.load_state_dict(torch.load("best_recommender_model.pt"))
    final = validate(
        model,
        device,
        val_loader,
        criterion,
        params["epochs"],
        tora,
        binary=params["binary"],
        k=params["k"],
    )
    if params["binary"]:
        log_metric(tora, "final_loss", final[0], params["epochs"])
        log_metric(tora, "final_accuracy", final[1], params["epochs"])
        log_metric(tora, "final_precision", final[2], params["epochs"])
        log_metric(tora, "final_recall", final[3], params["epochs"])
    else:
        log_metric(tora, "final_loss", final[0], params["epochs"])

    tora.shutdown()
