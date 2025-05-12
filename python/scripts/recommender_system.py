import os


import random


import time


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.optim as optim


from sklearn.metrics import ndcg_score, precision_score, recall_score


from tora import Tora as Tora


from torch.utils.data import DataLoader, Dataset, random_split


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


def download_movielens_dataset():
    if not os.path.exists("data/ml-100k"):
        print("Downloading MovieLens 100K dataset...")

        import urllib.request

        import zipfile

        os.makedirs("data", exist_ok=True)

        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

        zip_path = "data/ml-100k.zip"

        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data")

        os.remove(zip_path)

        print("Download complete!")

    else:
        print("MovieLens 100K dataset already exists. Skipping download.")


def load_movielens_data():
    download_movielens_dataset()

    ratings_path = "data/ml-100k/u.data"

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    movies_path = "data/ml-100k/u.item"

    movies = pd.read_csv(
        movies_path,
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

    threshold = 4.0

    ratings["rating_binary"] = (ratings["rating"] >= threshold).astype(int)

    n_users = ratings["user_id"].nunique()

    n_items = ratings["item_id"].nunique()

    print("Loaded MovieLens 100K dataset:")

    print(f"  Users: {n_users}")

    print(f"  Items: {n_items}")

    print(f"  Ratings: {len(ratings)}")

    return ratings, movies, n_users, n_items


class RecommenderDataset(Dataset):
    def __init__(self, ratings_df, n_users, n_items, binary=True):
        self.ratings_df = ratings_df

        self.n_users = n_users

        self.n_items = n_items

        self.binary = binary

        self.rating_col = "rating_binary" if binary else "rating"

        self.user_ids = ratings_df["user_id"].values - 1

        self.item_ids = ratings_df["item_id"].values - 1

        self.ratings = ratings_df[self.rating_col].values

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]

        item_id = self.item_ids[idx]

        rating = self.ratings[idx]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "rating": torch.tensor(
                rating, dtype=torch.float if not self.binary else torch.long
            ),
        }


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors, dropout=0.2, binary=True):
        super(MatrixFactorization, self).__init__()

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
        user_embed = self.user_embedding(user_id)

        item_embed = self.item_embedding(item_id)

        user_embed = self.dropout(user_embed)

        item_embed = self.dropout(item_embed)

        user_b = self.user_bias(user_id).squeeze()

        item_b = self.item_bias(item_id).squeeze()

        dot_product = (user_embed * item_embed).sum(dim=1)

        prediction = dot_product + user_b + item_b + self.global_bias

        if self.binary:
            prediction = self.sigmoid(prediction)

        return prediction


def create_negative_samples(ratings_df, n_items, neg_ratio=1):
    user_item_set = set(zip(ratings_df["user_id"], ratings_df["item_id"]))

    negative_samples = []

    for user_id in ratings_df["user_id"].unique():
        interacted_items = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"])

        candidates = list(set(range(1, n_items + 1)) - interacted_items)

        if not candidates:
            continue

        n_neg = min(int(len(interacted_items) * neg_ratio), len(candidates))

        neg_items = random.sample(candidates, n_neg)

        for item_id in neg_items:
            negative_samples.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 0,
                    "timestamp": 0,
                    "rating_binary": 0,
                }
            )

    neg_df = pd.DataFrame(negative_samples)

    combined_df = pd.concat([ratings_df, neg_df], ignore_index=True)

    return combined_df


def train_epoch(
    model, device, train_loader, optimizer, criterion, epoch, tora, binary=True
):
    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        user_ids = batch["user_id"].to(device)

        item_ids = batch["item_id"].to(device)

        ratings = batch["rating"].to(device)

        optimizer.zero_grad()

        try:
            outputs = model(user_ids, item_ids)

            if binary:
                loss = criterion(outputs, ratings.float())

                predicted = (outputs >= 0.5).float()

                correct += (predicted == ratings).sum().item()

            else:
                loss = criterion(outputs, ratings)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * user_ids.size(0)

            total += user_ids.size(0)

            if batch_idx % 50 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(user_ids)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")

    epoch_loss = running_loss / max(total, 1)

    epoch_time = time.time() - start_time

    log_metric(tora, "train_loss", epoch_loss, epoch)

    log_metric(tora, "epoch_time", epoch_time, epoch)

    if binary:
        accuracy = 100.0 * correct / max(total, 1)

        log_metric(tora, "train_accuracy", accuracy, epoch)

        return epoch_loss, accuracy

    else:
        return epoch_loss


def validate(model, device, val_loader, criterion, epoch, tora, binary=True, k=10):
    model.eval()

    running_loss = 0.0

    total = 0

    all_targets = []

    all_predictions = []

    all_prediction_scores = []

    user_predictions = {}

    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch["user_id"].to(device)

            item_ids = batch["item_id"].to(device)

            ratings = batch["rating"].to(device)

            outputs = model(user_ids, item_ids)

            if binary:
                loss = criterion(outputs, ratings.float())

                all_targets.extend(ratings.cpu().numpy())

                all_prediction_scores.extend(outputs.cpu().numpy())

                all_predictions.extend((outputs >= 0.5).float().cpu().numpy())

            else:
                loss = criterion(outputs, ratings)

            running_loss += loss.item() * user_ids.size(0)

            total += user_ids.size(0)

            for i in range(len(user_ids)):
                user_id = user_ids[i].item()

                item_id = item_ids[i].item()

                pred = outputs[i].item()

                true_rating = ratings[i].item()

                if user_id not in user_predictions:
                    user_predictions[user_id] = []

                user_predictions[user_id].append((item_id, pred, true_rating))

    val_loss = running_loss / max(total, 1)

    if binary and all_targets:
        try:
            accuracy = (np.array(all_predictions) == np.array(all_targets)).mean() * 100

            precision = (
                precision_score(all_targets, all_predictions, zero_division=0) * 100
            )

            recall = recall_score(all_targets, all_predictions, zero_division=0) * 100

            log_metric(tora, "val_loss", val_loss, epoch)

            log_metric(tora, "val_accuracy", accuracy, epoch)

            log_metric(tora, "val_precision", precision, epoch)

            log_metric(tora, "val_recall", recall, epoch)

            print(
                f"\nValidation: Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, "
                f"Precision: {precision:.2f}%, Recall: {recall:.2f}%\n"
            )

            ndcg_values = []

            precision_at_k_values = []

            recall_at_k_values = []

            for user_id, items in user_predictions.items():
                if not items:
                    continue

                items.sort(key=lambda x: x[1], reverse=True)

                top_k_items = items[:k]

                true_positives = sum(1 for _, _, rating in top_k_items if rating == 1)

                if true_positives > 0:
                    precision_at_k = true_positives / len(top_k_items)

                    recall_at_k = true_positives / sum(
                        1 for _, _, rating in items if rating == 1
                    )

                    precision_at_k_values.append(precision_at_k)

                    recall_at_k_values.append(recall_at_k)

                try:
                    y_pred = np.array([pred for _, pred, _ in items])

                    y_true = np.array([rating for _, _, rating in items])

                    ndcg = ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)

                    ndcg_values.append(ndcg)

                except:
                    pass

            if precision_at_k_values:
                avg_precision_at_k = np.mean(precision_at_k_values) * 100

                log_metric(tora, f"val_precision@{k}", avg_precision_at_k, epoch)

                print(f"Precision@{k}: {avg_precision_at_k:.2f}%")

            if recall_at_k_values:
                avg_recall_at_k = np.mean(recall_at_k_values) * 100

                log_metric(tora, f"val_recall@{k}", avg_recall_at_k, epoch)

                print(f"Recall@{k}: {avg_recall_at_k:.2f}%")

            if ndcg_values:
                avg_ndcg = np.mean(ndcg_values) * 100

                log_metric(tora, f"val_ndcg@{k}", avg_ndcg, epoch)

                print(f"NDCG@{k}: {avg_ndcg:.2f}%")

            return val_loss, accuracy, precision, recall

        except Exception as e:
            print(f"Error calculating validation metrics: {str(e)}")

            return val_loss, 0, 0, 0

    else:
        log_metric(tora, "val_loss", val_loss, epoch)

        print(f"\nValidation: Loss: {val_loss:.4f}\n")

        return val_loss


if __name__ == "__main__":
    np.random.seed(42)

    torch.manual_seed(42)

    random.seed(42)

    hyperparams = {
        "batch_size": 256,
        "epochs": 30,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "n_factors": 50,
        "dropout": 0.2,
        "negative_ratio": 1,
        "binary": True,
        "patience": 5,
        "k": 10,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    hyperparams["device"] = str(device)

    ratings, movies, n_users, n_items = load_movielens_data()

    print("Creating negative samples...")

    combined_ratings = create_negative_samples(
        ratings, n_items, neg_ratio=hyperparams["negative_ratio"]
    )

    dataset = RecommenderDataset(
        combined_ratings, n_users, n_items, binary=hyperparams["binary"]
    )

    train_size = int(0.8 * len(dataset))

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])

    hyperparams.update(
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
        description="Matrix factorization model for movie recommendations",
        hyperparams=hyperparams,
        tags=["recommender-system", "collaborative-filtering", "matrix-factorization"],
    )

    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        n_factors=hyperparams["n_factors"],
        dropout=hyperparams["dropout"],
        binary=hyperparams["binary"],
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hyperparams["model_parameters"] = model_params

    log_metric(tora, "model_parameters", model_params, 0)

    if hyperparams["binary"]:
        criterion = nn.BCELoss()

    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )

    patience = hyperparams["patience"]

    best_val_loss = float("inf")

    best_model_path = "best_recommender_model.pt"

    early_stopping_counter = 0

    for epoch in range(1, hyperparams["epochs"] + 1):
        if hyperparams["binary"]:
            train_loss, train_acc = train_epoch(
                model,
                device,
                train_loader,
                optimizer,
                criterion,
                epoch,
                tora,
                binary=True,
            )

        else:
            train_loss = train_epoch(
                model,
                device,
                train_loader,
                optimizer,
                criterion,
                epoch,
                tora,
                binary=False,
            )

        val_results = validate(
            model,
            device,
            val_loader,
            criterion,
            epoch,
            tora,
            binary=hyperparams["binary"],
            k=hyperparams["k"],
        )

        val_loss = val_results[0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), best_model_path)

            print(f"Best model saved with validation loss: {best_val_loss:.6f}")

            early_stopping_counter = 0

        else:
            early_stopping_counter += 1

            print(f"EarlyStopping counter: {early_stopping_counter} out of {patience}")

            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")

                break

    print(f"Loading best model with validation loss: {best_val_loss:.6f}")

    model.load_state_dict(torch.load(best_model_path))

    print("Running final evaluation...")

    final_results = validate(
        model,
        device,
        val_loader,
        criterion,
        hyperparams["epochs"],
        tora,
        binary=hyperparams["binary"],
        k=hyperparams["k"],
    )

    if hyperparams["binary"]:
        final_loss, final_acc, final_precision, final_recall = final_results

        log_metric(tora, "final_loss", final_loss, hyperparams["epochs"])

        log_metric(tora, "final_accuracy", final_acc, hyperparams["epochs"])

        log_metric(tora, "final_precision", final_precision, hyperparams["epochs"])

        log_metric(tora, "final_recall", final_recall, hyperparams["epochs"])

    else:
        final_loss = final_results[0]

        log_metric(tora, "final_loss", final_loss, hyperparams["epochs"])

    tora.shutdown()
