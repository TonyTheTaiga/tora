import os


import time


import matplotlib.pyplot as plt


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import torch.optim as optim


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from sklearn.preprocessing import MinMaxScaler


from tora import Tora


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


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences

        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def create_sequences(data, seq_length, horizon):
    xs, ys = [], []

    for i in range(len(data) - seq_length - horizon + 1):
        xs.append(data[i : i + seq_length])

        ys.append(data[i + seq_length : i + seq_length + horizon])

    return np.array(xs), np.array(ys)


class LSTMForecaster(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2
    ):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))

        predictions = self.fc(lstm_out[:, -1, :])

        return predictions


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, tora):
    model.train()

    running_loss = 0.0

    total_samples = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        try:
            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * data.size(0)

            total_samples += data.size(0)

            if batch_idx % 20 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")

    epoch_loss = running_loss / max(total_samples, 1)

    epoch_time = time.time() - start_time

    log_metric(tora, "train_loss", epoch_loss, epoch)

    log_metric(tora, "epoch_time", epoch_time, epoch)

    return epoch_loss


def validate(model, device, val_loader, criterion, scaler, epoch, tora, split="val"):
    model.eval()

    running_loss = 0.0

    total_samples = 0

    all_targets = []

    all_predictions = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)

            total_samples += data.size(0)

            scaled_preds = output.cpu().numpy()

            scaled_targets = target.cpu().numpy()

            if len(scaled_preds.shape) > 2:
                scaled_preds = scaled_preds.reshape(-1, scaled_preds.shape[-1])

                scaled_targets = scaled_targets.reshape(-1, scaled_targets.shape[-1])

            if scaler:
                dummy = np.zeros((scaled_preds.shape[0], scaler.scale_.shape[0]))

                dummy[:, -1] = scaled_preds[:, 0]

                preds_orig = scaler.inverse_transform(dummy)[:, -1]

                dummy = np.zeros((scaled_targets.shape[0], scaler.scale_.shape[0]))

                dummy[:, -1] = scaled_targets[:, 0]

                targets_orig = scaler.inverse_transform(dummy)[:, -1]

            else:
                preds_orig = scaled_preds[:, 0]

                targets_orig = scaled_targets[:, 0]

            all_predictions.extend(preds_orig)

            all_targets.extend(targets_orig)

    val_loss = running_loss / max(total_samples, 1)

    try:
        mse = mean_squared_error(all_targets, all_predictions)

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(all_targets, all_predictions)

        r2 = r2_score(all_targets, all_predictions)

    except:
        mse, rmse, mae, r2 = 0, 0, 0, 0

    prefix = "val" if split == "val" else "test"

    log_metric(tora, f"{prefix}_loss", val_loss, epoch)

    log_metric(tora, f"{prefix}_mse", mse, epoch)

    log_metric(tora, f"{prefix}_rmse", rmse, epoch)

    log_metric(tora, f"{prefix}_mae", mae, epoch)

    log_metric(tora, f"{prefix}_r2", r2, epoch)

    print(
        f"\n{split.capitalize()} set: Loss: {val_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}\n"
    )

    return val_loss, mse, rmse, mae, r2


def generate_synthetic_data():
    np.random.seed(42)

    time_steps = 1000

    t = np.arange(time_steps)

    trend = 0.01 * t

    season1 = 2 * np.sin(2 * np.pi * t / 50)

    season2 = 1 * np.sin(2 * np.pi * t / 100)

    seasonality = season1 + season2

    noise = 0.5 * np.random.randn(time_steps)

    data = trend + seasonality + noise

    features = np.column_stack(
        [
            data,
            np.sin(2 * np.pi * t / 50),
            np.cos(2 * np.pi * t / 50),
            np.sin(2 * np.pi * t / 100),
            np.cos(2 * np.pi * t / 100),
            t / 1000.0,
        ]
    )

    df = pd.DataFrame(
        features, columns=["value", "sin_50", "cos_50", "sin_100", "cos_100", "time"]
    )

    return df


if __name__ == "__main__":
    hyperparams = {
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "seq_length": 24,
        "horizon": 12,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-6,
        "early_stopping": 10,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    hyperparams["device"] = str(device)

    try:
        df = pd.read_csv("data/time_series_data.csv")

        print("Loaded real world dataset")

    except:
        print("Generating synthetic time series data")

        df = generate_synthetic_data()

    data = df.values

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(
        data_scaled,
        seq_length=hyperparams["seq_length"],
        horizon=hyperparams["horizon"],
    )

    dataset = TimeSeriesDataset(X, y)

    train_size = int(0.7 * len(dataset))

    val_size = int(0.15 * len(dataset))

    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])

    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"])

    hyperparams.update(
        {
            "input_size": X.shape[2],
            "output_size": y.shape[1],
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
        }
    )

    model = LSTMForecaster(
        input_size=hyperparams["input_size"],
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_layers"],
        output_size=hyperparams["output_size"],
        dropout_rate=hyperparams["dropout_rate"],
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hyperparams["model_parameters"] = model_params

    tora = Tora.create_experiment(
        name="TimeSeries_LSTM",
        description="LSTM model for time series forecasting with tracked metrics",
        hyperparams=hyperparams,
        tags=["time-series", "forecasting", "lstm", "regression"],
    )

    criterion = nn.MSELoss()

    if hyperparams["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )

    else:
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )

    if hyperparams["scheduler"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=hyperparams["factor"],
            patience=hyperparams["patience"],
            min_lr=hyperparams["min_lr"],
            verbose=True,
        )

    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")

    best_model_path = "best_time_series_model.pt"

    early_stopping_counter = 0

    for epoch in range(1, hyperparams["epochs"] + 1):
        log_metric(tora, "learning_rate", optimizer.param_groups[0]["lr"], epoch)

        train_loss = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, tora
        )

        val_loss, val_mse, val_rmse, val_mae, val_r2 = validate(
            model, device, val_loader, criterion, scaler, epoch, tora, split="val"
        )

        if hyperparams["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)

        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), best_model_path)

            print(f"Best model saved with validation loss: {best_val_loss:.6f}")

            early_stopping_counter = 0

        else:
            early_stopping_counter += 1

        if early_stopping_counter >= hyperparams["early_stopping"]:
            print(f"Early stopping triggered after {epoch} epochs")

            break

    print(f"Loading best model with validation loss: {best_val_loss:.6f}")

    model.load_state_dict(torch.load(best_model_path))

    test_loss, test_mse, test_rmse, test_mae, test_r2 = validate(
        model,
        device,
        test_loader,
        criterion,
        scaler,
        hyperparams["epochs"],
        tora,
        split="test",
    )

    log_metric(tora, "final_test_loss", test_loss, hyperparams["epochs"])

    log_metric(tora, "final_test_mse", test_mse, hyperparams["epochs"])

    log_metric(tora, "final_test_rmse", test_rmse, hyperparams["epochs"])

    log_metric(tora, "final_test_mae", test_mae, hyperparams["epochs"])

    log_metric(tora, "final_test_r2", test_r2, hyperparams["epochs"])

    model.eval()

    with torch.no_grad():
        sample_data, sample_target = next(iter(test_loader))

        sample_data = sample_data.to(device)

        sample_forecast = model(sample_data)

        sample_forecast = sample_forecast.cpu().numpy()

        sample_target = sample_target.cpu().numpy()

        forecast = sample_forecast[0]

        actual = sample_target[0]

        if scaler:
            dummy_forecast = np.zeros((forecast.shape[0], scaler.scale_.shape[0]))

            dummy_actual = np.zeros((actual.shape[0], scaler.scale_.shape[0]))

            dummy_forecast[:, -1] = forecast

            dummy_actual[:, -1] = actual

            forecast = scaler.inverse_transform(dummy_forecast)[:, -1]

            actual = scaler.inverse_transform(dummy_actual)[:, -1]

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(range(len(actual)), actual, label="Actual")

    plt.plot(range(len(forecast)), forecast, label="Forecast", linestyle="--")

    plt.title("Time Series Forecast vs Actual")

    plt.xlabel("Time Steps")

    plt.ylabel("Value")

    plt.legend()

    plt.grid(True)

    plt.savefig("results/time_series_forecast.png")

    tora.shutdown()
