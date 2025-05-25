import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tora import Tora

np.random.seed(42)


def safe_value(value):
    """
    Convert various types to safe numeric types for logging.
    """
    try:
        return float(value)
    except Exception:
        return None


class LogisticRegression:
    """
    Simple logistic regression using NumPy.
    Attributes:
        num_features: number of input features
        weights: model weights
        learning_rate: step size for gradient updates
    """

    def __init__(self, num_features: int, learning_rate: float) -> None:
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-0.5, 0.5, size=(num_features,))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation of the linear model.
        """
        z = X.dot(self.weights)
        return self.sigmoid(z)

    def loss(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        """
        # Clip predictions for numerical stability
        eps = 1e-9
        p = np.clip(predictions, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def gradient(
        self, X: np.ndarray, predictions: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. weights.
        """
        return X.T.dot(y - predictions) / X.shape[0]

    def step(self, grad: np.ndarray) -> None:
        """
        Update weights using the gradient.
        """
        self.weights += self.learning_rate * grad


def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """
    Yield mini-batches for training.
    """
    indices = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def main():
    # Load and preprocess data
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Hyperparameters
    learning_rate = 2e-4
    epochs = 30
    batch_size = 8

    # Initialize experiment
    tora = Tora.create_experiment(
        name="Cancer_Classification",
        description="Logistic regression on breast cancer dataset",
        hyperparams={"epochs": epochs, "lr": learning_rate},
        tags=["cancer", "logistic_regression"],
        workspace_id="84679cc3-dae5-40c7-91f3-10de46123765",
    )

    model = LogisticRegression(num_features=X.shape[1], learning_rate=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        for batch_X, batch_y in get_batches(X, y, batch_size):
            preds = model(batch_X)
            loss = model.loss(preds, batch_y)
            grad = model.gradient(batch_X, preds, batch_y)
            model.step(grad)

            epoch_loss += loss * len(batch_X)
            all_preds.extend((preds >= 0.5).astype(int))
            all_labels.extend(batch_y)

        # Adjust learning rate
        if epoch <= 10:
            model.learning_rate *= 0.95
        else:
            model.learning_rate *= 0.85

        avg_loss = epoch_loss / len(X)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        # Log metrics
        tora.log(name="epoch_loss", value=safe_value(avg_loss), step=epoch)
        tora.log(name="accuracy", value=safe_value(accuracy), step=epoch)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

    tora.shutdown()


if __name__ == "__main__":
    main()
