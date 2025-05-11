import numpy as np


np.random.seed(42)


class LogisticRegression:
    def __init__(self, num_features, learning_rate) -> None:
        self.num_features = num_features

        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(num_features,))

        self.learning_rate = learning_rate

    def __call__(self, x):
        return self.forward(x)

    def forward(self, X):
        z = np.dot(X, self.weights)

        p = self.sigmoid(z)

        return p

    def loss(self, predictions, y):
        return -sum(y * np.log(predictions) - (1 - y) * np.log(1 - predictions))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, prediction, truth):
        grad = np.dot(X.T, (truth - prediction))

        return grad

    def backward(self, grad):
        self.weights += self.learning_rate * grad

        return self.weights


def get_batches(X, y, batch_size):
    random_indexes = np.random.permutation(X.shape[0])

    for idx in range(0, X.shape[0], batch_size):
        yield (
            X[random_indexes[idx : idx + batch_size]],
            y[random_indexes[idx : idx + batch_size]],
        )


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    from sklearn.preprocessing import StandardScaler

    from tora import Tora

    data = load_breast_cancer()

    X, y = data.data, data.target

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    learning_rate = 2e-4

    model = LogisticRegression(num_features=X.shape[1], learning_rate=learning_rate)

    epochs = 30

    tora = Tora.create_experiment(
        name="Cancer",
        description="Cancer dataset",
        hyperparams={"epochs": epochs, "lr": learning_rate},
        tags=["cancer"],
    )

    for epoch in range(epochs):
        predictions = []

        truths = []

        total_loss = 0

        samples = 0

        for bX, by in get_batches(X, y, batch_size=8):
            prediction = model(bX)

            loss = model.loss(prediction, by)

            total_loss += loss * bX.shape[0]

            samples += bX.shape[0]

            grad = model.gradient(bX, prediction, by)

            model.backward(grad)

            prediction_labels = (prediction >= 0.5).astype(int)

            predictions.extend(prediction_labels)

            truths.extend(by)

        if epoch <= 10:
            model.learning_rate *= 0.95

        else:
            model.learning_rate *= 0.85

        acc = (np.array(predictions) == np.array(truths)).mean()

        epoch_loss = total_loss / samples

        tora.log(name="epoch_loss", value=epoch_loss.item(), step=epoch)

        tora.log(name="accuracy", value=acc.item(), step=epoch)

    tora.shutdown()
