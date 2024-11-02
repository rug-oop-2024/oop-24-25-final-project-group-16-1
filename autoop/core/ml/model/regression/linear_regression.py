import numpy as np
from typing import Optional, Dict
from copy import deepcopy
from autoop.core.ml.model.model import Model

class LinearRegressionModel(Model):
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 10) -> None:
        super().__init__(name="LinearRegressionModel")
        self._learning_rate: float = learning_rate
        self._num_iterations: int = num_iterations
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float) -> None:
        self._learning_rate = rate

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, iterations: int) -> None:
        self._num_iterations = iterations

    @property
    def weights(self) -> Optional[np.ndarray]:
        return deepcopy(self._weights)

    @property
    def bias(self) -> float:
        return self._bias

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Reshape y to be 1-dimensional
        y = y.flatten()  # Ensure y has shape (num_samples,)
        
        num_samples, num_features = X.shape
        self._weights = np.zeros(num_features)
        self._bias = 0.0

        for _ in range(self._num_iterations):
            y_pred = np.dot(X, self._weights) + self._bias

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Ensure dw has the correct shape to match self._weights
            assert dw.shape == self._weights.shape, f"dw shape {dw.shape} does not match weights shape {self._weights.shape}"

            # Update weights and bias
            self._weights -= self._learning_rate * dw
            self._bias -= self._learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self._weights) + self._bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def save(self) -> None:
        super().save()

    def load(self, name: str) -> None:
        super().load(name)
