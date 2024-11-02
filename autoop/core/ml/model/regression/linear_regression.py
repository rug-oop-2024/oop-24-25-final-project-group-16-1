import numpy as np
from typing import Optional
from copy import deepcopy
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        """
        Initializes the Linear Regression model with specified learning rate and number of iterations.
        Args:
            learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.
            num_iterations (int, optional): The number of iterations for training the model. Defaults to 1000.
        """
        self._learning_rate: float = learning_rate
        self._num_iterations: int = num_iterations
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float) -> None:
        """Set the learning rate."""
        self._learning_rate = rate

    @property
    def num_iterations(self) -> int:
        """Get the number of training iterations."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, iterations: int) -> None:
        """Set the number of training iterations."""
        self._num_iterations = iterations

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get the model weights."""
        return deepcopy(self._weights)

    @property
    def bias(self) -> float:
        """Get the model bias."""
        return self._bias

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Linear Regression model using the provided training data.
        Args:
            X (np.ndarray): The input feature matrix (shape: [num_samples, num_features]).
            y (np.ndarray): The target output values (shape: [num_samples]).
        """
        num_samples, num_features = X.shape
        self._weights = np.zeros(num_features)
        self._bias = 0.0
        
        for _ in range(self._num_iterations):
            y_pred = np.dot(X, self._weights) + self._bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            self._weights -= self._learning_rate * dw
            self._bias -= self._learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Linear Regression model.
        Args:
            X (np.ndarray): The input feature matrix (shape: [num_samples, num_features]).
        Returns:
            np.ndarray: The predicted values (shape: [num_samples]).
        """
        return np.dot(X, self._weights) + self._bias
    
    
    def save(self) -> None:
        """Save model state."""
        super().save()

    def load(self, name: str) -> None:
        """Load model state from the specified name."""
        super().load(name)
