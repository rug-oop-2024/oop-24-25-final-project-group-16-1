import numpy as np
from typing import Optional
from copy import deepcopy
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    """
    A basic linear regression model that uses gradient descent
    for training.

    Attributes:
        learning_rate (float): The step size for each gradient descent update.
        num_iterations (int): Number of iterations to run the gradient descent.
        weights (Optional[np.ndarray]): The model parameters (weights),
        updated after training.
    """
    def __init__(
        self,
        name: str = "Linear Regression",
        type: str = "regression",
        learning_rate: float = 0.01,
        num_iterations: int = 10,
    ) -> None:
        """
        Initializes the LinearRegressionModel with a determined
        learning rate and iteration count.

        Args:
            learning_rate (float): The learning rate for
            gradient descent updates.
            num_iterations (int): The number of
            iterations for gradient descent.
        """
        super().__init__(name=name, type=type)
        self._learning_rate: float = learning_rate
        self._num_iterations: int = num_iterations
        self._weights: Optional[np.ndarray] = None

    @property
    def learning_rate(self) -> float:
        """
        Returns the current learning rate.
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float) -> None:
        """
        Sets a new learning rate value where it ensures it is positive.

        Args:
            rate (float): The new learning rate value.

        Raises:
            ValueError: If the provided learning rate is not positive.
        """
        if rate <= 0:
            raise ValueError("Learning rate must be positive.")
        self._learning_rate = rate

    @property
    def num_iterations(self) -> int:
        """
        Returns the number of iterations for training.
        """
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, iterations: int) -> None:
        """
        Sets the number of training iterations,
        ensuring it is a positive integer.

        Args:
            iterations (int): The new iteration count.

        Raises:
            ValueError: If the iteration count is not positive.
        """
        if iterations <= 0:
            raise ValueError("Number of iterations must be positive.")
        self._num_iterations = iterations

    @property
    def weights(self) -> Optional[np.ndarray]:
        """
        Returns the model's learned weights.

        Returns:
            Optional[np.ndarray]: The weights if the model
            is trained, else None.

        Raises:
            ValueError: If accessed before the model is trained.
        """
        if self._weights is None:
            raise ValueError(
                "Model weights are uninitialized. Call fit to train the model."
            )
        return deepcopy(self._weights)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the linear regression model using gradient descent.

        Args:
            X (np.ndarray): Input feature matrix of shape
            (num_samples, num_features).
            y (np.ndarray): Target values of shape (num_samples,).

        Notes:
            Updates the model's weights using gradient descent.
        """
        y = y.flatten()
        num_samples, num_features = X.shape
        self._weights = np.zeros(num_features)

        for _ in range(self._num_iterations):
            y_pred = np.dot(X, self._weights)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))

            assert (
                dw.shape == self._weights.shape
            ), f"dw shape {dw.shape} not match weights {self._weights.shape}"

            self._weights -= self._learning_rate * dw

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given input features.

        Args:
            X (np.ndarray): Input feature matrix
            of shape (num_samples, num_features).

        Returns:
            np.ndarray: Predicted values of shape (num_samples,).

        Raises:
            ValueError: If the model has not
            been trained (i.e., weights are None).
        """
        if self._weights is None:
            raise ValueError(
                "Model is not trained yet. Call fit to train the model."
            )
        return np.dot(X, self._weights)
