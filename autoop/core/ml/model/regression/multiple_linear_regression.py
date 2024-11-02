from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import LinearRegressionModel
import numpy as np
from copy import deepcopy

class MultipleLinearRegression(Model):
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        """
        Initializes the Multiple Linear Regression model.
        Args:
            learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.01.
            num_iterations (int, optional): The number of iterations for training the model. Defaults to 1000.
        """
        super().__init__()
        self._parameters = {}
        self._model = LinearRegressionModel(learning_rate, num_iterations)

    @property
    def parameters(self) -> dict:
        """Get a copy of the model parameters."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """Set the model parameters."""
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the observations and ground truth values.
        Args:
            observations (np.ndarray): The input feature matrix (shape: [num_samples, num_features]).
            ground_truth (np.ndarray): The target output values (shape: [num_samples]).
        """
        self._model.train(observations, ground_truth)
        self._parameters['weights'] = self._model.weights

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Multiple Linear Regression model.
        Args:
            observations (np.ndarray): The input feature matrix (shape: [num_samples, num_features]).
        Returns:
            np.ndarray: The predicted values (shape: [num_samples]).
        """
        return self._model.predict(observations)
    
    def save(self) -> None:
        """Save model state."""
        super().save()

    def load(self, name: str) -> None:
        """Load model state from the specified name."""
        super().load(name)
