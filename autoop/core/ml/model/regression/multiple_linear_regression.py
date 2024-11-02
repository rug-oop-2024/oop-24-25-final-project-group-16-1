from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import LinearRegressionModel
import numpy as np
from copy import deepcopy

class MultipleLinearRegression(Model):
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 10):
        super().__init__(name="MultipleLinearRegression")
        self._parameters = {}
        self._model = LinearRegressionModel(learning_rate, num_iterations)

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model with input features x and target labels y.
        """
        self.fit(x, y)  # Use the fit method to perform the training

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model with input features x and target labels y.
        """
        self._model.train(x, y)
        self._parameters['weights'] = self._model.weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict labels or values given input features x.
        """
        return self._model.predict(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model performance on test data using Mean Squared Error.
        """
        predictions = self.predict(x)
        return np.mean((predictions - y) ** 2)  # Mean Squared Error

    def save(self) -> None:
        super().save()

    def load(self, name: str) -> None:
        super().load(name)

