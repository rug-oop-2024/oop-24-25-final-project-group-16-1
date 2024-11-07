from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import LinearRegressionModel
import numpy as np
from copy import deepcopy


class MultipleLinearRegression(Model):
    def __init__(
        self,
        name: str = "Multiple Linear Regression",
        type: str = "regression",
        learning_rate: float = 0.01,
        num_iterations: int = 10,
    ) -> None:
        super().__init__(name=name, type=type)
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model with input features x and target labels y.
        """
        self._model.fit(x, y)
        self._parameters["weights"] = self._model.weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict labels or values given input features x.
        """
        return self._model.predict(x)
