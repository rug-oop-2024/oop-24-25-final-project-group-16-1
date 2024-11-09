from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import (
    LinearRegressionModel
)
import numpy as np
from copy import deepcopy


class MultipleLinearRegression(Model):
    """
    A multiple linear regression model implementation that uses
    the LinearRegressionModel as the base model for gradient
    descent optimization.

    Attributes:
        name (str): The name identifier for the model.
        type (str): Type of model, default is "regression".
        learning_rate (float): The step size for gradient descent.
        num_iterations (int): The number of gradient descent iterations.
        parameters (dict): The trained weights after model fitting.
    """
    def __init__(
        self,
        name: str = "Multiple Linear Regression",
        type: str = "regression",
        learning_rate: float = 0.01,
        num_iterations: int = 10,
    ) -> None:
        """
        Initializes the MultipleLinearRegression model using
        LinearRegressionModel as the base model.

        Args:
            name (str): The name of the model instance.
            type (str): Type of the model, default is "regression".
            learning_rate (float): Learning rate for gradient descent.
            num_iterations (int): Number of iterations for gradient descent.
        """
        super().__init__(name=name, type=type)
        self._parameters = {}
        self._model = LinearRegressionModel(learning_rate, num_iterations)

    @property
    def parameters(self) -> dict:
        """
        Retrieves the model parameters, such as weights after training.

        Returns:
            dict: A copy of the model's parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """
        Sets the model's parameters, ensuring they
        are provided as a dictionary.

        Args:
            value (dict): The parameters to set.

        Raises:
            ValueError: If the provided parameters are not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the provided input features and target
        labels using gradient descent.

        Args:
            x (np.ndarray): Input feature matrix with shape
            (n_samples, n_features).
            y (np.ndarray): Target values with shape (n_samples,).

        Notes:
            Updates the model parameters with weights after fitting.
        """
        self._model.fit(x, y)
        self._parameters["weights"] = self._model.weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts target values based on input features using the trained model.

        Args:
            x (np.ndarray): Input feature matrix with shape
            (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values with shape (n_samples,).

        Raises:
            ValueError: If the model has not been trained (weights are None).
        """
        return self._model.predict(x)
