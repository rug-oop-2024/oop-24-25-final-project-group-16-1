from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from autoop.core.ml.metric import mean_squared_error
from typing import Dict, Any
from copy import deepcopy


class Lasso(Model):
    def __init__(self, name: str, alpha: float = 1.0) -> None:
        """
        Initializes the Lasso model with a given name and regularization strength.
        Args:
            name (str): The name of the model.
            alpha (float): The regularization strength; must be positive.
        """
        super().__init__(name=name)
        self._alpha: float = alpha
        self._model: SklearnLasso = SklearnLasso(alpha=self._alpha)
        self._parameters: Dict[str, Any] = {}

    @property
    def alpha(self) -> float:
        """Get the regularization strength."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """
        Set the regularization strength.
        Args:
            value (float): The new value for alpha; must be positive.
        Raises:
            ValueError: If the provided value is not positive.
        """
        if value <= 0:
            raise ValueError("Alpha should be positive.")
        self._alpha = value
        self._model = SklearnLasso(alpha=self._alpha)

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get a copy of the model parameters."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: Dict[str, Any]) -> None:
        """
        Set the model parameters.
        Args:
            value (Dict[str, Any]): The parameters to set.
        Raises:
            ValueError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def train(self, X: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Lasso model with input features and target labels.
        Args:
            X (np.ndarray): Input feature matrix.
            ground_truth (np.ndarray): Target labels for the training data.
        """
        self._model.fit(X, ground_truth)
        self.parameters['weights'] = self._model.coef_
        self.parameters['intercept'] = self._model.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values given input features.
        Args:
            X (np.ndarray): Input feature matrix.
        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(X)

    def evaluate(self, X: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Evaluate the model performance using Mean Squared Error.
        Args:
            X (np.ndarray): Input feature matrix for evaluation.
            ground_truth (np.ndarray): True target labels.

        Returns:
            float: Mean Squared Error of the model predictions.
        """
        predictions = self.predict(X)
        return mean_squared_error(ground_truth, predictions)

    def save(self) -> None:
        """Save model state, including alpha and parameters."""
        super().save()

    def load(self, name: str) -> None:
        """Load model state from the specified name."""
        super().load(name)