from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from typing import Dict, Any


class Lasso(Model):
    def __init__(self, name: str, alpha: float = 1.0) -> None:
        """
        Initializes the Lasso model with a specified name and regularization strength.
        Args:
            name (str): The name identifier for the model instance.
            alpha (float): The regularization strength, which controls the amount of
                           penalty applied to the model coefficients; must be positive.
        """
        super().__init__(name=name)
        self._alpha: float = alpha
        self._model: SklearnLasso = SklearnLasso(alpha=self._alpha)
        self._parameters: Dict[str, Any] = {}

    @property
    def alpha(self) -> float:
        """
        Retrieves the regularization strength (alpha) of the model.
        Returns:
            float: The current value of alpha, which controls regularization.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """
        Sets a new value for the regularization strength (alpha) and updates the model.
        Args:
            value (float): The new alpha value, which must be positive.
        Raises:
            ValueError: If the provided alpha value is not positive.
        """
        if value <= 0:
            raise ValueError("Alpha should be positive.")
        self._alpha = value
        self._model = SklearnLasso(alpha=self._alpha)

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Retrieves the model parameters, including weights and intercept, if fitted.
        Returns:
            Dict[str, Any]: A dictionary containing model parameters.
                            Keys include 'weights' and 'intercept' after fitting.
        """
        return self._parameters

    def fit(self, X: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the Lasso model on provided features and target values.
        Args:
            X (np.ndarray): The input feature matrix with shape (n_samples, n_features).
            ground_truth (np.ndarray): The target values for training with shape (n_samples,).
        Notes:
            Updates the `_parameters` dictionary to store the model's weights and intercept
            after fitting.
        """
        self._model.fit(X, ground_truth)
        self._parameters["weights"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values based on the input features using the trained model.
        Args:
            X (np.ndarray): The input feature matrix with shape (n_samples, n_features).
        Returns:
            np.ndarray: The predicted target values with shape (n_samples,).
        """
        return self._model.predict(X)
