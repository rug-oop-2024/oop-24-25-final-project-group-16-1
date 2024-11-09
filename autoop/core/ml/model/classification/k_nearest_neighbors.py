from autoop.core.ml.model.model import Model
import numpy as np
import pandas as pd
from typing import Any
from copy import deepcopy


class KNearestNeighbors(Model):
    def __init__(
        self,
        k: int = 3,
        name: str = "K-Nearest Neighbors",
        type: str = "classification",
    ) -> None:
        """
        Initializes the KNearestNeighbors model with
        a specified number of neighbors.

        Args:
            k (int): The number of nearest neighbors to consider.
            name (str): The name of the model instance.
            type (str): The type of artifact, default is "classification".
        """
        super().__init__(name=name, type=type)
        self.k = k
        self.observations = None
        self.ground_truth = None
        self._parameters = {}

    @property
    def parameters(self) -> dict:
        """
        Retrieves the parameters of the model.

        Returns:
            dict: A dictionary of model parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """
        Sets the parameters of the model.

        Args:
            value (dict): A dictionary of parameters.

        Raises:
            ValueError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the model with the provided
        observations and ground truth labels.

        Args:
            observations (np.ndarray): The input feature
            matrix with shape (n_samples, n_features).
            ground_truth (np.ndarray): The true labels corresponding
            to the observations.
        """
        self.observations = observations
        self.ground_truth = ground_truth
        self._parameters = {
            "observations": observations, "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the given observations.

        Args:
            observations (np.ndarray): The input feature matrix to predict,
            with shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted labels as a 1D array.
        """
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> Any:
        """
        Predicts the label for a single observation by finding the most common
        label among the k nearest neighbors.

        Args:
            observation (np.ndarray): A single input feature array with shape
            (n_features,).

        Returns:
            Any: The predicted label for the observation.
        """
        module = observation - self._parameters["observations"]
        distances = np.linalg.norm(module, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [
            self._parameters["ground_truth"][i] for i in k_indices
        ]
        most_common = pd.Series(k_nearest_labels).value_counts()
        return most_common.index[0]
