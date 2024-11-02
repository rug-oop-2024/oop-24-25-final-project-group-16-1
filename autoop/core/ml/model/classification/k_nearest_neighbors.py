from autoop.core.ml.model.model import Model
import numpy as np
import pandas as pd
from typing import Any
from copy import deepcopy


class KNearestNeighbors(Model):
    def __init__(self, name: str, k: int = 3, artifact_type: str = "model") -> None:
        """Initializes the KNearestNeighbors model.
        Args:
            name (str): The name of the model.
            k (int): The number of nearest neighbors to consider.
            artifact_type (str): The type of artifact, defaults to "model".
        """
        super().__init__(name=name, artifact_type=artifact_type)
        self.k = k
        self.observations = None
        self.ground_truth = None
        self._parameters = {}

    @property
    def parameters(self) -> dict:
        """Gets the parameters of the model."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: dict) -> None:
        """Sets the parameters of the model.
        Args:
            value (dict): A dictionary of parameters.
        Raises:
            ValueError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def train(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Trains the model with the provided observations and ground truth.
        Args:
            observations (np.ndarray): The input features.
            ground_truth (np.ndarray): The true labels corresponding to the observations.
        """
        self.observations = observations
        self.ground_truth = ground_truth
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts labels for the given observations.
        Args:
            observations (np.ndarray): The input features for which to predict labels.
        Returns:
            np.ndarray: The predicted labels.
        """
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> Any:
        """Predicts the label for a single observation.
        Args:
            observation (np.ndarray): A single input feature.
        Returns:
            Any: The predicted label for the observation.
        """
        module = observation - self._parameters["observations"]
        distances = np.linalg.norm(module, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self._parameters["ground_truth"][i] for i in k_indices]
        most_common = pd.Series(k_nearest_labels).value_counts()
        return most_common.index[0]
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluates the model performance on test data.
        Args:
            x (np.ndarray): Test input features.
            y (np.ndarray): True labels for the test features.
        Returns:
            float: The accuracy of the model on the test data.
        """
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy

    def save(self) -> None:
        """Saves the model state."""
        super().save()

    def load(self, name: str) -> None:
        """Loads the model state."""
        super().load(name)
