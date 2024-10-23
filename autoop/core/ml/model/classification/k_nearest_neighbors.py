from models.base_model import Model
import numpy as np
import pandas as pd


class KNearestNeighbors(Model):
    def _init_(self, k=3):
        super()._init_()
        self.k = k
        self.observations = None
        self.ground_truth = None
        self._parameters = {}

    @property
    def parameters(self):
        return self._parameters.copy()

    @parameters.setter
    def parameters(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        self.observations = observations
        self.ground_truth = ground_truth
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in observations]
        return predictions

    def _predict_single(self, observations: np.ndarray) -> np.ndarray:
        module = observations-self._parameters["observations"]
        distances = np.linalg.norm(module, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [
            self._parameters["ground_truth"][i] for i in k_indices]
        most_common = pd.Series(k_nearest_labels).value_counts()
        return most_common.index[0]