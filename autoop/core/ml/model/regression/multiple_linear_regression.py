from models.base_model import Model
import numpy as np


class MultipleLinearRegression(Model):
    def _init_(self):
        super()._init_()
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
        X = np.c_[observations, np.ones((observations.shape[0], 1))]
        weight = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ground_truth)
        self._parameters['weights'] = weight

    def predict(self, observations: np.ndarray) -> np.ndarray:
        X = np.c_[observations, np.ones((observations.shape[0], 1))]
        prediction = X.dot(self._parameters['weights'])
        return prediction