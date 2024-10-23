from models.base_model import Model
import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso


class Lasso(Model):
    def _init_(self, alpha: float = 1.0):
        super()._init_()
        self._alpha = alpha
        self._model = None
        self._parameters = {}
        self.weights = None
        self.intercept = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Alpha should be positive.")
        self._alpha = value
        self._model = SklearnLasso(self._alpha)

    @property
    def parameters(self):
        return self._parameters.copy()

    @parameters.setter
    def parameters(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("Parameters should be provided as a dictionary.")
        self._parameters = value

    def fit(self, X: np.ndarray, ground_truth: np.ndarray) -> None:
        self._model = SklearnLasso(alpha=self._alpha)
        self._model.fit(X, ground_truth)
        self.parameters['weights'] = self._model.coef
        self.parameters['intercept'] = self._model.intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)