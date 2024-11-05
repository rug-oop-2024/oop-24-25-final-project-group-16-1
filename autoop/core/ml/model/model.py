from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
from typing import Any


class Model(ABC, Artifact):
    def __init__(self, name: str, type: str = "model"):
        super().__init__(name=name, type=type)

    @abstractmethod
    def fit(self, x: Any, y: Any) -> None:
        """Train the model with input features x and target labels y."""
        pass

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Predict labels or values given input features x."""
        pass
