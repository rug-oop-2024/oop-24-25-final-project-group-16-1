from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
from typing import Any

class Model(ABC, Artifact):
    def __init__(self, name: str, artifact_type: str = "model"):
        super().__init__(name=name, artifact_type=artifact_type)
    
    @abstractmethod
    def train(self, x: Any, y: Any) -> None:
        """Train the model with input features x and target labels y."""
        pass

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Predict labels or values given input features x."""
        pass

    @abstractmethod
    def evaluate(self, x: Any, y: Any) -> float:
        """Evaluate the model performance on test data."""
        pass

    def save(self) -> None:
        """Save model state."""
        super().save()

    def load(self, name: str) -> None:
        """Load model state."""
        super().load(name)

