from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
from typing import Any


class Model(ABC, Artifact):
    """
    Base abstract class for machine learning models.
    Inherits from `Artifact` to enable serialization and
    metadata management for trained models.

    Attributes:
        name (str): The name of the model.
        type (str): The type of the model if its:
            'classification', 'regression'.
    """

    def __init__(self, name: str, type: str) -> None:
        """
        Initiates the base Model with a name and type. This class is abstract
        and cannot be instantiated directly.

        Args:
            name (str): Name of the model.
            type (str): The type of the model if its:
                'classification', 'regression'.
        """
        super().__init__(name=name, type=type)

    @abstractmethod
    def fit(self, x: Any, y: Any) -> None:
        """
        Train the model with input features x and target labels y. This method
        is implemented by subclasses.

        Args:
            x (Any): Input features for training.
            y (Any): Target labels or values for training.
        """
        pass

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """
        Predict labels or values given input features x. This method is
        implemented by subclasses.

        Args:
            x (Any): Input features for prediction.

        Returns:
            Any: Predicted values based on the model's training.
        """
        pass
