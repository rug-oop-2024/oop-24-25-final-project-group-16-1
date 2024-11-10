from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeClassifier
from typing import Any


class DecisionTreeModel(Model):
    """
    A decision tree model that uses Scikit-Learn's
    DecisionTreeClassifier for classification tasks.
    """

    def __init__(
        self, name: str = "Decision Trees", type: str = "classification"
    ) -> None:
        """
        Initializes the DecisionTreeModel using
        Scikit-Learn's DecisionTreeClassifier.

        Args:
            name (str): The name of the model instance.
            type (str): Type of the model, default is "classification".
        """
        super().__init__(name=name, type=type)
        self._model = DecisionTreeClassifier()

    def fit(self, x: Any, y: Any) -> None:
        """
        Trains the decision tree model with the given input
        features and target labels.

        Args:
            x (Any): Input features for training,
            typically as a 2D array or DataFrame.
            y (Any): Target labels for training,
            typically a 1D array or Series.
        """
        self._model.fit(x, y)

    def predict(self, x: Any) -> Any:
        """
        Predicts labels for the provided input
        features using the trained model.

        Args:
            x (Any): Input features for prediction,
            typically a 2D array or DataFrame.

        Returns:
            Any: Predicted labels corresponding to the input features.
        """
        return self._model.predict(x)

    @property
    def model(self) -> DecisionTreeClassifier:
        """
        Gets the underlying DecisionTreeClassifier model.

        Returns:
            DecisionTreeClassifier: The Scikit-Learn
            DecisionTreeClassifier model.
        """
        return self._model
