from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeClassifier
from typing import Any


class DecisionTreeModel(Model):
    def __init__(self, name: str = "Decision Trees", type: str = "classification"):
        super().__init__(name=name, type=type)
        self.model = DecisionTreeClassifier()

    def fit(self, x: Any, y: Any) -> None:
        """
        Train model with input features x and target labels y.
        """
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        """
        Predict labels for the given input features x.
        """
        return self.model.predict(x)
