from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeClassifier
from typing import Any


class DecisionTreeModel(Model):
    def _init_(self, name: str, **kwargs):
        super()._init_(name=name, type="DecisionTreeClassifier")
        self.model = DecisionTreeClassifier(**kwargs)

    def fit(self, x: Any, y: Any) -> None:
        """Train model with input features x and target labels y."""
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        """Predict labels for the given input features x."""
        return self.model.predict(x)
