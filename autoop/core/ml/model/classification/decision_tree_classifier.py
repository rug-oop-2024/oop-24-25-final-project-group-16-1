from autoop.core.ml.model.model import Model
from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.metric import Accuracy
from typing import Any

class DecisionTreeModel(Model):
    def _init_(self, name: str, **kwargs):
        super()._init_(name=name, type="DecisionTreeClassifier")
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, x: Any, y: Any) -> None:
        """Train the Decision Tree model with input features x and target labels y."""
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        """Predict labels for the given input features x."""
        return self.model.predict(x)

    def evaluate(self, x: Any, y: Any) -> float:
        """Evaluate the model performance on test data using accuracy."""
        predictions = self.predict(x)
        return Accuracy(y, predictions)

    def save(self) -> None:
        """Save the model state."""
        super().save()

    def load(self, name: str) -> None:
        """Load the model state."""
        super().load(name)
