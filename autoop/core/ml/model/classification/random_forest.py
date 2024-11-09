from autoop.core.ml.model.model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Any


class RandomForestModel(Model):
    """
    Random Forest model implementation that inherits
    from the base "Model" class.
    Uses RandomForestClassifier from scikit-learn
    to perform classification tasks.
    """

    def __init__(
        self,
        name: str = "Random Forest",
        type: str = "classification",
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = None,
    ):
        """
        Initializes the Random Forest model with specified hyperparameters.

        Args:
            name (str): The name of the model (default: "Random Forest").
            type (str): The type of task ('classification' by default).
            n_estimators (int): The number of trees
            in the forest (default: 100).
            max_depth (int): The maximum depth of the trees (default: None).
            random_state (int): Seed used by the
            random number generator (default: None).
        """
        super().__init__(name=name, type=type)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, x: Any, y: Any) -> None:
        """
        Train the model with input features x and target labels y.

        Args:
            x (Any): The input features used for training.
            y (Any): The target labels for supervised training.
        """
        self.model.fit(x, y)

    def predict(self, x: Any) -> Any:
        """
        Predict labels for the given input features x.

        Args:
            x (Any): The input features to predict labels for.

        Returns:
            Any: The predicted labels for the input features.
        """
        return self.model.predict(x)
