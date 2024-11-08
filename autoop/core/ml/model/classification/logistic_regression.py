from autoop.core.ml.model.model import Model
from sklearn.linear_model import LogisticRegression
from typing import Any


class LogisticRegressionModel(Model):
    """
    Logistic Regression model implementation that inherits from the base
    "Model" class.

    Uses LogisticRegression from scikit-learn to perform classification tasks.
    """

    def __init__(
        self,
        name: str = "Logistic Regression",
        type: str = "classification",
        max_iter: int = 100,
        penalty: str = "l2",
        C: float = 1.0,
    ):
        """
        Initializes the Logistic Regression model with specified hyperparameters.

        Args:
            name (str): The name of the model (default: "Logistic Regression").
            type (str): The type of task ('classification' by default).
            max_iter (int): Maximum number of iterations for the solver (default: 100).
            penalty (str): The norm used in the penalization ('l2' by default).
            C (float): Inverse of regularization strength (default: 1.0).
        """
        super().__init__(name=name, type=type)
        self.model = LogisticRegression(max_iter=max_iter, penalty=penalty, C=C)

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
