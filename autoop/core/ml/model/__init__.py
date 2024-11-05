from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeModel,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors
)
from autoop.core.ml.model.classification.neural_networks import NeuralNetwork
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import (
    LinearRegressionModel
)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_regression import Lasso

CLASSIFICATION_MODELS = [
    "Decision Trees",
    "K Nearest Neighbors",
    "Neural Networks"
]

REGRESSION_MODELS = [
    "Lasso Regression",
    "Linear Regression",
    "Multiple Linear Regression",
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "Decision Trees":
        return DecisionTreeModel(name="Decision Trees")
    elif model_name == "K Nearest Neighbors":
        return KNearestNeighbors(name="K Nearest Neighbors")
    elif model_name == "Neural Networks":
        return NeuralNetwork(name="Neural Networks")
    elif model_name == "Lasso Regression":
        return Lasso(name="Lasso Regression")
    elif model_name == "Linear Regression":
        return LinearRegressionModel(name="Linear Regression")
    elif model_name == "Multiple Linear Regression":
        return MultipleLinearRegression(name="Multiple Linear Regression")
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")
