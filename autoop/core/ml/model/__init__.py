from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeModel,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors,
)
from autoop.core.ml.model.classification.random_forest import RandomForestModel
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import (
    LinearRegressionModel
)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_regression import Lasso
from typing import Type

CLASSIFICATION_MODELS = [
    "Decision Trees",
    "K-Nearest Neighbors",
    "Random Forest",
]

REGRESSION_MODELS = [
    "Lasso Regression",
    "Linear Regression",
    "Multiple Linear Regression",
]


def get_model(name: str) -> Type[Model]:
    """
    Factory function to retrieve a Model instance based on the model name.

    Args:
        name (str): The name of the Model.

    Returns:
        Type[Model]: An instance of the specified Model class.

    Raises:
        ValueError: If the Model name is not recognized.

    Available Models:
        - Classification Models: Decision Trees, K-Nearest Neighbors,
        Random Forest
        - Regression Models: Lasso Regression, Linear Regression,
          Multiple Linear Regression
    """
    models_map = {
        "Decision Trees": DecisionTreeModel,
        "K-Nearest Neighbors": KNearestNeighbors,
        "Random Forest": RandomForestModel,
        "Lasso Regression": Lasso,
        "Linear Regression": LinearRegressionModel,
        "Multiple Linear Regression": MultipleLinearRegression,
    }

    if name not in models_map:
        raise ValueError(
            f"Model '{name}' is not recognized. Available Models: "
            f"{', '.join(models_map.keys())}"
        )

    model_class = models_map[name]
    model_type = (
        "classification" if name in CLASSIFICATION_MODELS else "regression"
    )

    return model_class(name=name, type=model_type)
