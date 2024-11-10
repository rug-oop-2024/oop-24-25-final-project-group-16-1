from abc import ABC, abstractmethod
from typing import Any, List, Type
import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy",
    "precision",
    "recall",
]


def get_metric(name: str) -> Type["Metric"]:
    """
    Retrieves a metric class instance based on the provided name.

    Args:
        name (str): The name of the desired metric.

    Returns:
        Type[Metric]: An instance of the metric class
        associated with the specified name.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    metrics_map = {
        "mean_squared_error": MeanSquaredError,
        "mean_absolute_error": MeanAbsoluteError,
        "r2_score": R2Score,
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall,
    }
    if name not in metrics_map:
        raise ValueError(
            f"Metric '{name}' is not recognized. Available metrics: {METRICS}"
        )
    return metrics_map[name]()


class Metric(ABC):
    """
    Abstract base class for all metrics, requiring the implementation
    of an `evaluate` method for calculating the metric value.
    """

    @abstractmethod
    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Computes the metric based on the true and predicted values.

        Args:
            y_true (List[Any]): The actual labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The calculated metric value.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns the metric name as a string representation.

        Returns:
            str: The name of the metric.
        """
        pass

    def __repr__(self) -> str:
        """
        Returns a string representation of the class name,
        useful for debugging and logging purposes.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__


class MeanSquaredError(Metric):
    """
    Metric class for calculating the Mean Squared Error (MSE) of predictions.
    """

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Computes the MSE, the mean of squared differences
        between true and predicted values.

        Args:
            y_true (List[float]): The actual labels.
            y_pred (List[float]): The predicted labels.

        Returns:
            float: The calculated Mean Squared Error.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        if len(y_true) != len(y_pred):
            raise ValueError(
                "The lengths of true and predicted labels must match."
            )
        return np.mean((y_true - y_pred) ** 2)

    def __str__(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name "mean_squared_error" representing this metric.
        """
        return "mean_squared_error"


class MeanAbsoluteError(Metric):
    """
    Metric class for calculating the Mean Absolute Error (MAE) of predictions.
    """

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Computes the MAE, the mean of absolute
        differences between true and predicted values.

        Args:
            y_true (List[float]): The actual labels.
            y_pred (List[float]): The predicted labels.

        Returns:
            float: The calculated Mean Absolute Error.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "The lengths of true and predicted labels must match."
            )
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    def __str__(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name "mean_absolute_error" representing this metric.
        """
        return "mean_absolute_error"


class R2Score(Metric):
    """
    Metric class for calculating the R-squared (R²) score of predictions.
    """

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Computes the R-squared value, representing the proportion of variance
        in `y_true` that is predictable from `y_pred`.

        Args:
            y_true (List[float]): The actual labels.
            y_pred (List[float]): The predicted labels.

        Returns:
            float: The R-squared score. Returns
            0.0 if the variance of `y_true` is zero.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "The lengths of true and predicted labels must match."
            )
        ss_total = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
        ss_residual = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

    def __str__(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name "r2_score" representing this metric.
        """
        return "r2_score"


class Precision(Metric):
    """
    Metric class for calculating the precision
    of binary classification predictions.
    """

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Computes precision, the ratio of true positive
        predictions to all positive predictions.

        Args:
            y_true (List[Any]): The actual labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The precision score, or 0.0 if no
            positive predictions were made.
        """
        true_positive = np.count_nonzero((y_true == 1) & (y_pred == 1))
        predict_positive = np.count_nonzero(y_pred == 1)
        return (
            true_positive / predict_positive
            if predict_positive > 0
            else 0.0
        )

    def __str__(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name "precision" representing this metric.
        """
        return "precision"


class Recall(Metric):
    """
    Metric class for calculating the recall of
    binary classification predictions.
    """

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Computes recall, the ratio of true positive predictions
        to all actual positives.

        Args:
            y_true (List[Any]): The actual labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The recall score, or 0.0 if there
            are no actual positive labels.
        """
        true_positive = np.count_nonzero((y_true == 1) & (y_pred == 1))
        actual_positive = np.count_nonzero(y_true == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0.0

    def __str__(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name "recall" representing this metric.
        """
        return "recall"


class Accuracy(Metric):
    """
    Metric class for calculating the accuracy of predictions.
    """

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Computes accuracy, the ratio of correct predictions
        to the total predictions.

        Args:
            y_true (List[Any]): The actual labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The accuracy score, or 0.0 if `y_true` is empty.
        """
        return np.mean(y_true == y_pred)

    def __str__(self) -> str:
        """
        Returns the name of the metric as a string.

        Returns:
            str: The name "accuracy" representing this metric.
        """
        return "accuracy"
