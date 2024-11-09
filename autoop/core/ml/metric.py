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
    Factory function to retrieve a metric class instance by name.

    Args:
        name (str): The name of the metric.

    Returns:
        Type[Metric]: A metric class instance based on the provided name.

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
    Base class for all metrics. Enforces implementation
    of the `evaluate` method.
    """

    @abstractmethod
    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Calculates the metric based on true and predicted values.

        Args:
            y_true (List[Any]): The true labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The computed metric value.
        """
        pass


class Accuracy(Metric):
    """
    Calculates the accuracy of predictions.
    """

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Evaluates accuracy, calculating the ratio of correct predictions
        to the total number of predictions.

        Args:
            y_true (List[Any]): The true labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The accuracy score, or 0.0 if `y_true` is empty.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
        return correct / len(y_true) if y_true else 0.0


class MeanSquaredError(Metric):
    """
    Calculates the Mean Squared Error (MSE) of predictions.
    """

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Computes the mean of the squared differences
        between true and predicted values.

        Args:
            y_true (List[float]): The true labels.
            y_pred (List[float]): The predicted labels.

        Returns:
            float: The Mean Squared Error.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """
    Calculates the Mean Absolute Error (MAE) of predictions.
    """

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Computes the mean of absolute differences
        between true and predicted values.

        Args:
            y_true (List[float]): The true labels.
            y_pred (List[float]): The predicted labels.

        Returns:
            float: The Mean Absolute Error.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


class R2Score(Metric):
    """
    Calculates the R-squared (R²) score of predictions.
    """

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Computes the R-squared value, representing the proportion of variance
        in `y_true` that is predictable from `y_pred`.

        Args:
            y_true (List[float]): The true labels.
            y_pred (List[float]): The predicted labels.

        Returns:
            float: The R-squared score, or 0.0 if variance in `y_true` is zero.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        ss_total = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
        ss_residual = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0


class Precision(Metric):
    """
    Calculates the precision of predictions.
    """

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Computes the ratio of true positive
        predictions to all positive predictions.

        Args:
            y_true (List[Any]): The true labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The precision score, or 0.0 if no
            positive predictions were made.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        true_positive = sum(
            1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1
        )
        predicted_positive = sum(1 for yp in y_pred if yp == 1)
        return (
            true_positive / predicted_positive
            if predicted_positive > 0
            else 0.0
        )


class Recall(Metric):
    """
    Calculates the recall of predictions.
    """

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """
        Computes the ratio of true positive predictions
        to all actual positive cases.

        Args:
            y_true (List[Any]): The true labels.
            y_pred (List[Any]): The predicted labels.

        Returns:
            float: The recall score, or 0.0 if there are no actual positives.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        true_positive = sum(
            1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1
        )
        actual_positive = sum(1 for yt in y_true if yt == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0.0
