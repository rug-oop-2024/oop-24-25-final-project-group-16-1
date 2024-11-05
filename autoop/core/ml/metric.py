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
    """Factory function to get a metric by name.
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
    """Base class for all metrics."""

    @abstractmethod
    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        """Calculates the metric based on true and predicted values.
        Args:
            y_true (List[Any]): The true labels.
            y_pred (List[Any]): The predicted labels.
        Returns:
            float: The computed metric value.
        """
        pass


class Accuracy(Metric):
    """Accuracy metric implementation."""

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
        if len(y_true) > 0:
            return correct / len(y_true)
        return 0.0


class MeanSquaredError(Metric):
    """Mean Squared Error metric implementation."""

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric implementation."""

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


class R2Score(Metric):
    """R-squared score metric implementation."""

    def evaluate(self, y_true: List[float], y_pred: List[float]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        ss_total = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
        ss_residual = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
        if ss_total > 0:
            return 1 - (ss_residual / ss_total)
        return 0.0


class Precision(Metric):
    """Precision metric implementation."""

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        true_positive = sum(
            1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1
        )
        predicted_positive = sum(1 for yp in y_pred if yp == 1)
        if predicted_positive > 0:
            return true_positive / predicted_positive
        return 0.0


class Recall(Metric):
    """Recall metric implementation."""

    def evaluate(self, y_true: List[Any], y_pred: List[Any]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length of true labels and predicted labels must match."
            )
        true_positive = sum(
            1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1
        )
        actual_positive = sum(1 for yt in y_true if yt == 1)
        if actual_positive > 0:
            return true_positive / actual_positive
        return 0.0
