from typing import Any, Dict, List, Literal
from copy import deepcopy
import numpy as np


class Feature:
    def __init__(
        self,
        name: str,
        feature_type: Literal["numeric", "categorical"],
        values: List[Any],
    ) -> None:
        """
        Initializes the Feature instance.
        Args:
            name (str): The name of the feature.
            feature_type (Literal['numeric', 'categorical'):
                The type of the feature, which can be 'numeric' or
                'categorical'.
            values (List[Any]): A list of values for the feature, where values
                can be of any type depending on the feature type.
        """
        self._name: str = name
        self._feature_type: Literal["numeric", "categorical"] = feature_type
        self._values: List[Any] = values

    @property
    def name(self) -> str:
        """Gets the name of the feature."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Sets the name of the feature.
        Args:
            value (str): The new name for the feature.
        Raises:
            ValueError: If the name is an empty string.
        """
        if len(value) == 0:
            raise ValueError("Feature name cannot be an empty string.")
        self._name = value

    @property
    def feature_type(self) -> Literal["numeric", "categorical"]:
        """Gets the feature type."""
        return self._feature_type

    @feature_type.setter
    def feature_type(self, value: Literal["numeric", "categorical"]) -> None:
        """
        Sets the feature type.
        Args:
            value (Literal['numeric', 'categorical']): The new
            type for the feature.
        Raises:
            ValueError: If the feature type is not one of the allowed values.
        """
        if value not in {"numeric", "categorical"}:
            raise ValueError("Feature type must be 'numeric' or 'categorical'.")
        self._feature_type = value

    @property
    def values(self) -> List[Any]:
        """Gets a copy of the feature values."""
        return deepcopy(self._values)

    @values.setter
    def values(self, value: List[Any]) -> None:
        """
        Sets the values of the feature.
        Args:
            value (List[Any]): A new list of values for the feature.
        Raises:
            ValueError: If the provided value is not a list.
        """
        if not isinstance(value, list):
            raise ValueError("Values must be provided as a list.")
        self._values = value

    def get_statistics(self) -> Dict[str, float]:
        """Calculates basic statistics for numeric features.
        Returns:
            Dict[str, float]: A dictionary containing the mean, standard
            deviation, minimum, and maximum values of the numeric feature.
        Raises:
            ValueError: If the feature type is not 'numeric'.
        """
        if self.feature_type == "numeric":
            return {
                "mean": float(np.mean(self._values)),
                "std": float(np.std(self._values)),
                "min": float(np.min(self._values)),
                "max": float(np.max(self._values)),
            }
        raise ValueError("Statistics can only be computed 4 numeric features.")

    def __str__(self) -> str:
        """String representation of the Feature instance.

        Returns:
            str: A formatted string describing the feature, including its name,
            type, and values.
        """
        return (
            f"Feature(name={self.name}, "
            f"type={self.feature_type}, "
            f"values={self.values})"
        )
