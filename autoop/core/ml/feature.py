from typing import Any, Dict, List, Literal
from copy import deepcopy
import numpy as np


class Feature:
    """
    Represents a feature with a name, type, and list of values in which
    it encapsulates attributes to ensure validation and supports
    calculating basic statistics for numeric features.

    Attributes:
        _name (str): The name of the feature.
        _feature_type (
            Literal["numeric", "categorical"]
        ): The type of feature, either numeric or categorical.
        _values (List[Any]): The values of the feature.
    """

    def __init__(
        self,
        name: str,
        feature_type: Literal["numeric", "categorical"],
        values: List[Any],
    ) -> None:
        """
        Initiates the Feature instance.

        Args:
            name (str): The name of the feature.
            feature_type (Literal['numeric', 'categorical']): The type
            of the feature, which can
                be 'numeric' or 'categorical'.
            values (List[Any]): A list of values for the feature,
                where values can be of any type depending on the feature type.
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
            raise ValueError(
                "Feature type must be 'numeric' or 'categorical'."
            )
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
        """
        Calculates basic statistics for numeric features, including
        mean, standard deviation, minimum, and maximum values.

        Returns:
            Dict[str, float]: A dictionary containing statistics for feature.

        Raises:
            ValueError: If the feature type is not 'numeric'.
        """
        if self._feature_type == "numeric":
            return {
                "mean": float(np.mean(self._values)),
                "std": float(np.std(self._values)),
                "min": float(np.min(self._values)),
                "max": float(np.max(self._values)),
            }
        raise ValueError(
            "Statistics can only be computed for numeric features."
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the Feature instance,
        including its name, type, and values.

        Returns:
            str: A formatted string describing the feature.
        """
        return (
            f"Feature(name={self.name}, "
            f"type={self.feature_type}, "
            f"values={self.values})"
        )
