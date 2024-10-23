from typing import Any, Dict, List, Literal
import numpy as np


class Feature:
    def __init__(
            self,
            name: str,
            feature_type: Literal['numeric', 'categorical'],
            values: List[Any]
    ) -> None:
        """
        Initializes the Feature instance.
        Args:
            name (str): The name of the feature.
            feature_type (Literal['numeric', 'categorical']): The type of the
            feature, which can either be 'numeric' or 'categorical'.
            values (List[Any]): A list of values for the feature, where values
            can be of any type, depending on the feature type.
        """
        self.name: str = name
        self.feature_type: Literal['numeric', 'categorical'] = feature_type
        self.values: List[Any] = values

    def get_statistics(self) -> Dict[str, float]:
        """Calculates basic statistics for numeric features.
        Returns:
            Dict[str, float]: A dictionary containing the mean, standard
            deviation, minimum, and maximum values of the numeric feature.
        Raises:
            ValueError: If the feature type is not 'numeric'.
        """
        if self.feature_type == 'numeric':
            return {
                'mean': float(np.mean(self.values)),
                'std': float(np.std(self.values)),
                'min': float(np.min(self.values)),
                'max': float(np.max(self.values))
            }
        raise ValueError(
            "Statistics can only be computed for numeric features."
        )

    def __str__(self) -> str:
        """String representation of the Feature instance.
        Returns:
            str: A formatted string describing the feature, including its name,
            type, and values.
        """
        return (f"Feature(name={self.name}, "
                f"type={self.feature_type}, "
                f"values={self.values})")
