from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_features(
    features: List[Feature], dataset: Dataset
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Preprocesses features by encoding categorical and scaling numeric data.

    Args:
        features (List[Feature]): List of features to preprocess.
        dataset (Dataset): Dataset object containing raw data.

    Returns:
        List[Tuple[str, np.ndarray, dict]]: A list of tuples, each containing:
            - feature name (str)
            - preprocessed data as a NumPy array
            - artifact dictionary with encoder/scaler metadata
    """
    results = []
    raw = dataset.read()

    for feature in features:
        if feature.feature_type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
            )
            artifact = {
                "type": "OneHotEncoder", "encoder": encoder.get_params()
            }
            results.append((feature.name, data.toarray(), artifact))
        elif feature.feature_type in ["numeric", "numerical"]:
            scaler = StandardScaler()
            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
            )
            artifact = {
                "type": "StandardScaler", "scaler": scaler.get_params()
            }
            results.append((feature.name, data, artifact))
        else:
            raise ValueError(
                f"Unsupported feature type: {feature.feature_type} "
                f"for feature {feature.name}"
            )

    results = sorted(results, key=lambda x: x[0])
    return results
