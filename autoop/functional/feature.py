from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects feature types in a dataset as either 'categorical' or 'numeric'.

    Args:
        dataset (Dataset): The dataset for which feature types
        are to be detected.

    Returns:
        List[Feature]: List of features with detected types.
    """
    features = []
    df = dataset.read()

    for column in df.columns:
        if pd.api.types.is_integer_dtype(
            df[column]) or pd.api.types.is_float_dtype(
            df[column]
        ):
            feature_type = "numeric"
        else:
            feature_type = "categorical"

        feature = Feature(
            name=column, feature_type=feature_type, values=df[column].tolist()
        )
        features.append(feature)

    return features
