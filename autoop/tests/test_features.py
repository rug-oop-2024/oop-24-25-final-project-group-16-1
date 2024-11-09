import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


class TestFeatures(unittest.TestCase):
    """
    Unit tests for detecting feature types in datasets,
    focusing on numerical and categorical data differentiation.
    """

    def setUp(self) -> None:
        """
        Initialize test environment; no specific setup required.
        """
        pass

    def test_detect_features_continuous(self):
        """
        Test detection of continuous/numerical features in the Iris dataset.
        """
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertIn(feature.name, iris.feature_names)
            self.assertIn(
                feature.feature_type, ["numerical", "numeric", "continuous"]
            )

    def test_detect_features_with_categories(self):
        """
        Test detection of both categorical and numerical features
        in the Adult dataset.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)

        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertIn(feature.name, data.feature_names)

        for detected_feature in filter(
            lambda x: x.name in numerical_columns, features
        ):
            self.assertIn(
                detected_feature.feature_type, [
                    "numerical", "numeric", "continuous"
                ]
            )

        for detected_feature in filter(
            lambda x: x.name in categorical_columns, features
        ):
            self.assertEqual(detected_feature.feature_type, "categorical")
