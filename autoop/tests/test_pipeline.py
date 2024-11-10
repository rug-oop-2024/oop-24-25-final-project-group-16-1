from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.metric import MeanSquaredError


class TestPipeline(unittest.TestCase):
    """
    Unit tests for the Pipeline class, validating key functionalities
    such as initialization, feature preprocessing, data splitting,
    model training, and evaluation.
    """

    def setUp(self) -> None:
        """
        Set up the test environment by initializing the dataset,
        features, and pipeline for the Adult dataset.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(
                filter(lambda x: x.name != "age", self.features)
            ),
            target_feature=Feature(
                name="age", feature_type="numeric", values=[]
            ),
            metrics=[MeanSquaredError()],
            split=0.8,
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        """
        Test pipeline initialization to ensure it's an instance of Pipeline.
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        """
        Test feature preprocessing to ensure all features are processed.
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        """
        Test data splitting to validate correct train-test split sizes.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(
            self.pipeline._train_X[0].shape[0], int(0.8 * self.ds_size)
        )
        self.assertEqual(
            self.pipeline._test_X[0].shape[0],
            self.ds_size - int(0.8 * self.ds_size)
        )

    def test_train(self):
        """
        Test model training to ensure parameters are updated after training.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        """
        Test evaluation to verify predictions and metric results are generated.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()

        train_X = self.pipeline._compact_vectors(self.pipeline._train_X)
        train_y = self.pipeline._train_y

        self.pipeline._evaluate(train_X, train_y)

        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)
