from typing import List
import pickle
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    The Pipeline class manages the sequence of operations required for
    a machine learning workflow, including data preprocessing,
    model training, and evaluation.
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ) -> None:
        """
        Initiates the Pipeline instance with the args: metrics, model,
        dataset, and features.

        Args:
            metrics (List[Metric]): List of metric objects for evaluation.
            dataset (Dataset): Dataset instance for training/testing.
            model (Model): Model instance for training and predictions.
            input_features (List[Feature]): List of input features.
            target_feature (Feature): The target feature for predictions.
            split (float): The train-test split ratio (set as default: 0.8).
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics if isinstance(metrics, list) else [metrics]
        self._artifacts = {}
        self._split = split

        if target_feature.feature_type == "categorical":
            if model.type != "classification":
                raise ValueError(
                    """
                    Model type must be classification for categorical
                    target feature
                    """
                )

        if target_feature.feature_type == "numeric":
            if model.type != "regression":
                raise ValueError(
                    """
                    Model type must be regression for numerical target feature
                    """
                )

    def __str__(self) -> str:
        """
        Returns a string representation of the Pipeline instance,
        showing the model type, input and target features, split ratio,
        and evaluation metrics.

        Returns:
            str: String representation of the Pipeline instance.
        """
        return (
            f"Pipeline(\n"
            f"    model={self._model.type},\n"
            f"    input_features={list(map(str, self._input_features))},\n"
            f"    target_feature={str(self._target_feature)},\n"
            f"    split={self._split},\n"
            f"    metrics={list(map(str, self._metrics))},\n"
            f")"
        )

    @property
    def model(self) -> Model:
        """
        Returns the model used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieves the artifacts generated during the pipeline execution
        in order to be saved.

        Returns:
            List[Artifact]: List of artifacts generated.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type == "OneHotEncoder":
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type == "StandardScaler":
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact):
        """
        Registers an artifact under the given name.

        Args:
            name (str): The artifact name.
            artifact: The artifact to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """
        Preprocesses the input and target features, storing necessary
        transformation artifacts for reproducibility.
        """
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)

        input_results = preprocess_features(
            self._input_features, self._dataset
        )

        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self):
        """
        Splits the data into training and
        testing sets based on the split ratio.
        Raises:
            ValueError: If no input vectors are found for splitting.
        """
        split = self._split

        if not self._input_vectors:
            raise ValueError(
                "No input vectors to split. Ensure _preprocess_features "
                "has populated input data."
            )

        self._train_X = [
            vector[
                : int(split * len(vector))
            ] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[
                int(split * len(vector)):
            ] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenates multiple vectors into a single array.

        Args:
            vectors (List[np.array]): List of arrays to concatenate.

        Returns:
            np.array: Concatenated array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self):
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self, X: np.ndarray, Y: np.ndarray) -> List[tuple]:
        """
        Evaluates the model predictions using the provided data.

        Args:
            X (np.ndarray): Input features for evaluation.
            Y (np.ndarray): True output values.

        Returns:
            List[tuple]: List of evaluation metrics and their values.
        """
        predictions = self._model.predict(X)
        self._predictions = predictions

        metrics_results = []
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            metrics_results.append((metric, result))

        self._metrics_results = metrics_results
        return metrics_results, predictions

    def execute(self):
        """
        Executes the pipeline: preprocessing, data splitting, training,
        and evaluation.

        Returns:
            dict: Dictionary containing training and testing metrics
                  and predictions.
        """
        self._preprocess_features()
        self._split_data()

        self._train()
        train_X = self._compact_vectors(self._train_X)
        train_y = self._train_y
        train_metrics_results, train_predictions = self._evaluate(
            train_X, train_y
        )

        test_X = self._compact_vectors(self._test_X)
        test_y = self._test_y
        test_metrics_results, test_predictions = self._evaluate(test_X, test_y)

        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results,
            "train_predictions": train_predictions,
            "test_predictions": test_predictions,
        }
