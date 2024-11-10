from autoop.core.ml.model.model import Model
from sklearn.ensemble import RandomForestClassifier
from typing import Any


class RandomForestModel(Model):
    """
    Random Forest model implementation that inherits
    from the base "Model" class.
    Uses RandomForestClassifier from scikit-learn
    to perform classification tasks.
    """

    def __init__(
        self,
        name: str = "Random Forest",
        type: str = "classification",
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = None,
    ):
        """
        Initializes the Random Forest model with specified hyperparameters.

        Args:
            name (str): The name of the model (default: "Random Forest").
            type (str): The type of task ('classification' by default).
            n_estimators (int): The number of trees
            in the forest (default: 100).
            max_depth (int): The maximum depth of the trees (default: None).
            random_state (int): Seed used by the
            random number generator (default: None).
        """
        super().__init__(name=name, type=type)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
        )

    @property
    def n_estimators(self) -> int:
        """
        Get the number of trees in the forest.

        Returns:
            int: The number of trees (n_estimators).
        """
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, value: int) -> None:
        """
        Set the number of trees in the forest and reinitialize the model.

        Args:
            value (int): The new number of trees.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        self._n_estimators = value
        self._initialize_model()

    @property
    def max_depth(self) -> int:
        """
        Get the maximum depth of the trees.

        Returns:
            int: The maximum depth of the trees.
        """
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        """
        Set the maximum depth of the trees and reinitialize the model.

        Args:
            value (int): The new maximum depth.

        Raises:
            ValueError: If value is not None or a positive integer.
        """
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("max_depth must be None or a positive integer.")
        self._max_depth = value
        self._initialize_model()

    @property
    def random_state(self) -> int:
        """
        Get the random state for the model.

        Returns:
            int: The random state used by the model.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int) -> None:
        """
        Set the random state for the model and reinitialize the model.

        Args:
            value (int): The new random state.

        Raises:
            ValueError: If value is not an integer.
        """
        if not isinstance(value, int):
            raise ValueError("random_state must be an integer.")
        self._random_state = value
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initializes the RandomForestClassifier with current parameters.
        """
        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
        )

    def fit(self, x: Any, y: Any) -> None:
        """
        Train the model with input features x and target labels y.

        Args:
            x (Any): The input features used for training.
            y (Any): The target labels for supervised training.
        """
        self._model.fit(x, y)

    def predict(self, x: Any) -> Any:
        """
        Predict labels for the given input features x.

        Args:
            x (Any): The input features to predict labels for.

        Returns:
            Any: The predicted labels for the input features.
        """
        return self._model.predict(x)
