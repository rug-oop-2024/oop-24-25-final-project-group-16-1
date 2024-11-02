import numpy as np
from collections import Counter
from typing import Optional, Tuple, List, Union
from autoop.core.ml.model.model import Model
from copy import deepcopy

class DecisionTreeNode:
    def __init__(self, 
                 feature: Optional[int] = None, 
                 threshold: Optional[float] = None, 
                 left: Optional['DecisionTreeNode'] = None, 
                 right: Optional['DecisionTreeNode'] = None, 
                 value: Optional[int] = None
    ) -> None:
        """
        Represents a node in the decision tree.
        Args:
            feature (Optional[int]): The index of the feature to split on.
            threshold (Optional[float]): The threshold value for the split.
            left (Optional[DecisionTreeNode]): The left child node.
            right (Optional[DecisionTreeNode]): The right child node.
            value (Optional[int]): The predicted value for leaf nodes.
        """
        self.feature: Optional[int] = feature
        self.threshold: Optional[float] = threshold
        self.left: Optional['DecisionTreeNode'] = left
        self.right: Optional['DecisionTreeNode'] = right
        self.value: Optional[int] = value


class DecisionTreeClassifier(Model):
    def __init__(
            self, name: str, max_depth: int = 10, artifact_type: str = "model"
    ) -> None:
        """
        Initializes the DecisionTreeClassifier.

        Args:
            name (str): The name of the model.
            max_depth (int): The maximum depth of the tree.
            artifact_type (str): The type of artifact, defaults to "model".
        """
        super().__init__(name=name, artifact_type=artifact_type)
        self._max_depth: int = max_depth
        self._root: Optional[DecisionTreeNode] = None

    @property
    def max_depth(self) -> int:
        """Get the maximum depth of the tree."""
        return self._max_depth

    @max_depth.setter
    def max_depth(self, depth: int) -> None:
        """Set the maximum depth of the tree."""
        self._max_depth = depth

    @property
    def root(self) -> Optional[DecisionTreeNode]:
        """Get the root node of the tree."""
        return deepcopy(self._root)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the decision tree classifier.
        Args:
            x (np.ndarray): The feature matrix.
            y (np.ndarray): The target values.
        """
        self._root = self._grow_tree(x, y)

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """
        Recursively grow the decision tree.
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The target values.
            depth (int): The current depth of the tree.
        Returns:
            DecisionTreeNode: The root node of the subtree.
        """
        if depth >= self._max_depth or len(np.unique(y)) == 1:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        feature_idx, threshold = self._best_split(x, y)
        left_idxs, right_idxs = self._split(x[:, feature_idx], threshold)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth + 1)

        return DecisionTreeNode(feature=feature_idx, threshold=threshold, left=left, right=right)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split the data.
        Args:
            x (np.ndarray): The feature matrix.
            y (np.ndarray): The target values.
        Returns:
            Tuple[int, float]: The index of the best feature and the threshold.
        """
        # Implementation of finding the best split goes here
        pass

    def _split(self, x_column: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset based on a feature and threshold.
        Args:
            x_column (np.ndarray): The feature column to split.
            threshold (float): The threshold to split on.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices of left and right splits.
        """
        left_idxs = np.where(x_column <= threshold)[0]
        right_idxs = np.where(x_column > threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Get the most common label in the target values.
        Args:
            y (np.ndarray): The target values.
        Returns:
            int: The most common label.
        """
        return np.bincount(y).argmax()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples.
        Args:
            x (np.ndarray): The feature matrix.
        Returns:
            np.ndarray: Predicted class labels.
        """
        return np.array([self._traverse_tree(xx, self._root) for xx in x])

    def _traverse_tree(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """
        Traverse the decision tree to make a prediction.
        Args:
            x (np.ndarray): The input sample.
            node (DecisionTreeNode): The current node in the tree.
        Returns:
            int: The predicted class label.
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model performance on test data.
        Args:
            X (np.ndarray): Test input features.
            y (np.ndarray): True labels for the test features.
        Returns:
            float: The accuracy of the model on the test data.
        """
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy

    def save(self) -> None:
        """Saves the model state."""
        super().save()

    def load(self, name: str) -> None:
        """Loads the model state."""
        super().load(name)
