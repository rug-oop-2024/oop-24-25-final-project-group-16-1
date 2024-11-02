import json
import os
import base64
from typing import Any, Dict, Optional
from copy import deepcopy


class Artifact:
    def __init__(
            self, name: str, artifact_type: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes the Artifact instance.
        Args:
            name (str): The name of the artifact.
            artifact_type (str): The type of the artifact
            (e.g. 'model', 'dataset').
            metadata (Optional[Dict[str, Any]]): Additional metadata about the
            artifact.
        """
        self.name: str = name
        self.artifact_type: str = artifact_type
        self._metadata: Dict[str, Any] = {} if metadata is None else metadata
        self._path: str = os.path.join("artifacts", f"{self.name}.json")

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Retrieves a deep copy of the metadata dictionary.
        Returns:
            Dict[str, Any]: A deep copy of the metadata dictionary associated
            with the artifact.
        """
        return deepcopy(self._metadata)

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """
        Sets the metadata dictionary for the artifact.
        Args:
            value (Dict[str, Any]): A dictionary containing metadata to be
            associated with the artifact.
        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError("Metadata must be a dictionary.")
        self._metadata = value

    @property
    def path(self) -> str:
        """
        Retrieves the file path where the artifact is stored.
        Returns:
            str: The path to the JSON file associated with the artifact.
        """
        return self._path

    @path.setter
    def path(self, value: str) -> None:
        """
        Prevents direct modification of the path attribute.
        Args:
            value (str): Any attempted value to set for the path attribute.
        Raises:
            AttributeError: Always raised to indicate that the path attribute
            is read-only and cannot be directly modified.
        """
        raise AttributeError("Path cannot be modified directly.")

    def save(self) -> None:
        """Saves the artifact to a JSON file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.toJSON(), f, indent=4)

    def load(self, name: str) -> None:
        """Loads an artifact from a JSON file.
        Args:
            name (str): The name of the artifact to load.
        """
        path = os.path.join("artifacts", f"{name}.json")
        with open(path, 'r') as f:
            data = f.read()
            loaded_artifact = self.fromJSON(data)
            self.name = loaded_artifact.name
            self.artifact_type = loaded_artifact.artifact_type
            self.metadata = loaded_artifact.metadata

    def toJSON(self) -> str:
        """Converts the artifact to a JSON string and encodes it in base64.
        Returns:
            str: Base64 encoded JSON string representation of the artifact.
        """
        artifact_dict = {
            'name': self.name,
            'artifact_type': self.artifact_type,
            'metadata': self.metadata
        }
        artifact_json = json.dumps(artifact_dict)
        return base64.b64encode(artifact_json.encode()).decode()

    @classmethod
    def fromJSON(cls, encoded_data: str) -> 'Artifact':
        """Decodes a base64 encoded string back into an Artifact instance.
        Args:
            encoded_data (str): The base64 encoded string.
        Returns:
            Artifact: An instance of the Artifact class.
        """
        artifact_json = base64.b64decode(encoded_data.encode()).decode()
        data = json.loads(artifact_json)
        return cls(
            name=data['name'],
            artifact_type=data['artifact_type'],
            metadata=data['metadata']
        )

    def __repr__(self) -> str:
        return (f"Artifact(name={self.name}, "
                f"artifact_type={self.artifact_type}, "
                f"metadata={self.metadata})")
