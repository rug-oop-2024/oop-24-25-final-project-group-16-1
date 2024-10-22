import json
import os
import base64
from typing import Any, Dict, Optional


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
        self.metadata: Dict[str, Any] = (
            metadata if metadata is not None else {}
        )
        self.path: str = os.path.join("artifacts", f"{self.name}.json")

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
            data = json.load(f)
            self.name = data['name']
            self.artifact_type = data['artifact_type']
            self.metadata = data['metadata']

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
