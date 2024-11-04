import json
import re
import os
import base64
from typing import Any, Dict, Optional, List
from copy import deepcopy


class Artifact:
    def __init__(
            self,
            name: str,
            type: str,
            metadata: Optional[Dict[str, Any]] = None,
            data: Optional[bytes] = None,
            asset_path: str = "",
            version: str = "1.0",
            tags: Optional[List[str]] = None
    ) -> None:
        """
        Initializes the Artifact instance.

        Args:
            name (str): The name of the artifact.
            type (str): The type of the artifact (e.g., 'model', 'dataset').
            metadata (Optional[Dict[str, Any]]): Additional metadata about the artifact.
            data (Optional[bytes]): The binary data associated with the artifact.
            asset_path (str): The path where the artifact's data is stored.
            version (str): The version of the artifact.
            tags (Optional[List[str]]): A list of tags associated with the artifact.
        """
        self.name = name
        self.type = type
        self.metadata = {} if metadata is None else metadata
        self.data = data
        self.asset_path = asset_path
        self.version = version
        self.tags = tags or []
        self.id = f"{self.name}_{self.version}"
        self.path = os.path.join("artifacts", f"{self.name}.json")

    @staticmethod
    def sanitize_filename(filename):
        return re.sub(r"[^a-zA-Z0-9_]", "_", filename)

    def save(self) -> None:
        """Saves the artifact to a JSON file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self) -> None:
        """Loads the artifact from its JSON file."""
        with open(self.path, 'r') as f:
            data = json.load(f)
            self.name = data['name']
            self.type = data['type']
            self.type = data['type']
            self.metadata = data['metadata']
            self.data = base64.b64decode(data['data']) if 'data' in data else None
            self.asset_path = data.get('asset_path', '')
            self.version = data.get('version', '1.0')
            self.tags = data.get('tags', [])
            self.id = data.get('id', f"{self.name}_{self.version}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for JSON serialization."""
        artifact_dict = {
            'name': self.name,
            'type': self.type,
            'metadata': self.metadata,
            'asset_path': self.asset_path,
            'version': self.version,
            'tags': self.tags,
            'id': self.id
        }
        if self.data:
            artifact_dict['data'] = base64.b64encode(self.data).decode('utf-8')
        return artifact_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Artifact':
        """Creates an Artifact instance from a dictionary."""
        data_bytes = base64.b64decode(data['data']) if 'data' in data else None
        return cls(
            name=data['name'],
            type=data['type'],
            metadata=data.get('metadata', {}),
            data=data_bytes,
            asset_path=data.get('asset_path', ''),
            version=data.get('version', '1.0'),
            tags=data.get('tags', [])
        )

    def __repr__(self) -> str:
        return (f"Artifact(name={self.name}, "
                f"type={self.type}, "
                f"version={self.version}, "
                f"tags={self.tags}, "
                f"id={self.id})")
