import json
import base64
import os
from typing import Optional, Dict, Any, List
import pandas as pd
from autoop.core.ml.artifact import Artifact
import io


class Dataset(Artifact):
    def __init__(
        self,
        name: str,
        data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        asset_path: str = "",
        description: str = " ",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name=name,
            type="dataset",
            metadata=metadata,
            data=data,
            asset_path=asset_path,
            version=version,
            tags=tags,
        )
        self.description = description

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        description: str = " ",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
    ) -> "Dataset":
        csv_data = data.to_csv(index=False).encode()
        base64_data = base64.b64encode(csv_data)
        return Dataset(
            name=name,
            data=base64_data,
            asset_path=asset_path,
            metadata={"description": description},
            description=description,
            version=version,
            tags=tags,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset either from in-memory data or from the specified asset path.
        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        if self.data:
            csv_data = base64.b64decode(self.data).decode("utf-8")
            return pd.read_csv(io.StringIO(csv_data))

        if not os.path.exists(self.asset_path):
            raise FileNotFoundError(f"File not found: {self.asset_path}")

        with open(self.asset_path, "r") as f:
            data = json.load(f)
            csv_data = base64.b64decode(data["data"])
            return pd.read_csv(io.StringIO(csv_data.decode()))

    def save(self) -> None:
        """
        Saves the dataset to a JSON file at the specified asset path.
        The dataset is saved with its metadata and base64-encoded CSV data.
        """
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        with open(self.asset_path, "w") as f:
            json.dump(self.toJSON(), f, indent=4)

    def toJSON(self) -> Dict[str, Any]:
        """
        Converts the dataset to a dictionary for JSON serialization.
        Returns:
            Dict[str, Any]: The dataset as a dictionary.
        """
        artifact_dict = {
            "name": self.name,
            "type": self.type,
            "metadata": self.metadata,
            "asset_path": self.asset_path,
            "version": self.version,
            "tags": self.tags,
            "id": self.id,
        }
        if self.data:
            artifact_dict["data"] = base64.b64encode(self.data).decode("utf-8")
        return artifact_dict

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name}, "
            f"description={self.description}, "
            f"version={self.version}, "
            f"tags={self.tags}, "
            f"id={self.id})"
        )
