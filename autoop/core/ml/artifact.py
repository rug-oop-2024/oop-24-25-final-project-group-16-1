import binascii
from typing import Any, Dict, Optional, List
import os
import pandas as pd
import base64
import json
import io


class Artifact:
    def __init__(
        self,
        name: str,
        type: str,
        metadata: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        asset_path: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
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

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset either from in-memory data or from the specified asset path.
        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        if self.data:
            try:
                # Check if `self.data` is a string or bytes
                if isinstance(self.data, bytes):
                    csv_data = base64.b64decode(self.data).decode("utf-8")
                else:
                    raise ValueError("`self.data` is not in the expected base64 bytes format.")
                
                # Convert CSV data into a DataFrame
                df = pd.read_csv(io.StringIO(csv_data))
                print("Loaded DataFrame shape (in-memory):", df.shape)
                return df

            except (binascii.Error, UnicodeDecodeError) as e:
                print("Error decoding base64 data:", e)
                print("Problematic data:", self.data[:50])
                raise ValueError("Failed to decode base64 data in `self.data`.")

        if not os.path.exists(self.asset_path):
            raise FileNotFoundError(f"File not found: {self.asset_path}")

        with open(self.asset_path, "r") as f:
            data = json.load(f)
            csv_data = base64.b64decode(data["data"]).decode("utf-8")
            df = pd.read_csv(io.StringIO(csv_data))
            print("Loaded DataFrame shape (from file):", df.shape)
            return df


    def save(self) -> None:
        """
        Save the artifact's data to the specified asset path.
        Raises an exception if the directory does not exist.
        """
        os.makedirs(os.path.dirname(self.asset_path), exist_ok=True)
        with open(self.asset_path, "wb") as file:
            file.write(self.data)
