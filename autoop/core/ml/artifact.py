import base64
import io
from typing import Any, Dict, Optional, List
import pandas as pd


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
        Reads the dataset either from in-memory data or
        from the specified asset path.

        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        if isinstance(self.data, str):
            self.data = self.data.encode()

        if isinstance(self.data, bytes):
            try:
                decoded_data = base64.b64decode(self.data)
                return pd.read_csv(io.BytesIO(decoded_data))
            except Exception:
                return pd.read_csv(io.BytesIO(self.data))
        else:
            raise ValueError("Data is neither a valid base64 string nor binary data.")

    def save(self, data: bytes) -> str:
        """
        Encodes and saves the artifact's data as a Base64 string.

        Args:
            data (bytes): The data to be saved.

        Returns:
            str: The encoded data as a Base64 string.
        """
        self.data = base64.b64encode(data).decode("utf-8")
        return self.data
