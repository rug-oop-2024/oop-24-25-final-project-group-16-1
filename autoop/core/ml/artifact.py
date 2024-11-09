import base64
import io
from typing import Any, Dict, Optional, List
import pandas as pd


class Artifact:
    """
    A class that represents an artifact, which can be any form of data
    that needs to be serialized and stored.
    Each artifact has associated metadata, a data source, and can be encoded
    and decoded.

    Attributes:
        name (str): The name of the artifact.
        type (str): The artifact type (e.g., 'model', 'dataset').
        metadata (Dict[str, Any]): Additional metadata about the artifact.
        data (Optional[bytes]): Binary data associated with the artifact.
        asset_path (str): Path where artifact data is stored.
        version (str): Artifact version identifier.
        tags (List[str]): List of tags associated with the artifact.
        id (str): Unique identifier created from name and version.
    """

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
        Initiates the Artifact instance with essential properties including
        name, type, metadata, and an optional binary data payload.

        Args:
            name (str): Name of the artifact.
            type (str): Type of the artifact (e.g., 'model', 'dataset').
            metadata (Optional[Dict[str, Any]]): Metadata about artifact.
            data (Optional[bytes]): Binary data associated with artifact.
            asset_path (str): Path where the artifact's data is stored.
            version (str): Version of the artifact (default is '1.0.0').
            tags (Optional[List[str]]): List of tags for categorization.
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
        Decodes and reads the data stored in the artifact into a
        Pandas DataFrame, which is provided with a Base64-ecoded
        string or as raw binary data.

        Returns:
            pd.DataFrame: Data read from the artifact as a DataFrame.

        Raises:
            ValueError: If data is neither a Base64 string nor binary.
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
            raise ValueError(
                "Data is neither a valid base64 string nor binary data."
            )

    def save(self, data: bytes) -> str:
        """
        Encodes binary data to a Base64 string for storage. This allows
        the artifact's data to be stored in a format for text-based
        databases or serialization methods.

        Args:
            data (bytes): The binary data to encode and save.

        Returns:
            str: Base64-encoded data as a string, ready for storage.
        """
        self.data = base64.b64encode(data).decode("utf-8")
        return self.data
