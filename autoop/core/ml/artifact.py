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
        _name (str): The name of the artifact.
        _type (str): The artifact type (e.g., 'model', 'dataset').
        _metadata (Dict[str, Any]): Additional metadata about the artifact.
        _data (Optional[bytes]): Binary data associated with the artifact.
        _asset_path (str): Path where artifact data is stored.
        _version (str): Artifact version identifier.
        _tags (List[str]): List of tags associated with the artifact.
        _id (str): Unique identifier created from name and version.
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
        self._name = name
        self._type = type
        self._metadata = {} if metadata is None else metadata
        self._data = data
        self._asset_path = asset_path
        self._version = version
        self._tags = tags or []
        self._id = f"{self._name}_{self._version}"

    @property
    def name(self) -> str:
        """
        Gets the name of the artifact.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Sets the name of the artifact.
        """
        self._name = value

    @property
    def type(self) -> str:
        """
        Gets the type of the artifact.
        """
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """
        Sets the type of the artifact.
        """
        self._type = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Gets the metadata of the artifact.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """
        Sets the metadata of the artifact.
        """
        self._metadata = value

    @property
    def data(self) -> Optional[bytes]:
        """
        Gets the data of the artifact.
        """
        return self._data

    @data.setter
    def data(self, value: Optional[bytes]) -> None:
        """
        Sets the data of the artifact.
        """
        self._data = value

    @property
    def asset_path(self) -> str:
        """
        Gets the asset path where the artifact is stored.
        """
        return self._asset_path

    @asset_path.setter
    def asset_path(self, value: str) -> None:
        """
        Sets the asset path for the artifact.
        """
        self._asset_path = value

    @property
    def version(self) -> str:
        """
        Gets the version of the artifact.
        """
        return self._version

    @version.setter
    def version(self, value: str) -> None:
        """
        Sets the version of the artifact.
        """
        self._version = value

    @property
    def tags(self) -> List[str]:
        """
        Gets the tags associated with the artifact.
        """
        return self._tags

    @tags.setter
    def tags(self, value: List[str]) -> None:
        """
        Sets the tags for the artifact.
        """
        self._tags = value

    @property
    def id(self) -> str:
        """
        Gets the unique identifier of the artifact.
        """
        return self._id

    def read(self) -> pd.DataFrame:
        """
        Decodes and reads the data stored in the artifact into a
        Pandas DataFrame, which is provided with a Base64-encoded
        string or as raw binary data.

        Returns:
            pd.DataFrame: Data read from the artifact as a DataFrame.

        Raises:
            ValueError: If data is neither a Base64 string nor binary.
        """
        if isinstance(self._data, str):
            self._data = self._data.encode()

        if isinstance(self._data, bytes):
            try:
                decoded_data = base64.b64decode(self._data)
                return pd.read_csv(io.BytesIO(decoded_data))
            except Exception:
                return pd.read_csv(io.BytesIO(self._data))
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
        self._data = base64.b64encode(data).decode("utf-8")
        return self._data
