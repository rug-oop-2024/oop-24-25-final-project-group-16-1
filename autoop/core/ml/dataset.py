from typing import Optional, List
import pandas as pd
from autoop.core.ml.artifact import Artifact
import io
import base64


class Dataset(Artifact):
    """
    The Dataset class represents a data artifact that extends
    the functionality of the Artifact class for datasets.
    It also provides methods for creating datasets from DataFrames
    and reading/saving data with Base64 encoding for serialization.

    Attributes:
        description (str): A brief description of the dataset's contents.
    """

    def __init__(self, description: str = " ", *args, **kwargs) -> None:
        """
        Initializes the Dataset instance with a description and
        properties inherited from Artifact.

        Args:
            description (str): A description of the
            dataset's content.
        """
        super().__init__(type="dataset", *args, **kwargs)
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
        """
        Creates a Dataset instance from a Pandas DataFrame by encoding
        it as Base64 CSV data.

        Args:
            data (pd.DataFrame): The DataFrame being saved as the dataset.
            name (str): The name of the dataset.
            asset_path (str): Path where the dataset is stored.
            description (str): Description of the dataset.
            version (str): Version of the dataset.
            tags (Optional[List[str]]): List of tags for categorization.

        Returns:
            Dataset: An instance of the Dataset class
            containing the encoded data.
        """
        csv_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            data=csv_data,
            asset_path=asset_path,
            metadata={"description": description},
            description=description,
            version=version,
            tags=tags,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset from Base64-encoded or binary CSV
        data and decodes it into a Pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a DataFrame.

        Raises:
            ValueError: If data is neither Base64-encoded string nor binary.
        """
        if isinstance(self.data, str):
            decoded_data = base64.b64decode(self.data)
        elif isinstance(self.data, bytes):
            decoded_data = self.data
        else:
            raise ValueError(
                "Data is neither a valid Base64 string nor binary data."
            )

        return pd.read_csv(io.BytesIO(decoded_data))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the dataset by encoding it as Base64 CSV data.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            bytes: The encoded dataset in bytes for storage.
        """
        csv_data = data.to_csv(index=False).encode()
        return super().save(csv_data)
