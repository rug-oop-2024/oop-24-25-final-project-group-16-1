from .artifact import Artifact
from typing import Optional, Dict, Any, List
import pandas as pd
import io


class Dataset(Artifact):
    def __init__(
        self,
        name: str,
        data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        asset_path: str = "",
        description: str = " ",
        version: str = "1.0",
        tags: Optional[List[str]] = None
    ):
        super().__init__(
            name=name,
            type="dataset",
            metadata=metadata,
            data=data,
            asset_path=asset_path,
            version=version,
            tags=tags
        )
        self.description = description

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        description: str = " ",
        version: str = "1.0",
        tags: Optional[List[str]] = None
    ) -> 'Dataset':
        csv_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            data=csv_data,
            asset_path=asset_path,
            metadata={'description': description},
            description=description,
            version=version,
            tags=tags
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset data into a pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame representation of the dataset.

        Raises:
            ValueError: If no data is available to read.
        """
        if self.data:
            csv_data = self.data.decode()
            return pd.read_csv(io.StringIO(csv_data))
        else:
            csv_data = self.metadata.get('data')
            if csv_data:
                return pd.read_csv(io.StringIO(csv_data))
            else:
                raise ValueError("No data available to read.")

    def save(self) -> None:
        """
        Saves the dataset data and metadata.
        """
        if not self.data and 'data' in self.metadata:
            self.data = self.metadata['data'].encode()
        super().save()

