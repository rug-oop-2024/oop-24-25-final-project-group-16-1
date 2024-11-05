from typing import Optional, List
import pandas as pd
from autoop.core.ml.artifact import Artifact
import io


class Dataset(Artifact):
    def __init__(self, description: str = " ", *args, **kwargs) -> None:
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
        Reads the dataset either from in-memory data or
        from the specified asset path.
        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        csv_data = super().read().decode()
        return pd.read_csv(io.BytesIO(csv_data))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the dataseta.
        """
        csv_data = data.to_csv(index=False).encode()
        return super().save(csv_data)
