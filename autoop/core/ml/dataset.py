from autoop.core.ml.artifact import Artifact
from typing import Optional, Dict, Any
import pandas as pd
import io


class Dataset(Artifact):
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None, asset_path: str = "", description: str = " "):
        # Call parent class constructor with the appropriate arguments
        super().__init__(name=name, artifact_type="dataset", metadata=metadata)
        self.asset_path = asset_path
        self.description = description

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, description: str = " "):
        return Dataset(
            name=name,
            asset_path=asset_path,
            metadata={'data': data.to_csv(index=False), 'description': description}
        )

    def read(self) -> pd.DataFrame:
        # Assuming the artifact's metadata contains the CSV data as a string
        csv_data = self.metadata['data']
        return pd.read_csv(io.StringIO(csv_data))

    def save(self, data: pd.DataFrame) -> None:
        # Save the data into the metadata field and persist the artifact
        self.metadata['data'] = data.to_csv(index=False)
        super().save()

