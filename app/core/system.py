import os
import pickle
import streamlit as st
from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    A registry to manage storage and retrieval
    of artifacts in the AutoML system.
    """

    def __init__(self, database: Database, storage: Storage):
        """
        Initializes the ArtifactRegistry with a database and storage.

        Args:
            database (Database): The database to store artifact metadata.
            storage (Storage): The storage service for saving artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact, saving it in both storage and the database.

        Args:
            artifact (Artifact): The artifact to register.
        """
        if artifact.type == "pipeline":
            try:
                pipeline_dir = "assets/pipelines"
                pipeline_path = os.path.join(
                    pipeline_dir, f"{artifact.name}.pkl"
                )
                pipeline_data = {
                    "name": artifact.name,
                    "version": artifact.version,
                    "asset_path": artifact.asset_path,
                    "tags": artifact.tags,
                    "metadata": artifact.metadata,
                    "type": artifact.type,
                }
                with open(pipeline_path, "wb") as f:
                    pickle.dump(pipeline_data, f, protocol=5)
                st.success(f"Pipeline '{artifact.name}' saved successfully.")
            except Exception as e:
                st.error(f"Failed to save pipeline: {e}")
        elif artifact.type == "dataset":
            self._storage.save(artifact.data, artifact.asset_path)
            st.success(f"Dataset '{artifact.name}' saved successfully.")

        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all artifacts, optionally filtered by type.

        Args:
            type (str, optional): The type of artifact to filter by
            Defaults to None.

        Returns:
            List[Artifact]: A list of artifacts matching the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The unique identifier of the artifact.

        Returns:
            Artifact: The requested artifact instance.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by its ID, removing it from
        both storage and the database.

        Args:
            artifact_id (str): The unique identifier of the artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class representing the core AutoML
    system managing artifacts and storage.
    """

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the AutoML system with storage and database services.

        Args:
            storage (LocalStorage): Local storage instance
            for saving artifacts.
            database (Database): Database instance for artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieves the singleton instance of AutoMLSystem.

        Returns:
            AutoMLSystem: The shared instance of AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """Provides access to the artifact registry."""
        return self._registry
