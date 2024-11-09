from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Custom exception for handling non-existent file paths in storage.
    """

    def __init__(self, path: str):
        """
        Initialize NotFoundError with the missing path.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class defining a storage interface.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load data from.

        Returns:
            bytes: Loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete data from.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all file paths under a given path.

        Args:
            path (str): Path to list contents of.

        Returns:
            List[str]: List of file paths.
        """
        pass


class LocalStorage(Storage):
    """
    Local file system storage for managing
    artifacts in a specified directory.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes LocalStorage with a specified base directory.

        Args:
            base_path (str): The base directory for storing files.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Saves data to the specified key in local storage.

        Args:
            data (bytes): The data to save.
            key (str): The storage key (relative path).
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads data from the specified key in local storage.

        Args:
            key (str): The storage key (relative path).

        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Deletes data at the specified key in local storage.

        Args:
            key (str): The storage key (relative path).
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        Lists all files under the given prefix in local storage.

        Args:
            prefix (str): Directory prefix to list contents of.

        Returns:
            List[str]: List of relative paths to each file.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [
            os.path.relpath(
                p, self._base_path
                ) for p in keys if os.path.isfile(p)
        ]

    def _assert_path_exists(self, path: str) -> None:
        """
        Raises an error if the specified path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Joins the base path with the provided relative path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
