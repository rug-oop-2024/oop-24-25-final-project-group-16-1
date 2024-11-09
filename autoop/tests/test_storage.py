import unittest
from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile
import os


class TestStorage(unittest.TestCase):

    def setUp(self):
        """
        Set up a temporary directory and LocalStorage instance for testing.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        """
        Test whether the LocalStorage instance initializes correctly.
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        """
        Test storing and loading data in LocalStorage.
        """
        key = f"test{os.sep}path"
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)

        other_key = f"test{os.sep}otherpath"
        with self.assertRaises(NotFoundError):
            self.storage.load(other_key)

    def test_delete(self):
        """
        Test that deletes data from LocalStorage and verifies if is removed.
        """
        key = f"test{os.sep}path"
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        self.storage.save(test_bytes, key)
        self.storage.delete(key)

        with self.assertRaises(NotFoundError):
            self.storage.load(key)

    def test_list(self):
        """
        Test listing all stored keys within a
        specific directory in LocalStorage.
        """
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [
            f"test{os.sep}{random.randint(0, 100)}" for _ in range(10)
        ]

        for key in random_keys:
            self.storage.save(test_bytes, key)

        keys = self.storage.list("test")
        keys = [f"{os.sep}".join(key.split(f"{os.sep}")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
