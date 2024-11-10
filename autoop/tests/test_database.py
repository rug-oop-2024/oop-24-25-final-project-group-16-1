import unittest
from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile


class TestDatabase(unittest.TestCase):
    """
    Unit tests for the Database class, covering CRUD operations,
    persistence, and refreshing data from storage.
    """

    def setUp(self):
        """
        Set up temporary local storage for testing purposes.
        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self):
        """
        Test that the Database instance is created successfully.
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self):
        """
        Test setting a key-value pair in the database and retrieving it.
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self):
        """
        Test deletion of a key-value pair from the database.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistence(self):
        """
        Test that data persists in storage across database instances.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self):
        """
        Test that refreshing a database instance reloads updated data.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self):
        """
        Test listing all key-value pairs in a specified collection.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        self.assertIn((key, value), self.db.list("collection"))
