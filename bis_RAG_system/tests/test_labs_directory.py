"""
Unit tests verifying labs_directory.json static dataset integrity,
LabLocator filter capabilities, and FastAPI /api/labs endpoint responses.
"""

import json
import os
import sys

import unittest
from pathlib import Path

# Add src, tests, and root to python path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [SRC_DIR, BASE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["USE_MOCK_RETRIEVAL"] = "true"

from lab_locator import LabLocator
from app import app
from fastapi.testclient import TestClient



class TestLabsDirectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labs_file = BASE_DIR / "labs_directory.json"
        cls.locator = LabLocator(labs_path=cls.labs_file)
        cls.client = TestClient(app)

    def test_01_labs_dataset_structure_and_counts(self):
        """Verify labs_directory.json exists, is valid JSON, and has status available with records."""
        self.assertTrue(self.labs_file.exists(), "labs_directory.json does not exist on disk.")

        with open(self.labs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("status"), "available")
        self.assertGreaterEqual(data.get("total_records", 0), 15)
        records = data.get("records", [])
        self.assertEqual(len(records), data.get("total_records"))

        required_keys = ["lab_id", "lab_name", "city", "state", "location", "address", "testing_scope", "is_recognized"]
        for idx, rec in enumerate(records):
            for k in required_keys:
                self.assertIn(k, rec, f"Record #{idx} missing required key '{k}'")
            self.assertTrue(rec["is_recognized"], f"Record #{idx} is_recognized must be True")
            self.assertIsInstance(rec["testing_scope"], list, f"Record #{idx} testing_scope must be a list")
            self.assertGreater(len(rec["testing_scope"]), 0, f"Record #{idx} testing_scope must not be empty")

    def test_02_lab_locator_search_by_state(self):
        """Verify LabLocator searches labs by state (e.g. Maharashtra, Tamil Nadu, Delhi)."""
        res = self.locator.search_labs("testing labs in Maharashtra", state="Maharashtra")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["source"], "labs_directory_json")
        self.assertFalse(res["fallback_used"])
        self.assertGreater(res["total_found"], 0)
        self.assertIn("Mumbai", res["formatted_text"])

    def test_03_lab_locator_search_by_standard_scope(self):
        """Verify LabLocator searches labs by Indian Standard testing scope (e.g. IS 1786)."""
        res = self.locator.search_labs("where can I test steel IS 1786")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["source"], "labs_directory_json")
        self.assertGreater(res["total_found"], 0)
        self.assertIn("IS 1786", res["formatted_text"])

    def test_04_api_labs_endpoint(self):
        """Verify GET /api/labs endpoint returns JSON lab search results."""
        response = self.client.get("/api/labs?query=Sahibabad")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data.get("status"), "success")
        self.assertGreater(json_data.get("total_found", 0), 0)
        self.assertIn("Sahibabad", json_data.get("formatted_text", ""))


if __name__ == "__main__":
    unittest.main()
