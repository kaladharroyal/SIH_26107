"""
Unit tests verifying Frontend UI Language Selector in index.html
and FastAPI /api/chat multilingual selection integration.
"""

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

from app import app
from fastapi.testclient import TestClient


class TestUILanguageSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index_file = BASE_DIR / "index.html"
        cls.client = TestClient(app)

    def test_01_index_html_language_selector_markup(self):
        """Verify index.html contains id='language-select' and language options."""
        self.assertTrue(self.index_file.exists(), "index.html does not exist on disk.")

        with open(self.index_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        self.assertIn('id="language-select"', html_content)
        self.assertIn('value="auto"', html_content)
        self.assertIn('value="en"', html_content)
        self.assertIn('value="hi"', html_content)
        self.assertIn('value="hinglish"', html_content)
        self.assertIn('value="ta"', html_content)
        self.assertIn('value="te"', html_content)
        self.assertIn('value="bn"', html_content)

    def test_02_chat_endpoint_with_explicit_language_selection(self):
        """Verify POST /api/chat with explicit language selection (e.g. Hindi, Hinglish)."""
        # Hindi Request with target language override
        res_hi = self.client.post("/api/chat", json={"query": "is certification mandatory for LED bulbs", "language": "hi"})
        self.assertEqual(res_hi.status_code, 200)
        data_hi = res_hi.json()
        self.assertEqual(data_hi.get("lang_code"), "hi")

        # Hinglish Request with target language override
        res_hinglish = self.client.post("/api/chat", json={"query": "is certification mandatory for LED bulbs", "language": "hinglish"})
        self.assertEqual(res_hinglish.status_code, 200)
        data_hinglish = res_hinglish.json()
        self.assertEqual(data_hinglish.get("lang_code"), "hinglish")

        # Native Devanagari Hindi Query
        res_native = self.client.post("/api/chat", json={"query": "क्या एलईडी बल्ब के लिए बीआईएस प्रमाणन अनिवार्य है?"})
        self.assertEqual(res_native.status_code, 200)
        data_native = res_native.json()
        self.assertEqual(data_native.get("lang_code"), "hi")
        self.assertIn("Hindi", data_native.get("detected_language", ""))



if __name__ == "__main__":
    unittest.main()
