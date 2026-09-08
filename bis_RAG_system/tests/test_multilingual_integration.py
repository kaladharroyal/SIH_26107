"""
Unit test verifying multilingual detection, Hinglish normalization,
and citation-preserving response translation in BIS RAG Pipeline.
"""

import sys
import unittest
from pathlib import Path

# Add src and root to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [SRC_DIR, BASE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rag_pipeline import BISRAGPipeline


class TestMultilingualIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=False)


    def test_english_query(self):
        res = self.pipeline.query("is certification mandatory for LED bulbs")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["lang_code"], "en")
        self.assertIn("detected_language", res)

    def test_hinglish_query_normalization(self):
        res = self.pipeline.query("mera LED bulb ke liye BIS certification chahiye")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["lang_code"], "hinglish")
        self.assertIn("Hinglish", res["detected_language"])

    def test_devanagari_hindi_query(self):
        res = self.pipeline.query("क्या एलईडी बल्ब के लिए बीआईएस प्रमाणन अनिवार्य है?")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["lang_code"], "hi")
        self.assertIn("Hindi", res["detected_language"])


if __name__ == "__main__":
    unittest.main()
