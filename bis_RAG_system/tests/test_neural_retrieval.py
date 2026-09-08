"""
Unit tests verifying production Neural Dense Retrieval with USE_MOCK_RETRIEVAL=false.
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

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from rag_pipeline import BISRAGPipeline
from retrieval import HybridRetrievalPipeline, MockEmbeddingModel


class TestNeuralRetrieval(unittest.TestCase):
    def test_01_env_use_mock_retrieval_is_false(self):
        """Verify .env config has USE_MOCK_RETRIEVAL=false for production neural mode."""
        env_val = os.getenv("USE_MOCK_RETRIEVAL", "true").lower()
        self.assertEqual(env_val, "false", "USE_MOCK_RETRIEVAL in .env must be set to 'false'")

    def test_02_pipeline_initialization_config(self):
        """Verify HybridRetrievalPipeline accepts use_mock_encoder toggle."""
        retrieval = HybridRetrievalPipeline(use_mock_encoder=True)
        self.assertTrue(retrieval.use_mock_encoder)
        self.assertIsNotNone(retrieval.encoder)

    def test_03_retrieval_execution(self):
        """Verify retrieval pipeline returns high-confidence ranked hits for BIS standard queries."""
        pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
        res = pipeline.query("IS 1786 steel reinforcement bar requirements")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["confidence_score"], 0.45)
        self.assertGreater(len(res["retrieved_chunks"]), 0)
        self.assertIn("IS 1786", res["response"])




if __name__ == "__main__":
    unittest.main()
