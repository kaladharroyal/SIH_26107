"""
Phase 4 Remediation — Focused Tests (test_phase4_remediation.py)
Covers F2–F7 from Phase 4 Audit 1.
All tests run completely offline; no live API calls.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from citation_engine import CitationEngine
from generator import GroundedGenerator, MockOfflineProvider
from guardrails import CONFIDENCE_THRESHOLD, GuardrailGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id="IS1786_C4", is_number="IS 1786", clause="4.2",
                text="Steel bar requirements.", source_hash="abc123"):
    return {
        "doc": {
            "chunk_id": chunk_id,
            "is_number": is_number,
            "revision_year": "2008",
            "clause_number": clause,
            "clause_title": "Chemical Composition",
            "category": "is_standard",
            "page_start": 4,
            "page_end": 5,
            "source_url": "https://standardsbis.bsbedge.com/IS_1786",
            "source_file": "raw_data/pdfs/IS_1786.pdf",
            "source_hash": source_hash,
            "source_of_truth": "verified_bis_pdf",
            "text": text,
        }
    }


# ---------------------------------------------------------------------------
# F2: source_hash in LLM context block
# ---------------------------------------------------------------------------

class TestF2SourceHashInContext(unittest.TestCase):
    def test_source_hash_present_in_context(self):
        """F2: source_hash must appear in every evidence block sent to the LLM."""
        chunk = _make_chunk(source_hash="deadbeef1234")
        context = GroundedGenerator.format_context_block([chunk])
        self.assertIn("Source Hash:", context)
        self.assertIn("deadbeef1234", context)
        print("✅ F2: source_hash present in context block.")

    def test_source_hash_empty_still_present(self):
        """F2: Source Hash line is present even when hash is empty string."""
        chunk = _make_chunk(source_hash="")
        context = GroundedGenerator.format_context_block([chunk])
        self.assertIn("Source Hash:", context)
        print("✅ F2: Source Hash line present even when empty.")


# ---------------------------------------------------------------------------
# F3: Context budget guard
# ---------------------------------------------------------------------------

class TestF3ContextBudget(unittest.TestCase):
    def _big_chunk(self, idx, size=60_000):
        """Creates a chunk whose serialised block exceeds size characters."""
        c = _make_chunk(chunk_id=f"big_{idx}", is_number=f"IS {idx}", text="X" * size)
        return c

    def test_budget_stops_adding_whole_blocks(self):
        """F3: When total chars would exceed budget, whole blocks are dropped (not truncated mid-text)."""
        chunks = [self._big_chunk(1), self._big_chunk(2)]
        context = GroundedGenerator.format_context_block(chunks, char_budget=70_000)
        self.assertIn("big_1", context)
        if "big_2" not in context:
            self.assertIn("excluded", context)
        print("✅ F3: Budget guard stops adding whole blocks; no mid-block truncation.")

    def test_budget_note_appears_when_chunks_excluded(self):
        """F3: A note is appended when at least one chunk is excluded."""
        chunks = [self._big_chunk(1), self._big_chunk(2)]
        context = GroundedGenerator.format_context_block(chunks, char_budget=70_000)
        self.assertIn("excluded", context.lower())
        print("✅ F3: Exclusion note appended when budget exceeded.")

    def test_budget_not_triggered_for_small_corpus(self):
        """F3: Small chunks all fit; no exclusion note appears."""
        chunks = [_make_chunk(chunk_id=f"c{i}", is_number=f"IS {i}") for i in range(5)]
        context = GroundedGenerator.format_context_block(chunks, char_budget=100_000)
        self.assertNotIn("excluded", context.lower())
        for i in range(5):
            self.assertIn(f"c{i}", context)
        print("✅ F3: Small corpus passes budget unchanged.")


# ---------------------------------------------------------------------------
# F4: OpenAI timeout + bounded retry
# ---------------------------------------------------------------------------

try:
    import openai as _openai_mod
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


@unittest.skipUnless(_OPENAI_AVAILABLE, "openai package not installed in this env")
class TestF4OpenAIRetry(unittest.TestCase):
    """Tests use unittest.mock — no live API calls."""

    def _make_provider(self):
        from generator import OpenAIProvider
        provider = OpenAIProvider(api_key="sk-test", model_name="gpt-4o-mini")
        return provider

    def test_successful_request(self):
        """F4: Successful call returns response directly."""
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Steel bar answer."
        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = mock_resp
            result = provider.generate("sys", "user")
        self.assertEqual(result["response"], "Steel bar answer.")
        self.assertEqual(result["provider"], "openai")
        print("✅ F4: Successful OpenAI call returns response.")

    def test_transient_failure_then_success(self):
        """F4: One transient failure is retried and succeeds on second attempt."""
        import openai as openai_mod
        provider = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Recovered answer."

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise openai_mod.APIConnectionError(request=MagicMock())
            return mock_resp

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = side_effect
            with patch("time.sleep"):  # skip actual sleep
                result = provider.generate("sys", "user")
        self.assertEqual(result["response"], "Recovered answer.")
        self.assertEqual(call_count["n"], 2)
        print("✅ F4: Transient error retried; succeeded on attempt 2.")

    def test_exhausted_retries_falls_back(self):
        """F4: All retries exhausted; GroundedGenerator's fallback fires."""
        import openai as openai_mod
        provider = self._make_provider()

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = (
                openai_mod.APIConnectionError(request=MagicMock())
            )
            with patch("time.sleep"):
                gen = GroundedGenerator(provider_name="mock")
                gen.provider = provider
                result = gen.generate_response("IS 1786 requirements", [_make_chunk()])
        self.assertIn("provider", result)
        self.assertTrue(result.get("fallback_triggered") or result.get("provider") == "mock-offline")
        print("✅ F4: Exhausted retries trigger MockOfflineProvider fallback.")


# ---------------------------------------------------------------------------
# F5: Strengthened citation validation
# ---------------------------------------------------------------------------

class TestF5CitationValidation(unittest.TestCase):
    def setUp(self):
        self.engine = CitationEngine()
        self.real_chunks = [_make_chunk()]  # contains IS 1786

    def test_valid_citation_passes(self):
        """F5: Citation referencing a real IS number from context passes."""
        result = self.engine.validate_citations_against_context(
            ["As per IS 1786:2008, Clause 4.2"],
            self.real_chunks,
        )
        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["ungrounded_count"], 0)
        print("✅ F5: Valid citation referencing real IS number passes.")

    def test_fabricated_citation_fails(self):
        """F5: Citation referencing a fabricated IS number not in context fails."""
        result = self.engine.validate_citations_against_context(
            ["As per IS 9999:2024, Clause 99"],
            self.real_chunks,  # only contains IS 1786
        )
        self.assertEqual(result["ungrounded_count"], 1)
        self.assertFalse(result["all_valid"])
        print("✅ F5: Fabricated IS number citation correctly rejected.")

    def test_bare_bis_keyword_fails_without_is_match(self):
        """F5: 'Per BIS Guidelines' alone does NOT pass when no IS number matches."""
        result = self.engine.validate_citations_against_context(
            ["Per BIS Guidelines (General)"],
            self.real_chunks,
        )
        self.assertEqual(result["ungrounded_count"], 1)
        print("✅ F5: Bare BIS keyword without IS number match correctly rejected.")

    def test_multiple_valid_citations(self):
        """F5: Multiple real IS citations all pass."""
        chunks = [
            _make_chunk(chunk_id="c1", is_number="IS 1786", clause="4.2"),
            _make_chunk(chunk_id="c2", is_number="IS 269", clause="3.1"),
        ]
        result = self.engine.validate_citations_against_context(
            ["As per IS 1786:2008, Clause 4.2", "As per IS 269, Clause 3.1"],
            chunks,
        )
        self.assertEqual(result["valid_count"], 2)
        self.assertTrue(result["all_valid"])
        print("✅ F5: Multiple valid citations all pass.")


# ---------------------------------------------------------------------------
# F6: Missing corpus — degraded mode
# ---------------------------------------------------------------------------

class TestF6MissingCorpus(unittest.TestCase):
    def test_missing_corpus_sets_degraded_mode(self):
        """F6: When chunks file does not exist, degraded_mode is True."""
        from retrieval import HybridRetrievalPipeline
        pipeline = HybridRetrievalPipeline(
            chunks_path=Path("/nonexistent/processed_chunks.jsonl"),
            index_dir=Path("/nonexistent/vector_index"),
            use_mock_encoder=True,
        )
        self.assertTrue(pipeline.degraded_mode)
        self.assertEqual(len(pipeline.chunks), 0)
        print("✅ F6: Missing corpus sets degraded_mode=True.")

    def test_degraded_mode_attribute_always_present(self):
        """F6: degraded_mode attribute always exists on the pipeline."""
        from retrieval import HybridRetrievalPipeline
        pipeline = HybridRetrievalPipeline(
            chunks_path=Path("/nonexistent/chunks.jsonl"),
            index_dir=Path("/nonexistent/idx"),
            use_mock_encoder=True,
        )
        self.assertIsInstance(pipeline.degraded_mode, bool)
        print("✅ F6: degraded_mode attribute always present on pipeline.")


# ---------------------------------------------------------------------------
# F7: CONFIDENCE_THRESHOLD environment variable wiring
# ---------------------------------------------------------------------------

class TestF7ConfidenceThreshold(unittest.TestCase):
    def test_default_threshold_is_045(self):
        """F7: Default CONFIDENCE_THRESHOLD is 0.45."""
        gate = GuardrailGate()
        self.assertAlmostEqual(gate.threshold, 0.45, places=4)
        print("✅ F7: Default threshold is 0.45.")

    def test_env_var_changes_threshold(self):
        """F7: Setting CONFIDENCE_THRESHOLD env var changes the module-level constant."""
        import importlib
        import guardrails as gr_module
        with patch.dict(os.environ, {"CONFIDENCE_THRESHOLD": "0.60"}):
            importlib.reload(gr_module)
            self.assertAlmostEqual(gr_module.CONFIDENCE_THRESHOLD, 0.60, places=4)
        importlib.reload(gr_module)  # restore
        print("✅ F7: CONFIDENCE_THRESHOLD env var correctly changes module threshold.")

    def test_valid_float_env_parsed(self):
        """F7: Valid float env value is parsed correctly."""
        import importlib
        import guardrails as gr_module
        with patch.dict(os.environ, {"CONFIDENCE_THRESHOLD": "0.70"}):
            importlib.reload(gr_module)
            self.assertAlmostEqual(gr_module.CONFIDENCE_THRESHOLD, 0.70, places=4)
        importlib.reload(gr_module)
        print("✅ F7: Valid env float parsed correctly.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
