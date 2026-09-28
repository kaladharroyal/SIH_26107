"""
Automated Unit/Integration Test Suite for Corpus Recovery Engine (test_recover_corpus.py)
Validates all recovery engine edge cases before executing full canonical crawl:
- Valid PDF retrieval and SHA-256 verification
- Hash mismatch quarantine handling
- Invalid/corrupt PDF quarantine handling
- HTTP error quarantine handling
- Resume & idempotency behavior
- Registry and log structure integrity
"""

import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
import pypdf
import requests
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "bis_RAG_system" / "src" / "ingestion") not in sys.path:
    sys.path.insert(0, str(root_dir / "bis_RAG_system" / "src" / "ingestion"))

from bis_RAG_system.src.ingestion.recover_corpus import CorpusRecoveryEngine


class TestCorpusRecoveryEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw_data"
        self.manifest_path = self.test_dir / "recovery_manifest.jsonl"
        self.log_path = self.test_dir / "recovery_log.jsonl"
        self.registry_path = self.test_dir / "corpus_registry.jsonl"
        self.quarantine_log_path = self.test_dir / "quarantine_log.jsonl"

        # Generate a small valid synthetic PDF in memory
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        self.valid_pdf_bytes = buf.getvalue()
        self.valid_sha256 = hashlib.sha256(self.valid_pdf_bytes).hexdigest()

        # Engine instance
        self.engine = CorpusRecoveryEngine(
            manifest_path=self.manifest_path,
            raw_dir=self.raw_dir,
            log_path=self.log_path,
            registry_path=self.registry_path,
            quarantine_log_path=self.quarantine_log_path,
            workers=2,
            timeout=5
        )

    def tearDown(self):
        shutil.rmtree(str(self.test_dir), ignore_errors=True)

    def test_01_valid_recovery(self):
        """Test successful download, PDF validation, hash check, and atomic move."""
        rec = {
            "record_id": "REC_TEST_001",
            "document_id": "DOC_TEST_001",
            "source_url": "https://example.com/test.pdf",
            "expected_sha256": self.valid_sha256,
            "filename": "test_valid.pdf",
            "domain": "standards",
            "classification": "PDF_STANDARD",
            "title": "Test Valid Standard",
            "standard_number": "IS 100",
            "canonical": True
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.iter_content.return_value = [self.valid_pdf_bytes]

        with patch.object(requests.Session, "get", return_value=mock_resp):
            res = self.engine.process_record(rec)

        self.assertEqual(res["recovery_status"], "RECOVERED")
        self.assertEqual(res["validation_status"], "VALID")
        self.assertEqual(res["actual_sha256"], self.valid_sha256)
        self.assertGreater(res["page_count"], 0)

        # Check file exists on disk in domain directory
        target_file = self.raw_dir / "pdfs" / "standards" / "test_valid.pdf"
        self.assertTrue(target_file.exists())

    def test_02_hash_mismatch_quarantine(self):
        """Test that a file with unexpected hash is rejected and moved to quarantine."""
        rec = {
            "record_id": "REC_TEST_002",
            "document_id": "DOC_TEST_002",
            "source_url": "https://example.com/test_mismatch.pdf",
            "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "filename": "test_mismatch.pdf",
            "domain": "amendments",
            "classification": "PDF_AMENDMENT",
            "title": "Test Mismatch",
            "standard_number": "N/A",
            "canonical": True
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.iter_content.return_value = [self.valid_pdf_bytes]

        with patch.object(requests.Session, "get", return_value=mock_resp):
            res = self.engine.process_record(rec)

        self.assertEqual(res["recovery_status"], "QUARANTINED")
        self.assertEqual(res["failure_reason"], "HASH_MISMATCH")

        # Destination file should NOT exist
        target_file = self.raw_dir / "pdfs" / "amendments" / "test_mismatch.pdf"
        self.assertFalse(target_file.exists())

        # Quarantine file should exist
        quarantine_file = self.raw_dir / "quarantine" / "REC_TEST_002_MISMATCH_test_mismatch.pdf"
        self.assertTrue(quarantine_file.exists())

    def test_03_corrupt_pdf_quarantine(self):
        """Test that an HTML error page masquerading as PDF is caught and quarantined."""
        rec = {
            "record_id": "REC_TEST_003",
            "document_id": "DOC_TEST_003",
            "source_url": "https://example.com/test_html.pdf",
            "expected_sha256": "somehash",
            "filename": "test_html.pdf",
            "domain": "certification",
            "classification": "PDF_CERTIFICATION",
            "title": "Test HTML",
            "standard_number": "N/A",
            "canonical": True
        }

        html_bytes = b"<html><head><title>404 Not Found</title></head><body>Error</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.iter_content.return_value = [html_bytes]

        with patch.object(requests.Session, "get", return_value=mock_resp):
            res = self.engine.process_record(rec)

        self.assertEqual(res["recovery_status"], "QUARANTINED")
        self.assertTrue("NOT_A_PDF" in res["failure_reason"])

    def test_04_http_404_quarantine(self):
        """Test that HTTP 404 responses are recorded with status and failure reason."""
        rec = {
            "record_id": "REC_TEST_004",
            "document_id": "DOC_TEST_004",
            "source_url": "https://example.com/missing.pdf",
            "expected_sha256": "somehash",
            "filename": "missing.pdf",
            "domain": "general",
            "classification": "PDF_GENERAL",
            "title": "Missing File",
            "standard_number": "N/A",
            "canonical": True
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch.object(requests.Session, "get", return_value=mock_resp):
            res = self.engine.process_record(rec)

        self.assertEqual(res["recovery_status"], "QUARANTINED")
        self.assertEqual(res["failure_reason"], "HTTP_404")

    def test_05_resumability_and_idempotency(self):
        """Test that an already existing, valid PDF is verified without re-downloading."""
        target_dir = self.raw_dir / "pdfs" / "consumer"
        target_file = target_dir / "test_cached.pdf"
        with open(target_file, "wb") as f:
            f.write(self.valid_pdf_bytes)

        rec = {
            "record_id": "REC_TEST_005",
            "document_id": "DOC_TEST_005",
            "source_url": "https://example.com/test_cached.pdf",
            "expected_sha256": self.valid_sha256,
            "filename": "test_cached.pdf",
            "domain": "consumer",
            "classification": "PDF_CONSUMER",
            "title": "Test Cached Consumer",
            "standard_number": "N/A",
            "canonical": True
        }

        # session.get should NOT be called
        with patch.object(requests.Session, "get") as mock_get:
            res = self.engine.process_record(rec)
            mock_get.assert_not_called()

        self.assertEqual(res["recovery_status"], "RECOVERED")
        self.assertTrue(res["cached"])
        self.assertEqual(res["actual_sha256"], self.valid_sha256)

    def test_06_incremental_logging_and_idempotence(self):
        """Test that registry and quarantine entries are written immediately to disk."""
        # 1. Test immediate registry emission on RECOVERED
        rec_valid = {
            "record_id": "REC_TEST_006_A",
            "document_id": "DOC_TEST_006_A",
            "source_url": "https://example.com/test_valid.pdf",
            "expected_sha256": self.valid_sha256,
            "filename": "test_valid_6a.pdf",
            "domain": "standards",
            "classification": "STANDARDS",
            "title": "Test 6A",
            "standard_number": "IS-1",
            "canonical": True
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.iter_content = lambda chunk_size: [self.valid_pdf_bytes]

        with patch.object(requests.Session, "get", return_value=mock_resp):
            self.engine.process_record(rec_valid)

        # Check that registry file has this entry immediately
        self.assertTrue(self.registry_path.exists())
        with open(self.registry_path, "r", encoding="utf-8") as f:
            registry_lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(registry_lines), 1)
        self.assertEqual(registry_lines[0]["document_id"], "DOC_TEST_006_A")

        # Rerun process_record: idempotency should ensure no duplicate line
        with patch.object(requests.Session, "get") as mock_get:
            self.engine.process_record(rec_valid)
            mock_get.assert_not_called()

        with open(self.registry_path, "r", encoding="utf-8") as f:
            registry_lines_after = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(registry_lines_after), 1)

        # 2. Test immediate quarantine emission on FAILED
        rec_failed = {
            "record_id": "REC_TEST_006_B",
            "document_id": "DOC_TEST_006_B",
            "source_url": "https://example.com/missing_6b.pdf",
            "expected_sha256": "somehash",
            "filename": "missing_6b.pdf",
            "domain": "general",
            "classification": "GENERAL",
            "title": "Test 6B",
            "standard_number": "N/A",
            "canonical": True
        }
        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404
        mock_resp_404.headers = {"Content-Type": "text/html"}

        with patch.object(requests.Session, "get", return_value=mock_resp_404):
            self.engine.process_record(rec_failed)

        # Check that quarantine log has this entry immediately
        self.assertTrue(self.quarantine_log_path.exists())
        with open(self.quarantine_log_path, "r", encoding="utf-8") as f:
            q_lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(q_lines), 1)
        self.assertEqual(q_lines[0]["record_id"], "REC_TEST_006_B")


if __name__ == "__main__":
    unittest.main()

