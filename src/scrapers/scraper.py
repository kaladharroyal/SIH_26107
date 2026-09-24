"""
BIS Data Scraper — Phase 1, Step 2

Scrapes two categories of sources from bis.gov.in and related BIS portals:
  1. STRUCTURED sources: product-code -> standard mapping tables (CRS/QCO lists),
     lab directories, scheme fee schedules. These are usually HTML tables or
     downloadable CSV/XLS files.
  2. UNSTRUCTURED sources: IS standard PDFs, scheme explainer PDFs, FAQ pages.

Design notes:
  - BIS does not expose a public API, so this relies on scraping HTML + PDF links.
    Expect page structures to change — this script is written to fail loudly
    (log + skip) rather than silently, so gaps in the corpus are visible.
  - Every downloaded item is recorded in a manifest (manifest.jsonl) with its
    source URL, fetch timestamp, and a content hash — this is what lets you
    detect standard revisions later (re-scrape, diff hashes, flag changed docs).
  - Respect robots.txt and add a delay between requests — this is a government
    site serving many users, not a target to hammer.

Run standalone:
    python scraper.py --config sources.yaml --out ./raw_data
"""

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bis_scraper")

REQUEST_DELAY_SECONDS = 1.5  # be polite to a gov site
USER_AGENT = "BIS-Assistant-Research-Bot/0.1 (contact: <your-email>)"


@dataclass
class ScrapedItem:
    source_url: str
    local_path: str
    content_type: str          # "pdf" | "html_table" | "html_page"
    category: str              # "is_standard" | "crs_qco" | "lab_directory" | "scheme_doc" | "faq"
    fetched_at: str
    content_hash: str
    extra_metadata: dict


class BISScraper:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self.warning_log_path = Path("warning_downloads.txt")

    def _hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save(self, content: bytes, filename: str) -> Path:
        path = self.out_dir / filename
        path.write_bytes(content)
        return path

    def _record(self, item: ScrapedItem):
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item)) + "\n")

    def _record_warning(self, url: str, reason: str):
        with open(self.warning_log_path, "a", encoding="utf-8") as f:
            f.write(f"- {url} (Reason: {reason})\n")

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY_SECONDS)
            return BeautifulSoup(resp.content, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Failed to fetch {url}: {e}")
            self._record_warning(url, str(e))
            return None

    def scrape_pdf_links(self, listing_url: str, category: str):
        """
        Generic pattern: a listing page (e.g. 'List of Indian Standards')
        contains <a href="*.pdf"> links. Download each, hash it, record metadata.
        """
        soup = self.fetch_page(listing_url)
        if soup is None:
            return

        pdf_links = [
            urljoin(listing_url, a["href"])
            for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        ]
        log.info(f"Found {len(pdf_links)} PDF links on {listing_url}")

        for link in pdf_links:
            try:
                resp = self.session.get(link, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                log.warning(f"Skipping {link}: {e}")
                self._record_warning(link, str(e))
                continue

            content_hash = self._hash(resp.content)
            filename = f"{content_hash[:12]}_{Path(urlparse(link).path).name}"
            self._save(resp.content, filename)

            self._record(ScrapedItem(
                source_url=link,
                local_path=str(self.out_dir / filename),
                content_type="pdf",
                category=category,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                content_hash=content_hash,
                extra_metadata={"listing_page": listing_url},
            ))
            time.sleep(REQUEST_DELAY_SECONDS)

    def scrape_direct_pdf(self, pdf_url: str, category: str):
        """
        Download a direct PDF URL directly without searching for listing links.
        """
        try:
            resp = self.session.get(pdf_url, timeout=30)
            resp.raise_for_status()
            content_hash = self._hash(resp.content)
            filename = f"{content_hash[:12]}_{Path(urlparse(pdf_url).path).name}"
            self._save(resp.content, filename)
            self._record(ScrapedItem(
                source_url=pdf_url,
                local_path=str(self.out_dir / filename),
                content_type="pdf",
                category=category,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                content_hash=content_hash,
                extra_metadata={"direct_pdf": True},
            ))
            log.info(f"Saved direct PDF {pdf_url} -> {filename}")
        except requests.RequestException as e:
            log.warning(f"Failed to fetch direct PDF {pdf_url}: {e}")

    def scrape_html_page(self, url: str, category: str, content_selector: str = "article, main, .content, #content"):
        """
        For informational/FAQ pages (hallmarking overview, product certification
        FAQ, consumer protection, etc.) that are prose or Q&A, not tables.
        """
        soup = self.fetch_page(url)
        if soup is None:
            return

        container = None
        selectors = [content_selector, ".entry-content", ".td-page-content", "article", "main", "#content", "#main"]
        for sel in selectors:
            if not sel:
                continue
            c = soup.select_one(sel)
            if c and len(c.get_text(strip=True)) > 50:
                container = c
                break

        if container is None:
            container = soup.body

        if container is None:
            log.warning(f"No content container found on {url}")
            return

        text = container.get_text("\n", strip=True)

        # Try to split into Q&A pairs — BIS FAQs typically use "Q1.", "Q 1.",
        # "1." style numbering for questions.
        # Handles the real variants seen on bis.gov.in FAQ pages: "Q 1 What...",
        # "Q.2 I am...", "Q 12: Which...", "Q17. What...", "Q. 25 What...".
        qa_pattern = re.compile(r"(?:^|\n)\s*Q\s*\.?\s*\d+\s*[:.]?\s*")
        segments = [s.strip() for s in qa_pattern.split(text) if s.strip()]

        if len(segments) > 2:
            payload = [{"qa_text": seg} for seg in segments]
        else:
            payload = [{"page_text": text}]

        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        content_hash = self._hash(content)
        filename = f"{content_hash[:12]}_{category}.json"
        self._save(content, filename)

        self._record(ScrapedItem(
            source_url=url,
            local_path=str(self.out_dir / filename),
            content_type="html_page",
            category=category,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            content_hash=content_hash,
            extra_metadata={"segment_count": len(payload), "split_as_qa": len(segments) > 2},
        ))
        log.info(f"Saved {url} -> {filename} ({len(payload)} segments)")

    def scrape_html_table(self, url: str, category: str, table_selector: str = "table"):
        """
        For structured data like CRS/QCO product lists or lab directories,
        which are often rendered as plain HTML tables rather than PDFs.
        Saves as JSON rows so it slots directly into the Postgres loader.
        """
        soup = self.fetch_page(url)
        if soup is None:
            return

        table = soup.select_one(table_selector)
        if table is None:
            log.warning(f"No table matched '{table_selector}' on {url}")
            return

        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(dict(zip(headers, cells)))

        content = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        content_hash = self._hash(content)
        filename = f"{content_hash[:12]}_{category}.json"
        self._save(content, filename)

        self._record(ScrapedItem(
            source_url=url,
            local_path=str(self.out_dir / filename),
            content_type="html_table",
            category=category,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            content_hash=content_hash,
            extra_metadata={"row_count": len(rows), "headers": headers},
        ))
        log.info(f"Saved {len(rows)} rows from {url} -> {filename}")


def load_sources(config_path: str) -> list[dict]:
    """
    sources.yaml format:
      - url: "https://www.bis.gov.in/standards-list"
        type: pdf_listing
        category: is_standard
      - url: "https://www.bis.gov.in/crs-products"
        type: html_table
        category: crs_qco
        table_selector: "table.crs-list"
    """
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to sources.yaml")
    parser.add_argument("--out", default="./raw_data")
    args = parser.parse_args()

    scraper = BISScraper(args.out)
    sources = load_sources(args.config)

    for src in sources:
        log.info(f"Processing source: {src['url']} ({src['type']})")
        if src["url"].lower().endswith(".pdf") or src["type"] == "direct_pdf":
            scraper.scrape_direct_pdf(src["url"], src["category"])
        elif src["type"] == "pdf_listing":
            scraper.scrape_pdf_links(src["url"], src["category"])
        elif src["type"] == "html_table":
            scraper.scrape_html_table(src["url"], src["category"], src.get("table_selector", "table"))
        elif src["type"] == "html_page":
            scraper.scrape_html_page(src["url"], src["category"], src.get("content_selector", "article, main, .content, #content"))
        else:
            log.warning(f"Unknown source type: {src['type']}")


if __name__ == "__main__":
    main()
