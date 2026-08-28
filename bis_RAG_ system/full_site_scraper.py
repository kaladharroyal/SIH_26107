"""
Full BIS Site & Multi-Domain Automated Crawler & Downloader
Crawls bis.gov.in, crsbis.in, standards.bis.gov.in, standardsbis.bsbedge.com
Downloads every HTML page and every PDF file (large streams in 1MB chunks).
Automatically logs any 403, 404, or failed download URLs into warning_downloads.txt.
"""

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("full_site_scraper")

OUT_DIR = Path("./raw_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"
WARNING_LOG_PATH = Path("warning_downloads.txt")

ALLOWED_DOMAINS = [
    "bis.gov.in",
    "www.bis.gov.in",
    "crsbis.in",
    "www.crsbis.in",
    "standards.bis.gov.in",
    "standardsbis.bsbedge.com",
    "irportal.bis.gov.in",
]


def create_resilient_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bis.gov.in/",
    })
    return session


def record_warning(url: str, reason: str):
    try:
        with open(WARNING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"- {url} (Reason: {reason})\n")
    except Exception as e:
        log.error(f"Could not record warning: {e}")


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    return any(domain in parsed.netloc.lower() for domain in ALLOWED_DOMAINS)


class FullSiteScraper:
    def __init__(self):
        self.session = create_resilient_session()
        self.visited_urls = set()
        self.downloaded_filenames = set()
        self._load_existing_manifest()

    def _load_existing_manifest(self):
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        src_url = entry.get("source_url")
                        if src_url:
                            self.visited_urls.add(src_url)
                        loc_path = entry.get("local_path")
                        if loc_path:
                            self.downloaded_filenames.add(Path(loc_path).name)
                log.info(f"Loaded manifest resume state: {len(self.visited_urls)} URLs already downloaded/visited.")
            except Exception as e:
                log.warning(f"Could not load manifest resume state: {e}")

        # Also register existing files in OUT_DIR
        for p in OUT_DIR.glob("*.*"):
            if p.name != "manifest.jsonl" and p.stat().st_size > 0:
                self.downloaded_filenames.add(p.name)

    def hash_content(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def record_manifest(self, item: dict):
        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def download_pdf(self, pdf_url: str, category: str = "pdf_document"):
        if pdf_url in self.visited_urls:
            log.info(f"Skipping already visited PDF URL: {pdf_url}")
            return
        self.visited_urls.add(pdf_url)

        filename_raw = Path(urlparse(pdf_url).path).name
        if not filename_raw.lower().endswith(".pdf"):
            filename_raw += ".pdf"

        # Fast skip check if file matching this name exists
        for existing in self.downloaded_filenames:
            if existing.endswith(filename_raw) and not existing.startswith("temp_"):
                log.info(f"Skipping already downloaded PDF file ({existing}) for URL: {pdf_url}")
                return

        log.info(f"Downloading PDF: {pdf_url}")
        try:
            resp = self.session.get(pdf_url, stream=True, timeout=(15, 300))
            if resp.status_code != 200:
                log.warning(f"HTTP {resp.status_code} for PDF: {pdf_url}")
                record_warning(pdf_url, f"HTTP {resp.status_code} Status Error")
                return

            sha256 = hashlib.sha256()
            temp_file = OUT_DIR / f"temp_{filename_raw}"
            total_bytes = 0

            with open(temp_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        sha256.update(chunk)
                        total_bytes += len(chunk)

            content_hash = sha256.hexdigest()
            final_filename = f"{content_hash[:12]}_{filename_raw}"
            final_path = OUT_DIR / final_filename

            if temp_file.exists():
                temp_file.replace(final_path)
            self.downloaded_filenames.add(final_filename)

            self.record_manifest({
                "source_url": pdf_url,
                "local_path": str(final_path),
                "content_type": "pdf",
                "category": category,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
                "extra_metadata": {"size_bytes": total_bytes},
            })

            log.info(f"SUCCESS PDF: Saved {pdf_url} -> {final_filename} ({total_bytes / (1024*1024):.2f} MB)")

        except Exception as e:
            log.error(f"Failed PDF download {pdf_url}: {e}")
            record_warning(pdf_url, f"Download Error: {e}")

    def scrape_page(self, url: str, category: str = "webpage"):
        if url in self.visited_urls or not is_allowed_url(url):
            return
        self.visited_urls.add(url)

        log.info(f"Scraping HTML: {url}")
        try:
            resp = self.session.get(url, timeout=25)
            if resp.status_code != 200:
                log.warning(f"HTTP {resp.status_code} for {url}")
                record_warning(url, f"HTTP {resp.status_code} Status Error")
                return

            content = resp.content
            content_hash = self.hash_content(content)

            soup = BeautifulSoup(content, "html.parser")

            # Extract text content
            text_blocks = [p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li", "td"]) if p.get_text(strip=True)]
            full_text = "\n".join(text_blocks)

            local_filename = f"{content_hash[:12]}_{category}.json"
            local_path = OUT_DIR / local_filename

            page_data = [{
                "page_url": url,
                "page_text": full_text,
            }]

            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)

            self.record_manifest({
                "source_url": url,
                "local_path": str(local_path),
                "content_type": "html_page",
                "category": category,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
                "extra_metadata": {"char_count": len(full_text)},
            })

            log.info(f"SUCCESS HTML: Saved {url} -> {local_filename} ({len(full_text)} chars)")

            # Extract PDF and child HTML links
            pdf_links = []
            html_links = []

            for a in soup.find_all("a", href=True):
                full_link = urljoin(url, a["href"]).split("#")[0]
                if full_link.lower().endswith(".pdf"):
                    pdf_links.append(full_link)
                elif is_allowed_url(full_link) and full_link not in self.visited_urls:
                    html_links.append(full_link)

            log.info(f"Extracted {len(pdf_links)} PDFs and {len(html_links)} child links from {url}")

            for pdf_url in pdf_links:
                self.download_pdf(pdf_url, category=category)

            for child_url in html_links[:10]:  # Crawl child links
                self.scrape_page(child_url, category=category)

        except Exception as e:
            log.error(f"Failed page scrape {url}: {e}")
            record_warning(url, f"Scrape Error: {e}")


def main():
    scraper = FullSiteScraper()

    # Load seeds from sources.yaml
    seed_urls = []
    if Path("sources.yaml").exists():
        with open("sources.yaml", "r", encoding="utf-8") as f:
            sources = yaml.safe_load(f)
            for item in sources:
                if isinstance(item, dict) and "url" in item:
                    seed_urls.append((item["url"], item.get("category", "webpage"), item.get("type", "html_page")))

    # Base portal seeds
    seed_urls.extend([
        ("https://www.bis.gov.in/?lang=en", "main_portal", "html_page"),
        ("https://www.crsbis.in/BIS/publicdashAction.do", "crs_portal", "html_page"),
        ("https://standardsbis.bsbedge.com/", "standards_portal", "html_page"),
        ("https://standards.bis.gov.in/website/technical-departments/department-list", "technical_depts", "html_page"),
        ("https://irportal.bis.gov.in/home", "ir_portal", "html_page"),
    ])

    for url, category, src_type in seed_urls:
        if url.lower().endswith(".pdf") or src_type == "pdf_listing":
            if url.lower().endswith(".pdf"):
                scraper.download_pdf(url, category=category)
            else:
                scraper.scrape_page(url, category=category)
        else:
            scraper.scrape_page(url, category=category)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
