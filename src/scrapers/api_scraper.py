"""
JS-Rendered API Scraper for BIS Systems
Fetches structured JSON datasets directly from LIMS (lims.bis.gov.in),
Know Your Standards (standards.bis.gov.in), and ManakOnline portals.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api_scraper")

OUT_DIR = Path("./raw_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"


def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://lims.bis.gov.in/",
        "Content-Type": "application/json",
    }


def record_manifest(item: dict):
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def fetch_lims_labs():
    log.info("Fetching LIMS Recognized & Empanelled Labs API...")
    urls = [
        ("https://lims.bis.gov.in/home/labs/", "lims_recognized_labs"),
        ("https://lims.bis.gov.in/home/empaneled_labs/", "lims_empaneled_labs"),
        ("https://standards.bis.gov.in/website/know-your-standards", "know_your_standards"),
    ]
    
    session = requests.Session()
    session.headers.update(get_headers())

    for url, category in urls:
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                content = resp.content
                content_hash = hashlib.sha256(content).hexdigest()
                out_path = OUT_DIR / f"{content_hash[:12]}_{category}.json"
                
                # Save response JSON/HTML
                out_path.write_bytes(content)
                record_manifest({
                    "source_url": url,
                    "local_path": str(out_path),
                    "content_type": "json_api",
                    "category": category,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "content_hash": content_hash,
                    "extra_metadata": {"api_fetch": True},
                })
                log.info(f"SUCCESS: Saved {category} -> {out_path.name}")
            else:
                log.warning(f"LIMS HTTP {resp.status_code} for {url}")
        except Exception as e:
            log.error(f"Error fetching LIMS endpoint {url}: {e}")


if __name__ == "__main__":
    fetch_lims_labs()
