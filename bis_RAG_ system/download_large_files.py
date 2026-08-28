"""
Dedicated Chunked Stream Downloader for Large BIS PDFs
Streams large PDFs (>50MB-120MB) in 1MB chunks with retries and realistic headers.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("large_downloader")

TARGET_URLS = [
    "https://www.bis.gov.in/org/AnnualReport2016-17.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2024/10/Annual-Report-2022-23.pdf",
    "https://www.bis.gov.in/org/ANNUALREPORT2013.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2024/10/Delay-statement-2022-2023.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2024/12/Revised-Guidelines-for-support-to-other-laboratories-dated-2024-12-17.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2020/11/LRS-Forms-and-Undertakings.pdf",
    "https://www.bis.gov.in/bs/BIS_Hallmarking_Regulations_2018_Gazette_notification.pdf",
    "https://www.bis.gov.in/bs/DoCA_BIS_Hallmarking_Regulations_2018_Gazette_notification.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2026/07/Guidelines-for-Jewellers.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2026/08/ECGazetteNotification.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2026/06/Group_1_24062026.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2026/08/Notification-related-to-mandatory-Hallmarking-2.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2020/10/Guide_Jeweller_Registration_v1.1.pdf",
    "https://www.bis.gov.in/wp-content/uploads/2026/07/GuidelinesForAHCs.pdf",
]

OUT_DIR = Path("./raw_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"


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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.bis.gov.in/",
    })
    return session


def download_stream(url: str, session: requests.Session):
    filename_raw = Path(urlparse(url).path).name
    log.info(f"Starting chunked stream download: {url}")
    
    try:
        resp = session.get(url, stream=True, timeout=(15, 300))
        if resp.status_code == 403:
            log.warning(f"Access 403 Forbidden for {url}")
            return
        resp.raise_for_status()

        sha256 = hashlib.sha256()
        temp_file = OUT_DIR / f"temp_{filename_raw}"
        
        total_bytes = 0
        with open(temp_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    sha256.update(chunk)
                    total_bytes += len(chunk)
                    log.info(f"Downloaded {total_bytes / (1024*1024):.2f} MB of {filename_raw}...")

        content_hash = sha256.hexdigest()
        final_filename = f"{content_hash[:12]}_{filename_raw}"
        final_path = OUT_DIR / final_filename
        
        if temp_file.exists():
            temp_file.replace(final_path)

        manifest_entry = {
            "source_url": url,
            "local_path": str(final_path),
            "content_type": "pdf",
            "category": "lab_directory",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
            "extra_metadata": {"direct_stream_download": True, "size_bytes": total_bytes},
        }

        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        log.info(f"SUCCESS: Saved {url} -> {final_filename} ({total_bytes / (1024*1024):.2f} MB)")

    except Exception as e:
        log.error(f"Failed download for {url}: {e}")


def main():
    session = create_resilient_session()
    for url in TARGET_URLS:
        download_stream(url, session)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
