#!/usr/bin/env python3
"""
Download newly detected papers and save to Obsidian/Mendeley folders.
Run this on your Windows machine where you have network access and RUB credentials.

This script:
1. Reads journals_latest.json from the update workflow
2. Attempts to download PDFs from publisher links
3. Saves to both Obsidian raw materials and Mendeley reference folders
4. Tracks download success/failure for next sync
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import urllib.request
from urllib.error import URLError, HTTPError
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Windows paths
OBSIDIAN_RAW = Path(r"C:\Users\yooso\iCloudDrive\iCloud~md~obsidian\johny\raw")
MENDELEY_FOLDER = Path(r"C:\Users\yooso\Desktop\reference papers")

# Local paths (adjust based on where you clone the repo)
LOCAL_REPO = Path.home() / "yoosong2000"
JSON_FILE = LOCAL_REPO / "journal_data" / "journals_latest.json"
DOWNLOAD_LOG = LOCAL_REPO / "journal_data" / "downloads.json"


def validate_folders() -> bool:
    """Verify download folders exist and are accessible."""
    logger.info("Validating download folders...")

    folders = [
        (OBSIDIAN_RAW, "Obsidian raw materials"),
        (MENDELEY_FOLDER, "Mendeley reference papers")
    ]

    all_valid = True
    for folder, name in folders:
        if folder.exists():
            logger.info(f"  ✓ {name}: {folder}")
        else:
            logger.error(f"  ✗ {name} not found: {folder}")
            logger.info(f"    Create this folder first, then run again")
            all_valid = False

    return all_valid


def load_journal_data() -> Dict:
    """Load the latest journal data from JSON."""
    if not JSON_FILE.exists():
        logger.error(f"Journal data not found: {JSON_FILE}")
        logger.info("Run update_journal_feeds.py first on your local machine or cloud session")
        return None

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_download_log() -> Dict:
    """Load previous download attempts to avoid re-downloading."""
    if not DOWNLOAD_LOG.exists():
        return {"downloaded": {}, "failed": {}, "last_updated": None}

    with open(DOWNLOAD_LOG, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_filename(title: str, max_length: int = 200) -> str:
    """Create safe filename from paper title."""
    # Remove invalid characters
    invalid_chars = r'<>:"/\|?*'
    safe_title = "".join(c if c not in invalid_chars else "_" for c in title)
    # Remove trailing spaces and periods
    safe_title = safe_title.rstrip('. ')
    # Limit length
    if len(safe_title) > max_length:
        safe_title = safe_title[:max_length].rsplit(' ', 1)[0]
    return safe_title


def attempt_pdf_download(url: str, title: str) -> Tuple[bool, str]:
    """
    Attempt to download PDF from URL.

    Returns: (success, local_path)
    """
    filename = f"{sanitize_filename(title)}.pdf"

    try:
        logger.info(f"  Downloading: {title}")

        # Attempt download to temporary location
        temp_path = OBSIDIAN_RAW / filename

        # Add timeout and user-agent
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            if response.headers.get('content-type', '').startswith('application/pdf'):
                with open(temp_path, 'wb') as f:
                    f.write(response.read())

                # Verify file was created
                if temp_path.exists() and temp_path.stat().st_size > 1000:
                    logger.info(f"    ✓ Downloaded ({temp_path.stat().st_size // 1024} KB)")

                    # Copy to Mendeley folder
                    mendeley_path = MENDELEY_FOLDER / filename
                    shutil.copy2(temp_path, mendeley_path)
                    logger.info(f"    ✓ Copied to Mendeley folder")

                    return True, str(temp_path)
                else:
                    temp_path.unlink(missing_ok=True)
                    logger.warning(f"    ✗ Downloaded file too small (possibly error page)")
                    return False, ""
            else:
                logger.warning(f"    ✗ URL returned non-PDF content")
                return False, ""

    except HTTPError as e:
        if e.code == 403:
            logger.warning(f"    ✗ Access denied (403) — may need RUB login")
        elif e.code == 404:
            logger.warning(f"    ✗ PDF not found (404)")
        else:
            logger.warning(f"    ✗ HTTP Error {e.code}")
        return False, ""

    except URLError as e:
        logger.warning(f"    ✗ Network error: {e.reason}")
        return False, ""

    except Exception as e:
        logger.warning(f"    ✗ Download failed: {e}")
        return False, ""


def process_papers(articles: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """
    Attempt to download papers from article list.

    Returns: (successful_downloads, failed_attempts, failed_papers)
    """
    log = load_download_log()
    downloaded = set(log.get("downloaded", {}).keys())
    failed = set(log.get("failed", {}).keys())

    success_count = 0
    fail_count = 0
    failed_papers = []

    logger.info(f"\nProcessing {len(articles)} articles...")
    logger.info("="*60)

    for article in articles:
        title = article.get('title', 'Unknown')
        link = article.get('link', '')
        doi = article.get('doi', '')
        journal = article.get('journal', '')

        # Skip if already processed
        if title in downloaded:
            logger.debug(f"Skipping (already downloaded): {title}")
            continue

        if title in failed:
            logger.debug(f"Skipping (previous failure): {title}")
            continue

        # Try to download
        if not link:
            logger.warning(f"Skipping (no URL): {title}")
            fail_count += 1
            failed_papers.append({
                'title': title,
                'journal': journal,
                'reason': 'No URL provided'
            })
            continue

        logger.info(f"\n[{journal}] {title}")
        success, filepath = attempt_pdf_download(link, title)

        if success:
            log["downloaded"][title] = {
                'timestamp': datetime.now().isoformat(),
                'journal': journal,
                'link': link,
                'filepath': filepath
            }
            success_count += 1
        else:
            log["failed"][title] = {
                'timestamp': datetime.now().isoformat(),
                'journal': journal,
                'link': link,
                'reason': 'Download failed or access denied'
            }
            fail_count += 1
            failed_papers.append({
                'title': title,
                'journal': journal,
                'link': link,
                'reason': 'Automated download failed'
            })

    # Save download log
    log["last_updated"] = datetime.now().isoformat()
    with open(DOWNLOAD_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return success_count, fail_count, failed_papers


def generate_manual_download_guide(failed_papers: List[Dict]):
    """Generate guide for manually downloading failed papers."""
    if not failed_papers:
        return

    guide_file = LOCAL_REPO / "journal_data" / "MANUAL_DOWNLOADS.md"

    guide = f"""# Manual Download Guide — {datetime.now().strftime('%Y-%m-%d')}

{len(failed_papers)} papers failed automated download and need manual intervention.

## Papers to Download Manually

"""

    for i, paper in enumerate(failed_papers, 1):
        guide += f"""### {i}. {paper['title']}
- **Journal**: {paper['journal']}
- **Link**: {paper['link']}
- **Reason**: {paper['reason']}

**Steps to download**:
1. Open the link above in your browser
2. Search for PDF download button (may require RUB login)
3. Save to: `C:\\Users\\yooso\\Desktop\\reference papers\\`
4. File will auto-sync to Obsidian: `C:\\Users\\yooso\\iCloudDrive\\iCloud~md~obsidian\\johny\\raw\\`

---

"""

    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)

    logger.info(f"\n✓ Manual download guide created: {guide_file}")


def print_summary(success: int, failed: int, total: int):
    """Print download summary."""
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles processed: {total}")
    logger.info(f"Successfully downloaded: {success}")
    logger.info(f"Failed/needs manual: {failed}")
    logger.info(f"Success rate: {success/total*100:.1f}%" if total > 0 else "N/A")
    logger.info(f"\n📁 Obsidian raw materials: {OBSIDIAN_RAW}")
    logger.info(f"📁 Mendeley folder: {MENDELEY_FOLDER}")
    logger.info(f"\nDownload log: {DOWNLOAD_LOG}")
    logger.info(f"{'='*60}\n")


def main():
    """Main entry point."""
    logger.info("Paper Downloader — Obsidian + Mendeley Sync")
    logger.info("="*60)

    # Validate folders
    if not validate_folders():
        logger.error("\n❌ Please create the missing folders and try again")
        return 1

    # Load data
    logger.info("\nLoading journal data...")
    data = load_journal_data()
    if not data:
        return 1

    articles = data.get('articles', [])
    if not articles:
        logger.error("No articles found in journal data")
        return 1

    logger.info(f"Loaded {len(articles)} articles from journals")

    # Download papers
    success, failed, failed_papers = process_papers(articles)

    # Generate manual download guide for failed papers
    if failed_papers:
        generate_manual_download_guide(failed_papers)

    # Print summary
    print_summary(success, failed, len(articles))

    return 0


if __name__ == "__main__":
    sys.exit(main())
