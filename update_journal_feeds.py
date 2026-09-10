#!/usr/bin/env python3
"""
Academic Journal Reference Updater
Fetches latest articles from philosophy journals, downloads PDFs, and generates
BibTeX + JSON for integration with the snapshot webpage.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import urllib.request
import xml.etree.ElementTree as ET
from urllib.error import URLError, HTTPError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory for all output
BASE_DIR = Path.home() / "yoosong2000" / "journal_data"
PDF_DIR = BASE_DIR / "pdfs"
BIB_DIR = BASE_DIR / "bibtex"
JSON_DIR = BASE_DIR / "json"

# Journal RSS feeds - all 13 journals from snapshot
JOURNAL_FEEDS = {
    # Philosophy of Science journals
    'Philosophy of Science': 'https://www.cambridge.org/core/rss/product/id/3FA3E42C808A271752EDD8713E8FC268',
    'BJPS': 'https://www.journals.uchicago.edu/action/showFeed?jc=bjps&type=etoc&feed=rss',
    'Synthese': 'https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=11229&channel-name=Synthese',
    'Erkenntnis': 'https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=10670&channel-name=Erkenntnis',
    'SHPS': 'https://rss.sciencedirect.com/publication/science/00393681',
    'EJPS': 'https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=13194&channel-name=EJPS',
    'Perspectives on Science': 'https://direct.mit.edu/posc/feed/rss',

    # Social Epistemology journals
    'Episteme': 'https://www.cambridge.org/core/rss/product/id/71889765CB94C81977C512106F267CD1',
    'Social Epistemology': 'https://www.tandfonline.com/feed/rss/tsep20',

    # Adjacent General Philosophy journals
    'Journal of Philosophy': 'https://www.pdcnet.org/pdc/bvdb.nsf/getrssxml?openagent&synonym=jphil',
    'Philosophical Studies': 'https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=11098&channel-name=PhilStudies',
    'Philosophy and Phenomenological Research': 'https://onlinelibrary.wiley.com/feed/19331592/most-recent',
    'Metaphilosophy': 'https://onlinelibrary.wiley.com/feed/14679973/most-recent',
}


def initialize_directories():
    """Create necessary directories for output."""
    for directory in [PDF_DIR, BIB_DIR, JSON_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized directories under {BASE_DIR}")


def parse_rss_feed(feed_url: str) -> List[Dict]:
    """Parse RSS/Atom feed using built-in XML parser."""
    try:
        with urllib.request.urlopen(feed_url, timeout=15) as response:
            xml_content = response.read()
            root = ET.fromstring(xml_content)

            # Handle both RSS and Atom namespaces
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            articles = []

            for item in items:
                # RSS or Atom tags
                title_elem = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                link_elem = item.find('link') or item.find('{http://www.w3.org/2005/Atom}link')
                author_elem = item.find('author') or item.find('{http://www.w3.org/2005/Atom}author')
                desc_elem = item.find('description') or item.find('{http://www.w3.org/2005/Atom}summary')
                doi_elem = item.find('{http://purl.org/dc/elements/1.1/}identifier') or item.find('{http://prismstandard.org/namespaces/basic/2.0/}doi')

                article = {
                    'title': (title_elem.text or 'Unknown').replace('{', '').replace('}', '').strip(),
                    'link': link_elem.text or (link_elem.get('href') if link_elem is not None else ''),
                    'doi': (doi_elem.text or '').strip() if doi_elem is not None else '',
                    'authors': (author_elem.text or '').strip() if author_elem is not None else '',
                    'published': datetime.now().isoformat(),
                    'summary': ((desc_elem.text or '')[:200]).strip() if desc_elem is not None else '',
                }
                articles.append(article)

            return articles
    except (URLError, HTTPError, ET.ParseError) as e:
        logger.warning(f"  Feed parse error: {e}")
        return []


def fetch_journal_feed(journal_name: str, feed_url: str, max_entries: int = 10) -> List[Dict]:
    """Fetch and parse a journal RSS feed."""
    try:
        logger.info(f"Fetching {journal_name}...")
        articles = parse_rss_feed(feed_url)

        if not articles:
            logger.warning(f"  No entries found for {journal_name}")
            return []

        # Limit to max_entries
        articles = articles[:max_entries]

        # Add journal name to each article
        for article in articles:
            article['journal'] = journal_name

        logger.info(f"  ✓ Found {len(articles)} articles from {journal_name}")
        return articles

    except Exception as e:
        logger.error(f"  ✗ Error fetching {journal_name}: {e}")
        return []


def download_pdf_with_playwright(doi_url: str, journal_name: str, paper_title: str) -> bool:
    """
    Attempt to download PDF using Playwright.
    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed. Skipping PDF download. Install with: pip install playwright")
        return False

    target_dir = PDF_DIR / journal_name
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_title = "".join([c for c in paper_title if c.isalnum() or c in (' ', '_')]).rstrip()[:150]
    save_path = target_dir / f"{safe_title}.pdf"

    if save_path.exists():
        logger.debug(f"  Already cached: {safe_title}")
        return True

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                page.goto(doi_url, timeout=60000, wait_until="domcontentloaded")

                # Try to find and click PDF link
                pdf_selectors = [
                    "text=View PDF", "text=Download PDF", "text=Save PDF",
                    "a:has-text('PDF')", "button:has-text('PDF')"
                ]

                for selector in pdf_selectors:
                    try:
                        if page.locator(selector).first.is_visible(timeout=2000):
                            page.locator(selector).first.click()
                            break
                    except Exception:
                        continue

                # Attempt download
                try:
                    with page.context.expect_download(timeout=15000) as download_info:
                        page.keyboard.press("Control+s")
                    download = download_info.value
                    download.save_as(str(save_path))
                    logger.debug(f"  Downloaded: {safe_title}")
                    return True
                except Exception:
                    logger.debug(f"  No PDF available for: {safe_title}")
                    return False

            finally:
                page.close()
                browser.close()

    except Exception as e:
        logger.debug(f"  PDF download failed for {safe_title}: {e}")
        return False


def generate_bibtex_entry(article: Dict, success: bool = True) -> str:
    """Generate a BibTeX entry from article metadata."""
    timestamp_key = datetime.now().strftime("%Y%m%d%H%M%S")
    cite_key = f"{article['journal'].replace(' ', '').lower()}_{timestamp_key}"

    bib = f"@article{{{cite_key},\n"
    bib += f"  title = {{{article['title']}}},\n"
    if article.get('authors'):
        bib += f"  author = {{{article['authors']}}},\n"
    bib += f"  journal = {{{article['journal']}}},\n"
    bib += f"  year = {{2026}},\n"
    if article.get('doi'):
        bib += f"  doi = {{{article['doi']}}},\n"
    bib += f"  url = {{{article['link']}}},\n"
    bib += f"  urldate = {{{datetime.now().strftime('%Y-%m-%d')}}},\n"
    if not success:
        bib += f"  note = {{PDF download failed}},\n"
    bib += "}\n\n"

    return bib


def process_all_journals(download_pdfs: bool = False) -> Tuple[List[Dict], Dict]:
    """Process all journal feeds and optionally download PDFs."""
    all_articles = []
    stats = {
        'total_fetched': 0,
        'total_pdfs_attempted': 0,
        'pdfs_succeeded': 0,
        'by_journal': {}
    }

    logger.info(f"\n{'='*60}")
    logger.info("Starting journal feed processing...")
    logger.info(f"{'='*60}\n")

    for journal_name, feed_url in JOURNAL_FEEDS.items():
        articles = fetch_journal_feed(journal_name, feed_url)
        stats['total_fetched'] += len(articles)
        stats['by_journal'][journal_name] = {
            'fetched': len(articles),
            'pdfs_succeeded': 0
        }

        if download_pdfs:
            for article in articles:
                stats['total_pdfs_attempted'] += 1
                if download_pdf_with_playwright(article['link'], journal_name, article['title']):
                    stats['pdfs_succeeded'] += 1
                    stats['by_journal'][journal_name]['pdfs_succeeded'] += 1

        all_articles.extend(articles)

    return all_articles, stats


def save_outputs(articles: List[Dict], stats: Dict):
    """Save articles as BibTeX and JSON."""
    timestamp = datetime.now().strftime("%Y%m%d")

    # Save BibTeX
    success_bib = ""
    failed_bib = ""

    for article in articles:
        bib_entry = generate_bibtex_entry(article, success=True)
        success_bib += bib_entry

    bib_path_success = BIB_DIR / f"articles_latest.bib"
    bib_path_archive = BIB_DIR / f"articles_{timestamp}.bib"

    with open(bib_path_success, 'w', encoding='utf-8') as f:
        f.write(success_bib)
    with open(bib_path_archive, 'w', encoding='utf-8') as f:
        f.write(success_bib)

    logger.info(f"✓ Saved BibTeX: {bib_path_success}")

    # Save JSON for webpage integration
    json_data = {
        'snapshot_date': datetime.now().isoformat(),
        'total_articles': len(articles),
        'articles': articles,
        'stats': stats
    }

    json_path = JSON_DIR / f"journals_latest.json"
    json_archive = JSON_DIR / f"journals_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    with open(json_archive, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved JSON: {json_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles fetched: {stats['total_fetched']}")
    logger.info(f"Total PDFs attempted: {stats['total_pdfs_attempted']}")
    logger.info(f"PDFs succeeded: {stats['pdfs_succeeded']}")
    logger.info(f"\nOutput files:")
    logger.info(f"  Bibtex: {bib_path_success}")
    logger.info(f"  JSON:   {json_path}")
    logger.info(f"  PDFs:   {PDF_DIR}")
    logger.info(f"{'='*60}\n")


def main():
    """Main entry point."""
    import sys

    download_pdfs = '--download-pdfs' in sys.argv or '-d' in sys.argv

    initialize_directories()
    articles, stats = process_all_journals(download_pdfs=download_pdfs)
    save_outputs(articles, stats)

    if not download_pdfs:
        logger.info("Tip: Run with --download-pdfs or -d to attempt PDF downloads")
        logger.info("(Requires Playwright: pip install playwright && playwright install)")


if __name__ == "__main__":
    main()
