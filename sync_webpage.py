#!/usr/bin/env python3
"""
Synchronize journal feed data with the snapshot webpage.
Reads JSON from update_journal_feeds.py and updates the HTML artifact.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path.home() / "yoosong2000"
JSON_FILE = BASE_DIR / "journal_data" / "journals_latest.json"
HTML_FILE = BASE_DIR / "current-issue.html"


def load_journal_data() -> Dict:
    """Load the latest journal data from JSON."""
    if not JSON_FILE.exists():
        logger.error(f"Journal data not found: {JSON_FILE}")
        logger.info("Run update_journal_feeds.py first: python3 update_journal_feeds.py")
        return None

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_articles_by_journal(articles: List[Dict]) -> Dict[str, List[Dict]]:
    """Group articles by journal name."""
    grouped = {}
    for article in articles:
        journal = article['journal']
        if journal not in grouped:
            grouped[journal] = []
        grouped[journal].append(article)
    return grouped


def generate_article_html(article: Dict, journal_name: str) -> str:
    """Generate HTML for a single article."""
    title = article['title']
    link = article['link']
    doi = article.get('doi', '')
    authors = article.get('authors', '')

    # Format DOI as link if available
    doi_link = f'<span class="a-doi">https://doi.org/{doi}</span>' if doi else ''

    html = f'          <li><a class="a-title" href="{link}" target="_blank" rel="noopener">{title}'
    if doi:
        html += f'<span class="a-vol">{doi}</span>'
    html += '</a>'
    if authors:
        html += f'<span class="a-author">{authors}</span>'
    html += '</li>\n'

    return html


def generate_journal_card_html(journal_name: str, articles: List[Dict]) -> str:
    """Generate HTML card for a journal with its articles."""
    if not articles:
        return ""

    article_list = ''.join([generate_article_html(a, journal_name) for a in articles[:10]])  # Limit to 10

    card = f"""      <article class="card" data-search="{journal_name.lower()}">
        <div class="card-top">
          <h3 class="card-title">{journal_name}</h3>
          <a class="card-link" href="https://philpapers.org/s/{journal_name.replace(' ', '%20')}" target="_blank" rel="noopener">Search PhilPapers ↗</a>
        </div>
        <div class="meta-row"><span class="publisher">Latest from RSS feed</span></div>
        <ul class="articles">
{article_list}        </ul>
      </article>

"""
    return card


def update_snapshot_date(html: str, new_date: str) -> str:
    """Update the snapshot date in the HTML."""
    # Find and update the date range in the subtitle
    pattern = r'(first compiled \d{1,2} \w+ \d{4}, rechecked).*?(\d{4}\.)'
    replacement = rf'\1 the week of {new_date}. \2'
    return re.sub(pattern, replacement, html)


def sync_webpage(data: Dict):
    """Sync journal data to the HTML artifact."""
    if not HTML_FILE.exists():
        logger.error(f"HTML file not found: {HTML_FILE}")
        return

    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        html = f.read()

    # Update snapshot date
    today = datetime.now()
    date_str = today.strftime("%d %B %Y")
    html = update_snapshot_date(html, date_str)

    # Group articles by journal
    articles_by_journal = group_articles_by_journal(data['articles'])

    logger.info(f"\nSync Report:")
    logger.info(f"{'='*60}")
    for journal_name, articles in sorted(articles_by_journal.items()):
        logger.info(f"{journal_name}: {len(articles)} articles")

    # Save updated HTML
    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"{'='*60}")
    logger.info(f"✓ Updated: {HTML_FILE}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review changes: git diff {HTML_FILE.name}")
    logger.info(f"  2. Commit: git add {HTML_FILE.name} && git commit -m 'Update journal feeds'")
    logger.info(f"  3. Push: git push -u origin claude/create-webpage-artifact-q62gpf")
    logger.info(f"  4. Republish artifact via Claude Artifact tool")


def main():
    """Main entry point."""
    logger.info("Loading journal data...")
    data = load_journal_data()

    if not data:
        return

    logger.info(f"Loaded {data['total_articles']} articles")
    sync_webpage(data)


if __name__ == "__main__":
    main()
