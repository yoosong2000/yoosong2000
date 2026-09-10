#!/usr/bin/env python3
"""
DEMO VERSION - Academic Journal Reference Updater
Shows workflow with sample data (full version runs locally where network access is available).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path.home() / "yoosong2000" / "journal_data"
BIB_DIR = BASE_DIR / "bibtex"
JSON_DIR = BASE_DIR / "json"

# Sample data from your snapshot page
SAMPLE_ARTICLES = [
    {
        'journal': 'Philosophy of Science',
        'title': 'Understanding Symptoms: Diagnosis, Cure, and Bodily Reintegration',
        'link': 'https://www.cambridge.org/core/journals/philosophy-of-science/article/understanding-symptoms',
        'doi': '10.1017/psa.2026.XX',
        'authors': 'Helene Scott-Fordsmand',
        'published': '2026-07-15',
        'summary': 'A new approach to symptom analysis in medical philosophy.'
    },
    {
        'journal': 'BJPS',
        'title': 'Against Prohibition: When Using Ordinal Scales to Compare Groups Is OK',
        'link': 'https://www.journals.uchicago.edu/doi/full/10.1086/XXXXX',
        'doi': '10.1086/XXXXX',
        'authors': 'Cristian Larroulet Philippi',
        'published': '2026-06-20',
        'summary': 'Reconsidering statistical methodology for ordinal data analysis.'
    },
    {
        'journal': 'Synthese',
        'title': 'On progress in science and metaphysics',
        'link': 'https://link.springer.com/article/10.1007/s11229-026-XXXXX',
        'doi': '10.1007/s11229-026-XXXXX',
        'authors': 'Kian Salimkhani & Matthias Rolffs',
        'published': '2026-08-01',
        'summary': 'A comprehensive analysis of scientific progress in metaphysical contexts.'
    },
    {
        'journal': 'Episteme',
        'title': 'Sexual Epistemic Injustice',
        'link': 'https://www.cambridge.org/core/journals/episteme/article/sexual-epistemic-injustice',
        'doi': '10.1017/epi.2026.XX',
        'authors': 'Ognjen Arandjelović',
        'published': '2026-06-15',
        'summary': 'Examining epistemic injustice in contexts of sexual knowledge.'
    },
    {
        'journal': 'Social Epistemology',
        'title': 'How Has "Opening Up" Science Changed Scientific Practices?',
        'link': 'https://www.tandfonline.com/doi/full/10.1080/02691728.2026.XXXXX',
        'doi': '10.1080/02691728.2026.XXXXX',
        'authors': 'Rachel A. Ankeny',
        'published': '2026-07-10',
        'summary': 'Analysis of open science movements and their impact on epistemic practices.'
    }
]


def generate_bibtex_entry(article: Dict) -> str:
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
    bib += "}\n\n"

    return bib


def main():
    """Generate demo outputs."""
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Journal Feed Processor")
    logger.info(f"{'='*60}\n")

    BIB_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    # Generate BibTeX
    success_bib = ''.join([generate_bibtex_entry(a) for a in SAMPLE_ARTICLES])
    bib_path = BIB_DIR / "articles_latest.bib"

    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(success_bib)

    logger.info(f"✓ Generated BibTeX: {bib_path}")

    # Generate JSON
    json_data = {
        'snapshot_date': datetime.now().isoformat(),
        'total_articles': len(SAMPLE_ARTICLES),
        'articles': SAMPLE_ARTICLES,
        'stats': {
            'total_fetched': len(SAMPLE_ARTICLES),
            'total_pdfs_attempted': 0,
            'pdfs_succeeded': 0,
            'by_journal': {
                'Philosophy of Science': {'fetched': 1, 'pdfs_succeeded': 0},
                'BJPS': {'fetched': 1, 'pdfs_succeeded': 0},
                'Synthese': {'fetched': 1, 'pdfs_succeeded': 0},
                'Episteme': {'fetched': 1, 'pdfs_succeeded': 0},
                'Social Epistemology': {'fetched': 1, 'pdfs_succeeded': 0},
            }
        }
    }

    json_path = JSON_DIR / "journals_latest.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Generated JSON: {json_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total articles: {len(SAMPLE_ARTICLES)}")
    logger.info(f"\nJournals represented:")
    for article in SAMPLE_ARTICLES:
        logger.info(f"  • {article['journal']}")
    logger.info(f"\nOutput files:")
    logger.info(f"  BibTeX: {bib_path}")
    logger.info(f"  JSON:   {json_path}")
    logger.info(f"\nNext step:")
    logger.info(f"  python3 sync_webpage.py")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
