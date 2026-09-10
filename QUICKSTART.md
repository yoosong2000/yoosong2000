# Journal Feed Workflow — Quick Start

Your Python reference updater has been adapted to work with all 13 journals in your snapshot and integrated with the webpage workflow.

## 📋 What You Have

```
~/yoosong2000/
├── update_journal_feeds.py       # Main fetcher (run locally)
├── update_journal_feeds_DEMO.py   # Demo with sample data
├── sync_webpage.py                # Sync fetched data to HTML
├── JOURNAL_WORKFLOW.md            # Full documentation
├── current-issue.html             # Your snapshot webpage
├── the-current-issue-summary.md   # Markdown reference
└── journal_data/
    ├── json/journals_latest.json  # Article metadata
    ├── bibtex/articles_latest.bib # BibTeX citations
    └── pdfs/                       # Downloaded PDFs (optional)
```

## 🚀 Usage

### Local Machine (where you have network access):

```bash
# 1. Quick fetch (no PDFs)
python3 update_journal_feeds.py

# 2. Or, with PDF downloads
python3 update_journal_feeds.py --download-pdfs

# 3. Outputs:
#    - journal_data/json/journals_latest.json
#    - journal_data/bibtex/articles_latest.bib
#    - journal_data/pdfs/[Journal]/ (if --download-pdfs)
```

### Then Sync to Webpage:

```bash
# 4. Sync with webpage
python3 sync_webpage.py

# 5. Commit changes
git add current-issue.html
git commit -m "Update journal feeds — $(date +%Y-%m-%d)"
git push origin claude/create-webpage-artifact-q62gpf

# 6. Republish artifact via Claude
#    → Use Claude Artifact tool to update the published page
```

## 🔬 Try the Demo First

Test the full workflow locally with sample data:

```bash
python3 update_journal_feeds_DEMO.py
python3 sync_webpage.py
```

This generates example output without needing network access.

## 📚 Journals Covered (13 total)

**Philosophy of Science (7):**
- Philosophy of Science, BJPS, Synthese, Erkenntnis, SHPS, EJPS, Perspectives on Science

**Social Epistemology (2):**
- Episteme, Social Epistemology

**General Philosophy (4):**
- Journal of Philosophy, Philosophical Studies, Philosophy and Phenomenological Research, Metaphilosophy

## 📊 Output Formats

### JSON (for webpage integration)
```json
{
  "snapshot_date": "2026-09-10T...",
  "articles": [
    {
      "journal": "Philosophy of Science",
      "title": "...",
      "link": "https://...",
      "doi": "10.1234/...",
      "authors": "...",
      "summary": "..."
    }
  ]
}
```

### BibTeX (for reference manager)
```bibtex
@article{philosophyofscience_...,
  title = {...},
  author = {...},
  journal = {...},
  doi = {...},
  url = {...}
}
```

## 🔗 Full Guide

See `JOURNAL_WORKFLOW.md` for:
- Detailed setup instructions
- Network/authentication troubleshooting
- Scheduling automated updates
- PDF download configuration
- Custom RSS feed URLs

## ⚙️ Customization

Edit `JOURNAL_FEEDS` dict in `update_journal_feeds.py` to:
- Add/remove journals
- Update RSS feed URLs
- Change max articles per fetch
- Adjust publication limits

---

**Status:** Ready to use on your local machine. Run demo first to verify workflow.
