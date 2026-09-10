# Academic Journal Snapshot Workflow

This workflow automates fetching the latest articles from your 13 core philosophy journals, downloading PDFs, and syncing the data with your snapshot webpage.

## Setup

### 1. Install Dependencies

```bash
pip install feedparser playwright
playwright install chromium
```

### 2. Configure Your Environment

All output goes to `~/yoosong2000/journal_data/`:
- `pdfs/` — Downloaded PDF files, organized by journal
- `bibtex/` — BibTeX citation files (latest + date-stamped archives)
- `json/` — JSON data for webpage integration (latest + archives)

## Workflow

### Quick Fetch (no PDF downloads)
```bash
python3 update_journal_feeds.py
```

Outputs:
- `journal_data/json/journals_latest.json` — Article metadata
- `journal_data/bibtex/articles_latest.bib` — BibTeX citations

**Time:** ~30 seconds

### Full Fetch (with PDF downloads)
```bash
python3 update_journal_feeds.py --download-pdfs
```

Same outputs + attempts to download PDFs to `journal_data/pdfs/[Journal Name]/`

**Time:** 5-15 minutes (depending on PDF availability)

### Sync to Webpage
```bash
python3 sync_webpage.py
```

Updates `current-issue.html` with:
- Latest article metadata from the JSON feed
- Updated snapshot date
- Search-indexable content

Then commit and push:
```bash
git add current-issue.html the-current-issue-summary.md
git commit -m "Update journal feeds — $(date +%Y-%m-%d)"
git push -u origin claude/create-webpage-artifact-q62gpf
```

Finally, republish the artifact via Claude Artifact tool.

---

## Journals Tracked

### Philosophy of Science (7 journals)
1. **Philosophy of Science** (Cambridge Core)
2. **BJPS** (University of Chicago Press)
3. **Synthese** (Springer)
4. **Erkenntnis** (Springer)
5. **SHPS** (Elsevier)
6. **EJPS** (Springer)
7. **Perspectives on Science** (MIT Press)

### Social Epistemology (2 journals)
8. **Episteme** (Cambridge Core)
9. **Social Epistemology** (Taylor & Francis)

### Adjacent General Philosophy (4 journals)
10. **Journal of Philosophy** (Philosophy Documentation Center)
11. **Philosophical Studies** (Springer)
12. **Philosophy and Phenomenological Research** (Wiley)
13. **Metaphilosophy** (Wiley)

---

## Output Formats

### JSON Structure
```json
{
  "snapshot_date": "2026-09-10T12:34:56...",
  "total_articles": 127,
  "articles": [
    {
      "journal": "Philosophy of Science",
      "title": "Article Title",
      "link": "https://...",
      "doi": "10.1234/...",
      "authors": "Author Name",
      "published": "2026-09-01...",
      "summary": "Brief abstract..."
    }
  ],
  "stats": {
    "total_fetched": 127,
    "total_pdfs_attempted": 127,
    "pdfs_succeeded": 45,
    "by_journal": { ... }
  }
}
```

### BibTeX Structure
```bibtex
@article{philosophyofscience_20260910123456,
  title = {Article Title},
  author = {Author Name},
  journal = {Philosophy of Science},
  year = {2026},
  doi = {10.1234/...},
  url = {https://...},
  urldate = {2026-09-10},
}
```

---

## Scheduling (Optional)

To run weekly updates automatically via cron:

```bash
# Edit crontab
crontab -e

# Add line for weekly update (Sunday 10am)
0 10 * * 0 cd ~/yoosong2000 && python3 update_journal_feeds.py >> journal_data/logs/$(date +\%Y\%m\%d).log 2>&1
```

Or use the Claude Code routine system:
```bash
/create-trigger "Update journal feeds" "0 10 * * 0" "python3 ~/yoosong2000/update_journal_feeds.py"
```

---

## Troubleshooting

### "No entries found for [Journal]"
- RSS feed URL may have changed
- Journal may not provide RSS
- Network access may be blocked

**Fix:** Check feed URL in `JOURNAL_FEEDS` dict, update if needed

### PDF downloads failing
- Requires JavaScript rendering (Playwright with headless=False for interactive)
- Some publishers require authentication
- Some block automated access

**Fix:** Modify `download_pdf_with_playwright()` or manually review `articles_latest.bib` for failed entries

### Encoding errors
- Ensure UTF-8 encoding for all output files
- Already handled in scripts, but check if custom modifications needed

---

## Integration with Artifact

The `sync_webpage.py` script prepares data, but the actual artifact republishing happens through the Claude Artifact tool:

1. Run `update_journal_feeds.py` to fetch latest data
2. Run `sync_webpage.py` to merge with HTML
3. Use Claude `Artifact` tool to republish the updated HTML
4. Git commit & push for version control

This keeps your snapshot current while maintaining editorial control over formatting and presentation.
