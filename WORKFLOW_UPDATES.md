# Workflow Updates — October 6, 2026

## Changes Made

### 1. Expanded Journal Coverage
Added **Philosophical Transactions of the Royal Society B** to the RSS feeds to capture network science and evolutionary biology papers relevant to agent-based models and epistemology.

### 2. Manual Paper Additions System
Created `journal_data/manual_additions.json` to handle papers that:
- Are published but not yet indexed by RSS feeds
- Come from interdisciplinary sources outside the core 13 journals
- Need to be manually curated for the thematic sections

### 3. Updated Sync Workflow
`sync_webpage.py` now:
- Loads manual additions from `manual_additions.json`
- Merges them with RSS feed data automatically
- Reports total articles including manual additions
- Deduplicates based on title matching

## How to Add New Papers

### Quick Add (Manual)
1. Open `journal_data/manual_additions.json`
2. Add paper to the appropriate thematic section:
   - `agent-based-models`
   - `philosophy-of-data`
   - `philosophy-of-models`

Example:
```json
{
  "title": "Social Learning in Neural Agent-based Models",
  "authors": "Author Name",
  "journal": "Philosophy of Science",
  "year": 2026,
  "doi": "10.xxxx/xxxx",
  "link": "https://...",
  "published": "2026-10-06",
  "summary": "Brief description",
  "status": "awaiting_rss_indexing"
}
```

3. Run sync:
```bash
python3 sync_webpage.py
```

4. Republish artifact:
```bash
# The sync_webpage.py output will guide you
```

## Papers Added This Update

### Agent-Based Models
- **Social Learning in Neural Agent-based Models** (Philosophy of Science, 2026)
  - Status: Awaiting RSS indexing
  - Reason: Philosophy of Science feed may have indexing lag
  - URL: https://www.cambridge.org/core/journals/philosophy-of-science/article/social-learning-in-neural-agentbased-models/88677994DD97E8824060FEF77943B641

## Detection Strategy

Papers are now detected through multiple channels:

1. **RSS Feeds (13 core journals)** — Automated, ~1-7 day lag
2. **Additional RSS (Philosophical Transactions RSS)** — Automated, captures network/evolution papers
3. **Manual Additions** — For edge cases and pre-indexing capture
4. **PhilPapers/PhilSci-Archive** — Suggested for future API integration

## Next Steps

### For Future Syncs
1. Check `manual_additions.json` for papers awaiting RSS indexing
2. Once a paper appears in `journals_latest.json`, remove from manual_additions.json
3. Keep this system lightweight — only use for recent papers (last 2-3 months)

### For Full Automation
To eliminate manual additions, consider:
- PhilPapers API integration (if available)
- PhilSci-Archive keyword search automation
- Scholar.google.com RSS alerts for specific authors/topics

## Technical Notes

- Deduplication happens on title + journal matching
- Manual additions take precedence in sorting (appear first in data)
- Status field tracks whether paper is still awaiting RSS indexing
- All additions are timestamped for audit trail
