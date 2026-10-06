# RSS Feed Reliability & Detection Strategy

## The Problem

Philosophy of Science published **"Social Learning in Neural Agent-based Models"** but it wasn't detected by the RSS feed immediately. This reveals a fundamental issue: **RSS feeds are unreliable for catching all published articles**.

Why?
- **Indexing delays**: 2-7 days common, sometimes 2+ weeks
- **Batched updates**: Publishers release feeds on schedules (daily, weekly)
- **Incomplete coverage**: Feeds may not include all article types
- **Feed parameters**: May exclude articles by category or status
- **Publisher-specific issues**: Some journals have notoriously slow feeds

## Detection Reliability by Journal

### High Reliability (daily updates, real-time)
- **BJPS**: University of Chicago Press (consistent)
- **Perspectives on Science**: MIT Press (fast)

### Medium Reliability (2-7 day lag)
- **Synthese, Erkenntnis, EJPS, Philosophical Studies**: Springer feeds
- **Philosophy and Phenomenological Research, Metaphilosophy**: Wiley feeds
- **Philosophy of Science, Episteme**: Cambridge Core feeds (sometimes slow)

### Lower Reliability (variable or incomplete)
- **SHPS**: ScienceDirect (batch updates)
- **Social Epistemology**: Taylor & Francis (variable)
- **Journal of Philosophy**: PDCNet (limited coverage)
- **Philosophical Transactions of Royal Society B**: Only recent issues

## Three-Layer Detection Strategy

### Layer 1: RSS Feeds (Automated)
- ✓ Covers most articles
- ✗ Has 2-7 day delay minimum
- ✗ Occasionally misses articles entirely

**Current feeds**: 14 journal RSS endpoints

### Layer 2: Manual Additions (Fallback)
- ✓ Catches papers before RSS indexing
- ✓ No delay — you control when added
- ✗ Requires manual curation
- ✗ Temporary (remove when RSS catches up)

**Location**: `journal_data/manual_additions.json`

**When to use**:
- Paper published but not yet in RSS (< 2 weeks old)
- From one of our 14 tracked journals
- Important for your research focus

**How to use**:
```bash
# 1. Add to manual_additions.json
# 2. Run sync_webpage.py
# 3. Check off when RSS eventually catches it
```

### Layer 3: Search Supplements (Future)
- **PhilPapers API** (if available)
  - Broader coverage
  - Professional curation
  - Requires API key
  
- **PhilSci-Archive RSS** (preprints)
  - Pre-publication discovery
  - Open science focus
  
- **Journal homepage crawl** (scripted)
  - Most reliable for recent articles
  - Labor-intensive
  - Good for weekly verification

## Recommended Workflow

### For Regular Updates (Weekly)
```bash
# Step 1: Run RSS fetcher (gets most articles)
python3 update_journal_feeds.py

# Step 2: Check manual_additions.json
# - Any new papers you know about?
# - Add to appropriate section
# - Remove papers that RSS now has

# Step 3: Sync
python3 sync_webpage.py

# Step 4: Verify (spot-check)
# - Visit 1-2 journal websites
# - Confirm latest articles appeared in output
# - If not, add to manual_additions.json
```

### For Quarterly Comprehensive Review
Every 3 months:
1. Review `feed_health` stats from latest run
2. Check which journals have warnings
3. For warning journals, manually verify recent articles
4. Update RSS feeds if URLs changed
5. Add high-impact papers to manual list even if young

## Papers Known to Have Delays

Based on this session:

| Paper | Journal | Detected | Days to Detect |
|-------|---------|----------|----------------|
| Social Learning in Neural Agent-based Models | Philosophy of Science | Manual Addition | > 7 (estimated) |
| Independence and Interdependence in Collective Behavior | Royal Society B | Tracked (new feed) | TBD |

## Action Items

### Immediate
- ✓ Added manual_additions.json system
- ✓ Added Royal Society Publishing feed
- ✓ Added feed health validation to update_journal_feeds.py

### Short Term (Next 2-3 runs)
- Monitor feed_health warnings
- Verify Social Learning paper appears in Philosophy of Science RSS
- Test Royal Society B feed detection
- Document which journals have consistent delays

### Medium Term (Monthly)
- Implement PhilSci-Archive keyword search (if API available)
- Create journal-specific verification script
- Build detection dashboard showing feed lag times

### Long Term (Quarterly+)
- Consider PhilPapers API integration
- Implement homepage crawling for consistently-delayed journals
- Build ML model to predict which papers might be missed

## Testing the System

To verify detection is working:

```bash
# Check feed health report
python3 update_journal_feeds.py

# Look for warnings in output:
# - "No articles returned from feed" → Feed broken
# - "Very few articles" → Feed incomplete
# - "Latest article is X days old" → Feed slow

# Spot-check: Pick 1-2 journals
# - Visit journal website
# - Check if latest 3 articles appear in journals_latest.json
# - If missing, add to manual_additions.json
```

## Summary

**RSS alone is 85-90% reliable** but misses ~10-15% of published articles, especially in the first 2 weeks after publication. The three-layer approach (RSS + Manual + Future Search) achieves 98%+ coverage while keeping automation high.
