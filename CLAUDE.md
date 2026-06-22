# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Resellify

Market resell opportunity finder — watches OLX Romania, Facebook Marketplace, and eBay for flip deals, prices them against eMAG and eBay sold data, and scores them for arbitrage potential.

## Stack

- Python 3.11+, FastAPI, SQLite (aiosqlite + SQLAlchemy async), Playwright, Anthropic SDK, spaCy
- Node.js (for olx-mcp and facebook-marketplace-mcp MCP servers)

## Setup

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
pip install -r requirements.txt
playwright install chromium
python main.py              # dashboard at http://localhost:8000
```

## Running

```bash
python main.py              # dashboard only (manual scan via UI)
python main.py --agent      # dashboard + auto-scan background agent
python main.py --scan       # single scan run, then exit
```

## NER Pipeline (offline, run separately)

```bash
# 1. Label titles from DB with Claude Haiku
python -m src.title_engine.pipeline --from-db --limit 3000 --output-dir data/title_engine

# 2. Train spaCy NER model on labeled data
python -m src.title_engine.trainer --data data/title_engine/train_v0.jsonl --output models/title_ner

# Model loads automatically at runtime if models/title_ner/model-best exists
```

## OLX MCP setup

```bash
npm install -g olx-mcp     # or npx auto-downloads on first use
```

## Facebook MCP setup (macOS + Chrome)

```bash
git clone https://github.com/jdcodes1/facebook-marketplace-mcp
cd facebook-marketplace-mcp && npm install && npm run build
# Set mcp_server_path in config/settings.yaml
```

## Architecture

### Data flow (per scan)

```
config/settings.yaml (keywords + categories)
  → Scrapers (OLX MCP → HTTP fallback, Facebook MCP → Playwright fallback, eBay)
  → RawListing objects (src/scrapers/base.py)
  → agent.process_listing() per listing:
      1. NER tag extraction (models/title_ner/model-best, lazy-loaded singleton)
      2. Canonical eBay query via NER tags (src/title_engine/query_builder.py)
      3. eMAG price scrape — primary RON source, 60% blend weight (src/pricing/emag.py)
      4. eBay sold price scrape — demand signal + global validation, 40% blend (src/pricing/ebay_sold.py)
      5. LLM fallback (Claude Haiku) when confidence < 0.6 (src/pricing/llm_estimator.py)
      6. DealScorer (src/scoring/deal_scorer.py) → 0-100 score, grade S/A/B/C/D
  → SQLite (resellify.db) — listings, price_estimates, deal_scores, price_changes, scan_logs
  → FastAPI dashboard (src/dashboard/app.py) at :8000
```

### Key design decisions

**Scraper fallback chain**: Each market has MCP (preferred, no browser) → Playwright/HTTP fallback. OLX MCP uses `npx olx-mcp`; Facebook MCP requires Chrome with an active session. The agent catches any exception from the MCP layer and falls back silently.

**Pricing blend**: eMAG (60%) + eBay sold (40%) when both available. LLM gets 20% weight blended in when market data confidence is low. All prices normalized to RON; eBay USD converted via configurable exchange rates in `config/settings.yaml`.

**Per-scan caches**: `ebay_cache` and `emag_cache` dicts are created fresh each scan and keyed by canonical query string to avoid redundant HTTP requests when multiple listings match the same product.

**NER singleton**: `_get_ner()` in `agent.py` loads the spaCy model once on first call. If `models/title_ner/model-best` doesn't exist, the system falls back to regex/gazetteer prefill (`src/title_engine/cross_checker.prefill`).

**Price-change tracking**: On re-seen listings the agent checks if the RON price shifted. If it dropped, it re-scores cheaply using the stored `estimated_value_ron` without hitting any pricing APIs. `PriceChange` rows record every delta.

**Listing deduplication**: `id` is `platform:external_id`. Listings already in the DB are skipped unless the price changed.

### DB schema (src/db/models.py)

| Table | Purpose |
|---|---|
| `listings` | One row per unique listing; includes `ner_entities` JSON, price-drop fields |
| `price_estimates` | Multiple rows per listing, one per method (`emag`, `ebay_sold`, `llm`) |
| `deal_scores` | One row per listing; `total_score` 0-100, `grade` S/A/B/C/D |
| `price_changes` | Append-only log of every observed price delta |
| `scan_logs` | Per-keyword scan metadata and timing |

`init_db()` runs `CREATE TABLE IF NOT EXISTS` + an `ALTER TABLE` migration to add `ner_entities` to pre-existing databases.

### Title Engine (src/title_engine/)

Offline NER labeling and training pipeline for extracting structured product entities (BRAND, MODEL, VARIANT, CONDITION, etc.) from OLX listing titles.

- `pipeline.py` — orchestrates dedup → Haiku labeling → validate → cross-check → write JSONL
- `labeler.py` — Claude Haiku batch labeler producing BIO tags
- `trainer.py` — trains a spaCy NER model (`xx` lang, tok2vec + NER) on the labeled data
- `inference.py` — `TitleNER` wrapper used at runtime in `agent.py`
- `query_builder.py` — converts NER spans to a clean eBay search query
- `cross_checker.py` — regex/gazetteer prefill used both as a training cross-check and as NER fallback

### Configuration (config/settings.yaml)

All runtime tuning lives here — no config is hardcoded in Python. Key sections:

- `markets.olx/facebook/ebay` — enable/disable, scan intervals, rate limits
- `pricing` — exchange rates, LLM model, eBay lookback window, confidence thresholds
- `scoring.weights` — profit 40%, demand 35%, confidence 15%, risk 10%
- `categories` — keyword lists + OLX category paths + per-category price ranges
- `notifications` — Discord/Telegram webhook (leave blank to disable)
- `agent` — concurrency, listing TTL, pruning age
