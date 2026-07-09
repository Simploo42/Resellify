# Resellify

Market resell opportunity finder — watches OLX Romania, Facebook Marketplace, and eBay for flip deals.

## Stack
- Python 3.11+, FastAPI, SQLite (aiosqlite), Playwright, Groq LLM API (OpenAI-compatible; Anthropic fallback)
- Node.js (for olx-mcp and facebook-marketplace-mcp MCP servers)

## Setup
```bash
cp .env.example .env        # add GROQ_API_KEY (or ANTHROPIC_API_KEY as fallback)
pip install -r requirements.txt
playwright install chromium
python main.py              # dashboard at http://localhost:8000
```

## Architecture
- `src/scrapers/olx_mcp.py` — OLX Romania via olx-mcp MCP server (primary)
- `src/scrapers/olx.py` — OLX Romania via Playwright (fallback)
- `src/scrapers/facebook_mcp.py` — Facebook Marketplace via GraphQL MCP server (macOS)
- `src/scrapers/facebook.py` — Facebook Marketplace via Playwright (fallback)
- `src/scrapers/ebay_scraper.py` — eBay listings scraper
- `src/pricing/ebay_sold.py` — eBay completed/sold listings for market value + demand
- `src/pricing/llm_estimator.py` — LLM price fallback (Groq preferred, Anthropic fallback) when eBay data is sparse
- `src/scoring/deal_scorer.py` — Weighted deal scorer (profit 40%, demand 35%, confidence 15%, risk 10%)
- `src/dashboard/` — FastAPI + Jinja2 + Tailwind dashboard
- `src/agent.py` — Orchestrator with APScheduler-style async scan loop
- `config/settings.yaml` — All user config: keywords, categories, price ranges, thresholds, weights

## Running
```bash
python main.py              # dashboard only (manual scan via UI)
python main.py --agent      # dashboard + auto-scan background agent
python main.py --scan       # single scan run, then exit
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
