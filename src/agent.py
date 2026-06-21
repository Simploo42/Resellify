"""
Main orchestrator: schedules scraping, runs pricing and scoring, saves to DB.
"""
import asyncio
import time
from datetime import datetime
from typing import Optional

import yaml
from sqlalchemy import select, update

from .db.models import (
    AsyncSessionLocal, Listing, PriceEstimate, DealScore, ScanLog,
    PriceChange, init_db
)
from .scrapers.base import RawListing
from .scrapers.olx_mcp import OLXMCPScraper
from .scrapers.olx import OLXScraper                    # HTTP (httpx+BS4) fallback
from .scrapers.facebook_mcp import FacebookMCPScraper   # MCP primary
from .scrapers.facebook import FacebookMarketplaceScraper # Playwright fallback
from .scrapers.ebay_scraper import EbayScraper
from .pricing.ebay_sold import EbaySoldPricer
from .pricing.llm_estimator import LLMPriceEstimator
from .scoring.deal_scorer import DealScorer

_running = False
_last_scan: dict[str, datetime] = {}


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def convert_price_to_ron(price: float, currency: str, config: dict) -> float:
    rates = {
        "RON": 1.0,
        "EUR": config.get("eur_to_ron_rate", 4.97),
        "USD": config.get("usd_to_ron_rate", 4.55),
        "GBP": config.get("gbp_to_ron_rate", 5.75),
    }
    return round(price * rates.get(currency.upper(), 1.0), 2)


async def process_listing(
    raw: RawListing,
    ebay_pricer: EbaySoldPricer,
    llm_estimator: Optional[LLMPriceEstimator],
    scorer: DealScorer,
    pricing_cfg: dict,
    scoring_cfg: dict,
    price_config: dict,
) -> bool:
    """Price and score a single listing. Returns True if it's a good deal."""
    price_ron = convert_price_to_ron(raw.price, raw.currency, price_config)
    min_profit = scoring_cfg.get("min_profit_percent", 15)
    min_profit_ron = scoring_cfg.get("min_profit_ron", 50)

    async with AsyncSessionLocal() as session:
        # Check if we've already processed this listing recently
        result = await session.execute(
            select(Listing).where(Listing.id == raw.listing_id)
        )
        existing = result.scalar_one_or_none()
        if existing:
            price_dropped = False
            old_price = existing.price_ron or price_ron
            if abs(price_ron - old_price) >= 0.01:
                # Price changed since we last saw this listing — record it.
                delta = price_ron - old_price
                delta_pct = (delta / old_price * 100) if old_price else 0.0
                session.add(PriceChange(
                    listing_id=raw.listing_id,
                    old_price_ron=old_price,
                    new_price_ron=price_ron,
                    delta_ron=round(delta, 2),
                    delta_pct=round(delta_pct, 1),
                ))
                first_price = existing.first_price_ron or old_price
                total_drop = first_price - price_ron
                existing.price = raw.price
                existing.currency = raw.currency
                existing.price_ron = price_ron
                existing.price_drop_ron = round(total_drop, 2)
                existing.price_drop_pct = round(total_drop / first_price * 100, 1) if first_price else 0.0
                existing.price_changes = (existing.price_changes or 0) + 1
                price_dropped = delta < 0
                print(f"[Agent] Price change on '{existing.title[:40]}': "
                      f"{old_price:.0f} -> {price_ron:.0f} RON ({delta_pct:+.1f}%)")

            existing.last_seen_at = datetime.utcnow()
            existing.is_active = True

            if not price_dropped:
                await session.commit()
                return False
            # Price dropped — re-score against the existing estimate (cheap path)
            score_result = await session.execute(
                select(DealScore).where(DealScore.listing_id == raw.listing_id)
            )
            ds = score_result.scalar_one_or_none()
            if ds and ds.estimated_value_ron > 0:
                rescored = scorer.score(
                    asking_price_ron=price_ron,
                    estimated_value_ron=ds.estimated_value_ron,
                    demand_score=ds.demand_score,
                    confidence=(ds.confidence_score / 100.0),
                    condition=existing.condition,
                    platform=existing.platform,
                )
                ds.total_score = rescored.total_score
                ds.profit_score = rescored.profit_score
                ds.asking_price_ron = price_ron
                ds.estimated_profit_ron = rescored.estimated_profit_ron
                ds.profit_percent = rescored.profit_percent
                ds.notes = rescored.notes + [f"Price dropped to {price_ron:.0f} RON"]
                ds.updated_at = datetime.utcnow()
                await session.commit()
                return rescored.is_good_deal and rescored.estimated_profit_ron >= min_profit_ron
            await session.commit()
            return False

        # Save listing
        listing = Listing(
            id=raw.listing_id,
            platform=raw.platform,
            external_id=raw.external_id,
            title=raw.title,
            description=raw.description or "",
            price=raw.price,
            currency=raw.currency,
            price_ron=price_ron,
            first_price_ron=price_ron,
            condition=raw.condition,
            location=raw.location,
            url=raw.url,
            images=raw.images,
            seller_name=raw.seller_name,
            seller_url=raw.seller_url,
            category=raw.category,
            matched_keyword=raw.matched_keyword,
            posted_at=raw.posted_at,
        )
        session.add(listing)
        await session.flush()

        # ── eBay pricing ────────────────────────────────────────────────────
        search_query = raw.title[:80]
        ebay_data = await ebay_pricer.get_market_data(search_query)
        usd_to_ron = price_config.get("usd_to_ron_rate", 4.55)

        ebay_value_ron = 0.0
        ebay_confidence = 0.0
        demand_score = 0.0
        sold_count = 0
        recent_30d = 0

        if ebay_data.avg_price_usd > 0:
            ebay_value_ron = ebay_pricer.to_ron(ebay_data.avg_price_usd)
            ebay_confidence = ebay_data.confidence
            demand_score = ebay_pricer.demand_score(ebay_data)
            sold_count = ebay_data.sold_count
            recent_30d = ebay_data.recent_sold_30d

            pe = PriceEstimate(
                listing_id=raw.listing_id,
                method="ebay_sold",
                estimated_value_ron=ebay_value_ron,
                confidence=ebay_confidence,
                sold_count=sold_count,
                avg_price_ron=ebay_pricer.to_ron(ebay_data.avg_price_usd),
                min_price_ron=ebay_pricer.to_ron(ebay_data.min_price_usd),
                max_price_ron=ebay_pricer.to_ron(ebay_data.max_price_usd),
                raw_data={
                    "recent_sold_30d": recent_30d,
                    "recent_sold_7d": ebay_data.recent_sold_7d,
                    "query": search_query,
                },
            )
            session.add(pe)

        # ── LLM fallback ────────────────────────────────────────────────────
        final_value_ron = ebay_value_ron
        final_confidence = ebay_confidence

        min_for_confidence = pricing_cfg.get("min_sold_for_medium_confidence", 3)
        needs_llm = (
            llm_estimator is not None
            and pricing_cfg.get("llm_fallback_enabled", True)
            and (ebay_confidence < 0.6 or sold_count < min_for_confidence)
        )

        if needs_llm:
            llm_est = await llm_estimator.estimate(
                title=raw.title,
                description=raw.description or "",
                condition=raw.condition,
                platform=raw.platform,
                category=raw.category,
            )
            if llm_est.estimated_value_usd > 0 and not llm_est.error:
                llm_value_ron = llm_estimator.to_ron(llm_est.estimated_value_usd)
                session.add(PriceEstimate(
                    listing_id=raw.listing_id,
                    method="llm",
                    estimated_value_ron=llm_value_ron,
                    confidence=llm_est.confidence,
                    llm_reasoning=llm_est.reasoning,
                    raw_data={"demand_notes": llm_est.demand_notes},
                ))

                if ebay_value_ron > 0:
                    # Blend: eBay is more trusted
                    blend_weight = min(0.8, ebay_confidence)
                    final_value_ron = ebay_value_ron * blend_weight + llm_value_ron * (1 - blend_weight)
                    final_confidence = max(ebay_confidence, llm_est.confidence * 0.7)
                else:
                    final_value_ron = llm_value_ron
                    final_confidence = llm_est.confidence * 0.7  # LLM alone is less reliable
                    demand_score = max(demand_score, 25.0)  # assume some demand if LLM has a price

        # ── Score ────────────────────────────────────────────────────────────
        if final_value_ron <= 0:
            await session.commit()
            return False

        result = scorer.score(
            asking_price_ron=price_ron,
            estimated_value_ron=final_value_ron,
            demand_score=demand_score,
            confidence=final_confidence,
            condition=raw.condition,
            sold_count=sold_count,
            recent_sold_30d=recent_30d,
            platform=raw.platform,
        )

        session.add(DealScore(
            listing_id=raw.listing_id,
            total_score=result.total_score,
            profit_score=result.profit_score,
            demand_score=result.demand_score,
            confidence_score=result.confidence_score,
            risk_score=result.risk_score,
            asking_price_ron=price_ron,
            estimated_value_ron=final_value_ron,
            estimated_profit_ron=result.estimated_profit_ron,
            profit_percent=result.profit_percent,
            notes=result.notes,
        ))

        await session.commit()
        return result.is_good_deal and result.estimated_profit_ron >= min_profit_ron


async def run_scan(config: dict):
    pricing_cfg = config.get("pricing", {})
    scoring_cfg = config.get("scoring", {})
    categories = config.get("categories", [])
    markets = config.get("markets", {})
    price_config = {
        "usd_to_ron_rate": pricing_cfg.get("usd_to_ron_rate", 4.55),
        "eur_to_ron_rate": pricing_cfg.get("eur_to_ron_rate", 4.97),
        "gbp_to_ron_rate": pricing_cfg.get("gbp_to_ron_rate", 5.75),
    }

    ebay_cfg = markets.get("ebay", {})
    ebay_pricer = EbaySoldPricer(
        config={
            "country_tld": ebay_cfg.get("country_tld", "com"),
            "ebay_sold_lookback_days": pricing_cfg.get("ebay_sold_lookback_days", 90),
        },
        usd_to_ron=pricing_cfg.get("usd_to_ron_rate", 4.55),
        eur_to_ron=pricing_cfg.get("eur_to_ron_rate", 4.97),
    )

    llm_estimator: Optional[LLMPriceEstimator] = None
    if pricing_cfg.get("llm_fallback_enabled", True):
        try:
            llm_estimator = LLMPriceEstimator(
                config=pricing_cfg,
                usd_to_ron=pricing_cfg.get("usd_to_ron_rate", 4.55),
            )
        except Exception as e:
            print(f"[Agent] LLM estimator unavailable: {e}")

    scorer = DealScorer(scoring_cfg)
    active_categories = [c for c in categories if c.get("enabled", True)]

    # Collect all listings from enabled markets
    all_listings: list[RawListing] = []

    if markets.get("olx", {}).get("enabled", True):
        # Try MCP-based scraper first; fall back to HTTP scraper if npx unavailable
        olx_listings: list[RawListing] = []
        try:
            olx_mcp = OLXMCPScraper(markets["olx"])
            olx_listings = await olx_mcp.scrape_all_keywords(active_categories)
            await olx_mcp.close()
        except Exception as e:
            print(f"[Agent] OLX-MCP unavailable ({e}), falling back to HTTP scraper")
            olx = OLXScraper(markets["olx"])
            try:
                olx_listings = await olx.scrape_all_keywords(active_categories)
            finally:
                await olx.close()
        all_listings.extend(olx_listings)
        print(f"[Agent] OLX: {len(olx_listings)} listings found")

    if markets.get("facebook", {}).get("enabled", False):
        fb_listings: list[RawListing] = []
        fb_cfg = markets["facebook"]
        # Try GraphQL MCP scraper first (macOS, needs Chrome session)
        if fb_cfg.get("mcp_server_path") or fb_cfg.get("use_mcp", False):
            try:
                fb_mcp = FacebookMCPScraper(fb_cfg)
                fb_listings = await fb_mcp.scrape_all_keywords(active_categories)
                await fb_mcp.close()
            except Exception as e:
                print(f"[Agent] FB-MCP unavailable ({e}), falling back to Playwright")
        if not fb_listings:
            fb = FacebookMarketplaceScraper(fb_cfg)
            try:
                fb_listings = await fb.scrape_all_keywords(active_categories)
            finally:
                await fb.close()
        all_listings.extend(fb_listings)
        print(f"[Agent] Facebook: {len(fb_listings)} listings found")

    print(f"[Agent] Total listings collected: {len(all_listings)}")

    good_deals = 0
    for raw in all_listings:
        try:
            is_deal = await process_listing(
                raw, ebay_pricer, llm_estimator, scorer,
                pricing_cfg, scoring_cfg, price_config
            )
            if is_deal:
                good_deals += 1
        except Exception as e:
            print(f"[Agent] Error processing '{raw.title}': {e}")

    print(f"[Agent] Scan complete. Good deals found: {good_deals}/{len(all_listings)}")
    return good_deals


async def start_agent(config_path: str = "config/settings.yaml"):
    global _running
    config = load_config(config_path)
    await init_db()
    _running = True
    print("[Agent] Starting market watch agent...")

    while _running:
        try:
            await run_scan(config)
        except Exception as e:
            print(f"[Agent] Scan error: {e}")
        interval = config.get("markets", {}).get("olx", {}).get("scan_interval_minutes", 30)
        print(f"[Agent] Next scan in {interval} minutes...")
        await asyncio.sleep(interval * 60)


def stop_agent():
    global _running
    _running = False
