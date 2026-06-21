"""
Main orchestrator: schedules scraping, runs pricing and scoring, saves to DB.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import yaml
from sqlalchemy import select, update

from .db.models import (
    AsyncSessionLocal, Listing, PriceEstimate, DealScore, ScanLog,
    PriceChange, init_db
)
from .events import broadcast_deal, broadcast_pipeline
from .scrapers.base import RawListing
from .scrapers.olx_mcp import OLXMCPScraper
from .scrapers.olx import OLXScraper                    # HTTP (httpx+BS4) fallback
from .scrapers.facebook_mcp import FacebookMCPScraper   # MCP primary
from .scrapers.facebook import FacebookMarketplaceScraper # Playwright fallback
from .scrapers.ebay_scraper import EbayScraper
from .pricing.ebay_sold import EbaySoldPricer, EbayMarketData
from .pricing.emag import EmagPricer, EmagMarketData
from .pricing.olx_pricer import OLXPricer, OLXMarketData
from .pricing.facebook_pricer import FacebookPricer, FBMarketData
from .pricing.llm_estimator import LLMPriceEstimator
from .scoring.deal_scorer import DealScorer
from .title_engine.tokenizer import tokenize
from .title_engine.cross_checker import prefill
from .title_engine.query_builder import build_ebay_query

logger = logging.getLogger(__name__)

_running = False
_last_scan: dict[str, datetime] = {}
_scan_lock = asyncio.Lock()  # ensures only one scan runs at a time

# Lazy NER singleton — loaded once on first use if the trained model exists.
_ner = None
_ner_loaded = False

def _get_ner():
    global _ner, _ner_loaded
    if _ner_loaded:
        return _ner
    _ner_loaded = True
    model_path = __import__("pathlib").Path("models/title_ner/model-best")
    if model_path.exists():
        try:
            from .title_engine.inference import TitleNER
            _ner = TitleNER(model_path)
            logger.info(f"[Agent] NER model loaded from {model_path}")
        except Exception as e:
            logger.info(f"[Agent] NER model unavailable: {e}")
    return _ner


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


def _demand_from_olx(listing_count: int) -> float:
    """
    Estimate market *liquidity* 0-100 from the number of comparable active
    listings across used marketplaces.

    This is a supply-side proxy: a healthy number of comparable listings means
    the item is actively traded (easy to research, easy to resell), while zero
    comparables means an illiquid/unknown item we cannot vouch for. It is NOT
    a sales-velocity measure, so the ceiling is capped well below 100 and a
    zero count yields a near-zero score rather than a misleading floor.
    """
    if listing_count <= 0:  return 0.0    # no comparable market — illiquid/unknown
    if listing_count >= 30: return 45.0
    if listing_count >= 15: return 35.0
    if listing_count >= 7:  return 25.0
    if listing_count >= 3:  return 18.0
    return 10.0                           # 1-2 listings — very thin

async def process_listing(
    raw: RawListing,
    emag_pricer: EmagPricer,
    olx_pricer: OLXPricer,
    fb_pricer: Optional[FacebookPricer],
    llm_estimator: Optional[LLMPriceEstimator],
    scorer: DealScorer,
    pricing_cfg: dict,
    scoring_cfg: dict,
    price_config: dict,
    emag_cache: Optional[dict[str, EmagMarketData]] = None,
    olx_cache: Optional[dict[str, OLXMarketData]] = None,
    fb_cache: Optional[dict[str, FBMarketData]] = None,
) -> bool:
    """
    Price and score a single listing.
    Three short-lived DB sessions — no write lock ever held during HTTP calls.
    """
    price_ron = convert_price_to_ron(raw.price, raw.currency, price_config)
    min_profit_ron = scoring_cfg.get("min_profit_ron", 50)

    # ── Session 1: check existing / fast-path update ─────────────────────────
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Listing).where(Listing.id == raw.listing_id))
        existing = result.scalar_one_or_none()
        if existing:
            old_price = existing.price_ron or price_ron
            price_dropped = False
            if abs(price_ron - old_price) >= 0.01:
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
                logger.info(f"[Agent] Price change on '{existing.title[:40]}': "
                      f"{old_price:.0f} -> {price_ron:.0f} RON ({delta_pct:+.1f}%)")
            existing.last_seen_at = datetime.utcnow()
            existing.is_active = True
            if not price_dropped:
                await session.commit()
                return False
            # Price dropped — re-score cheaply without any HTTP calls
            score_res = await session.execute(select(DealScore).where(DealScore.listing_id == raw.listing_id))
            ds = score_res.scalar_one_or_none()
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

        # New listing — commit immediately so no write lock is held during HTTP calls
        _tokens = tokenize(raw.title)
        ner = _get_ner()
        ner_entities: dict = {}
        if ner is not None:
            try:
                spans = ner.tag(raw.title)
                for sp in spans:
                    label = sp.label.lower()
                    if label not in ner_entities:
                        ner_entities[label] = " ".join(_tokens[sp.start:sp.end])
                _tags = ner.tag_bio(raw.title)[1]
            except Exception as e:
                logger.info(f"[Agent] NER error on '{raw.title[:40]}': {e}")
                _tags = prefill(_tokens)
        else:
            _tags = prefill(_tokens)

        session.add(Listing(
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
            ner_entities=ner_entities,
        ))
        await session.commit()
    # write lock released — session 1 closed

    # ── HTTP pricing calls — NO open DB session ───────────────────────────────
    search_query = build_ebay_query(_tokens, _tags)
    if search_query != raw.title[:80]:
        logger.info(f"[Agent] Canonical query: {search_query!r}  (was: {raw.title[:60]!r})")

    olx_cache_key = f"{search_query}|{int(price_ron)}"
    if olx_cache is not None and olx_cache_key in olx_cache:
        olx_data = olx_cache[olx_cache_key]
    else:
        olx_data = await olx_pricer.get_market_data(search_query, price_hint=price_ron)
        if olx_cache is not None:
            olx_cache[olx_cache_key] = olx_data

    olx_value_ron = 0.0
    olx_confidence = 0.0
    if olx_data.listing_count > 0:
        olx_value_ron = olx_data.median_price_ron
        olx_confidence = olx_data.confidence
        logger.info(f"[Agent] OLX market: {olx_data.listing_count} listings, "
              f"median {olx_value_ron:.0f} RON for {search_query!r}")

    # Facebook Marketplace — second used-market source
    fb_data: Optional[FBMarketData] = None
    fb_value_ron = 0.0
    fb_confidence = 0.0
    if fb_pricer is not None:
        fb_cache_key = f"{search_query}|{int(price_ron)}"
        if fb_cache is not None and fb_cache_key in fb_cache:
            fb_data = fb_cache[fb_cache_key]
        else:
            fb_data = await fb_pricer.get_market_data(search_query, price_hint=price_ron)
            if fb_cache is not None:
                fb_cache[fb_cache_key] = fb_data
        if fb_data and fb_data.listing_count > 0:
            fb_value_ron = fb_data.median_price_ron
            fb_confidence = fb_data.confidence
            logger.info(f"[Agent] FB market: {fb_data.listing_count} listings, "
                  f"median {fb_value_ron:.0f} RON for {search_query!r}")

    emag_cache_key = search_query
    if emag_cache is not None and emag_cache_key in emag_cache:
        emag_data = emag_cache[emag_cache_key]
    else:
        emag_data = await emag_pricer.get_market_data(search_query, price_hint=price_ron)
        if emag_cache is not None:
            emag_cache[emag_cache_key] = emag_data

    emag_ceiling_ron = 0.0
    if emag_data.listing_count > 0:
        emag_ceiling_ron = emag_data.median_price_ron * 0.90
        logger.info(f"[Agent] eMAG ceiling: {emag_ceiling_ron:.0f} RON (new median {emag_data.median_price_ron:.0f})")

    _fb_count = fb_data.listing_count if fb_data else 0
    demand_score = _demand_from_olx(olx_data.listing_count + _fb_count)

    # Blend the used-market sources (OLX + Facebook) by confidence weighting.
    # Both reflect what used items actually sell for locally.
    used_parts: list[tuple[float, float]] = []  # (value, weight)
    if olx_value_ron > 0:
        used_parts.append((olx_value_ron, max(olx_confidence, 0.1)))
    if fb_value_ron > 0:
        used_parts.append((fb_value_ron, max(fb_confidence, 0.1)))

    if used_parts:
        wsum = sum(w for _, w in used_parts)
        used_value_ron = sum(v * w for v, w in used_parts) / wsum
        # Confidence rises when both sources agree/exist.
        used_confidence = min(1.0, max(c for _, c in used_parts) + (0.1 if len(used_parts) > 1 else 0.0))
        # eMAG = NEW retail; used value must never exceed the new-price ceiling.
        if emag_ceiling_ron > 0:
            final_value_ron = min(used_value_ron, emag_ceiling_ron)
        else:
            final_value_ron = used_value_ron
        final_confidence = used_confidence
    elif emag_ceiling_ron > 0:
        # No used data \
        final_value_ron = emag_data.median_price_ron * 0.65
        final_confidence = emag_data.confidence * 0.5
    else:
        final_value_ron = 0.0
        final_confidence = 0.0

    # LLM fallback when confidence is too low
    min_for_confidence = pricing_cfg.get("min_sold_for_medium_confidence", 3)
    if llm_estimator is not None and pricing_cfg.get("llm_fallback_enabled", True) and final_confidence < 0.4:
        try:
            llm_est = await llm_estimator.estimate(
                title=raw.title, description=raw.description or "",
                condition=raw.condition, platform=raw.platform,
                category=raw.category, ner_entities=ner_entities or None,
            )
            if llm_est.estimated_value_usd > 0 and not llm_est.error:
                llm_value_ron = llm_estimator.to_ron(llm_est.estimated_value_usd)
                if final_value_ron > 0:
                    final_value_ron = final_value_ron * 0.8 + llm_value_ron * 0.2
                    final_confidence = max(final_confidence, llm_est.confidence * 0.6)
                else:
                    final_value_ron = llm_value_ron * 0.65
                    final_confidence = llm_est.confidence * 0.5
                    demand_score = max(demand_score, 20.0)
        except Exception:
            pass

    if final_value_ron <= 0:
        broadcast_pipeline({"type": "scored", "id": raw.listing_id, "score": 0, "is_deal": False})
        return False

    scored = scorer.score(
        asking_price_ron=price_ron,
        estimated_value_ron=final_value_ron,
        demand_score=demand_score,
        confidence=final_confidence,
        condition=raw.condition,
        sold_count=olx_data.listing_count,
        recent_sold_30d=0,
        platform=raw.platform,
    )

    # ── Session 2: save price estimates + score ───────────────────────────────
    async with AsyncSessionLocal() as session:
        if olx_data.listing_count > 0:
            session.add(PriceEstimate(
                listing_id=raw.listing_id, method="olx_market",
                estimated_value_ron=olx_value_ron, confidence=olx_confidence,
                sold_count=olx_data.listing_count,
                avg_price_ron=olx_data.avg_price_ron,
                min_price_ron=olx_data.min_price_ron,
                max_price_ron=olx_data.max_price_ron,
                raw_data={
                    "query": search_query,
                    "entries": [
                        {"title": e.title, "price_ron": e.price_ron, "url": e.url,
                         "image": getattr(e, "image", "")}
                        for e in olx_data.entries[:12]
                    ],
                },
            ))
        if fb_data and fb_data.listing_count > 0:
            session.add(PriceEstimate(
                listing_id=raw.listing_id, method="facebook_market",
                estimated_value_ron=fb_value_ron, confidence=fb_confidence,
                sold_count=fb_data.listing_count,
                avg_price_ron=fb_data.avg_price_ron,
                min_price_ron=fb_data.min_price_ron,
                max_price_ron=fb_data.max_price_ron,
                raw_data={
                    "query": search_query,
                    "entries": [
                        {"title": e.title, "price_ron": e.price_ron, "url": e.url,
                         "image": getattr(e, "image", "")}
                        for e in fb_data.entries[:12]
                    ],
                },
            ))
        if emag_data.listing_count > 0:
            session.add(PriceEstimate(
                listing_id=raw.listing_id, method="emag",
                estimated_value_ron=emag_data.median_price_ron,
                confidence=emag_data.confidence,
                sold_count=emag_data.listing_count,
                avg_price_ron=emag_data.avg_price_ron,
                min_price_ron=emag_data.min_price_ron,
                max_price_ron=emag_data.max_price_ron,
                raw_data={
                    "query": search_query,
                    "ceiling": True,
                    "entries": [
                        {"title": e.title, "price_ron": e.price_ron, "url": e.url,
                         "image": getattr(e, "image", "")}
                        for e in emag_data.listings[:12]
                    ],
                },
            ))
        session.add(DealScore(
            listing_id=raw.listing_id,
            total_score=scored.total_score,
            profit_score=scored.profit_score,
            demand_score=scored.demand_score,
            confidence_score=scored.confidence_score,
            risk_score=scored.risk_score,
            asking_price_ron=price_ron,
            estimated_value_ron=final_value_ron,
            estimated_profit_ron=scored.estimated_profit_ron,
            profit_percent=scored.profit_percent,
            notes=scored.notes,
        ))
        await session.commit()

    is_deal = scored.is_good_deal and scored.estimated_profit_ron >= min_profit_ron
    broadcast_pipeline({
        "type": "scored", "id": raw.listing_id,
        "score": round(scored.total_score, 1), "is_deal": is_deal,
        "profit_ron": round(scored.estimated_profit_ron) if is_deal else 0,
    })
    if is_deal:
        broadcast_deal({
            "id": raw.listing_id, "title": raw.title, "url": raw.url,
            "platform": raw.platform, "category": raw.category,
            "condition": raw.condition, "location": raw.location,
            "image": raw.images[0] if raw.images else "",
            "total_score": scored.total_score, "profit_score": scored.profit_score,
            "demand_score": scored.demand_score, "confidence_score": scored.confidence_score,
            "asking_price_ron": price_ron, "estimated_value_ron": final_value_ron,
            "estimated_profit_ron": scored.estimated_profit_ron,
            "profit_percent": scored.profit_percent,
            "ner_entities": ner_entities,
        })
    return is_deal


async def run_scan(config: dict):
    if _scan_lock.locked():
        logger.info("[Agent] Scan already in progress — skipping duplicate request")
        return 0
    async with _scan_lock:
        return await _run_scan(config)


async def _run_scan(config: dict):
    pricing_cfg = config.get("pricing", {})
    scoring_cfg = config.get("scoring", {})
    categories = config.get("categories", [])
    markets = config.get("markets", {})
    price_config = {
        "usd_to_ron_rate": pricing_cfg.get("usd_to_ron_rate", 4.55),
        "eur_to_ron_rate": pricing_cfg.get("eur_to_ron_rate", 4.97),
        "gbp_to_ron_rate": pricing_cfg.get("gbp_to_ron_rate", 5.75),
    }

    # eBay sold endpoint returns 403 — skipping. Demand is derived from OLX market depth.

    llm_estimator: Optional[LLMPriceEstimator] = None
    if pricing_cfg.get("llm_fallback_enabled", True):
        try:
            llm_estimator = LLMPriceEstimator(
                config=pricing_cfg,
                usd_to_ron=pricing_cfg.get("usd_to_ron_rate", 4.55),
            )
        except Exception as e:
            logger.info(f"[Agent] LLM estimator unavailable: {e}")

    emag_pricer = EmagPricer()
    olx_pricer = OLXPricer(pages=1)
    # Facebook used-market pricer (disabled gracefully when no cookies)
    fb_pricer: Optional[FacebookPricer] = None
    _fb_market = markets.get("facebook", {})
    if _fb_market.get("enabled", False):
        # Merge FX rates so the FB pricer can normalize EUR/USD to RON.
        _fb_pricer_cfg = {**_fb_market, **price_config}
        fb_pricer = FacebookPricer(_fb_pricer_cfg)
        try:
            _ok, _why = await fb_pricer._scraper.preflight()
            if _ok:
                logger.info("[Agent] Facebook pricing: available")
            else:
                logger.info(f"[Agent] Facebook pricing DISABLED \u2014 {_why}")
                broadcast_pipeline({"type": "warning", "source": "facebook", "msg": _why})
                await fb_pricer.close()
                fb_pricer = None
        except Exception as e:
            logger.info(f"[Agent] Facebook preflight error: {e}")
            await fb_pricer.close()
            fb_pricer = None
    scorer = DealScorer(scoring_cfg)
    active_categories = [c for c in categories if c.get("enabled", True)]

    # Signal scan started immediately so the pipeline UI opens right away.
    # total=0 here — updated below once scraping is done.
    broadcast_pipeline({"type": "scan_start", "total": 0})

    all_listings: list[RawListing] = []
    olx_listings_all: list[RawListing] = []
    fb_listings_all: list[RawListing] = []

    async def _scrape_olx() -> list[RawListing]:
        if not markets.get("olx", {}).get("enabled", True):
            return []
        broadcast_pipeline({"type": "scraping", "platform": "olx",
                            "msg": f"Scraping OLX ({len(active_categories)} categories)..."})
        olx_listings: list[RawListing] = []
        if markets["olx"].get("use_mcp", True):
            try:
                olx_mcp = OLXMCPScraper(markets["olx"])
                olx_listings = await olx_mcp.scrape_all_keywords(active_categories)
                await olx_mcp.close()
            except Exception as e:
                logger.info(f"[Agent] OLX-MCP unavailable ({e}), falling back to HTTP scraper")
        if not olx_listings:
            olx = OLXScraper(markets["olx"])
            try:
                olx_listings = await olx.scrape_all_keywords(active_categories)
            finally:
                await olx.close()
        logger.info(f"[Agent] OLX: {len(olx_listings)} listings found")
        return olx_listings

    async def _scrape_fb() -> list[RawListing]:
        if not markets.get("facebook", {}).get("enabled", False):
            return []
        fb_cfg = markets["facebook"]
        broadcast_pipeline({"type": "scraping", "platform": "facebook",
                            "msg": "Scraping Facebook Marketplace..."})
        fb_listings: list[RawListing] = []
        # Try GraphQL MCP scraper first (macOS, needs Chrome session)
        if fb_cfg.get("mcp_server_path") or fb_cfg.get("use_mcp", False):
            try:
                fb_mcp = FacebookMCPScraper(fb_cfg)
                fb_listings = await fb_mcp.scrape_all_keywords(active_categories)
                await fb_mcp.close()
            except Exception as e:
                logger.info(f"[Agent] FB-MCP unavailable ({e}), falling back to Playwright")
        if not fb_listings:
            fb = FacebookMarketplaceScraper(fb_cfg)
            try:
                _ok, _why = await fb.preflight()
                if not _ok:
                    logger.info(f"[Agent] Facebook scraping skipped \u2014 {_why}")
                    broadcast_pipeline({"type": "warning", "source": "facebook", "msg": _why})
                    return []
                fb_listings = await fb.scrape_all_keywords(active_categories)
            finally:
                await fb.close()
        logger.info(f"[Agent] Facebook: {len(fb_listings)} listings found")
        return fb_listings

    # Scrape both marketplaces concurrently \
    olx_listings_all, fb_listings_all = await asyncio.gather(
        _scrape_olx(), _scrape_fb(),
    )

    # Round-robin interleave so Facebook results surface alongside OLX
    # instead of being starved at the tail of a large OLX batch.
    from itertools import zip_longest
    _sentinel = object()
    for o, f in zip_longest(olx_listings_all, fb_listings_all, fillvalue=_sentinel):
        if o is not _sentinel:
            all_listings.append(o)
        if f is not _sentinel:
            all_listings.append(f)

    total = len(all_listings)
    logger.info(f"[Agent] Total listings collected: {total}")
    broadcast_pipeline({"type": "scan_start", "total": total})

    emag_cache: dict[str, EmagMarketData] = {}
    olx_cache: dict[str, OLXMarketData] = {}
    fb_cache: dict[str, FBMarketData] = {}
    good_deals = 0
    for i, raw in enumerate(all_listings):
        broadcast_pipeline({
            "type": "listing", "idx": i + 1, "total": total,
            "id": raw.listing_id, "title": raw.title[:70],
            "platform": raw.platform, "price": raw.price, "currency": raw.currency,
        })
        try:
            is_deal = await process_listing(
                raw, emag_pricer, olx_pricer, fb_pricer, llm_estimator, scorer,
                pricing_cfg, scoring_cfg, price_config,
                emag_cache=emag_cache, olx_cache=olx_cache, fb_cache=fb_cache,
            )
            if is_deal:
                good_deals += 1
        except Exception as e:
            logger.info(f"[Agent] Error processing '{raw.title[:50]}': {e}")
            broadcast_pipeline({"type": "scored", "id": raw.listing_id, "score": 0, "is_deal": False, "error": True})

    await emag_pricer.close()
    await olx_pricer.close()
    if fb_pricer is not None:
        await fb_pricer.close()
    broadcast_pipeline({"type": "scan_done", "deals": good_deals, "total": total})
    logger.info(f"[Agent] Scan complete. Good deals found: {good_deals}/{total}")
    return good_deals


async def start_agent(config_path: str = "config/settings.yaml"):
    global _running
    config = load_config(config_path)
    await init_db()
    _running = True
    logger.info("[Agent] Starting market watch agent...")

    while _running:
        try:
            await run_scan(config)
        except Exception as e:
            logger.info(f"[Agent] Scan error: {e}")
        interval = config.get("markets", {}).get("olx", {}).get("scan_interval_minutes", 30)
        logger.info(f"[Agent] Next scan in {interval} minutes...")
        await asyncio.sleep(interval * 60)


def stop_agent():
    global _running
    _running = False
