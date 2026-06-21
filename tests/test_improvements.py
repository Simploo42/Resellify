"""
Offline unit tests for the architecture / scoring improvements.

No network, no DB, no pytest dependency. Run with:

    python -m tests.test_improvements      # from the project root
    python tests/test_improvements.py

Exits non-zero on the first failed assertion.
"""
import asyncio
import sys
import time
from pathlib import Path

# Allow `python tests/test_improvements.py` from anywhere by putting the
# project root (parent of this file's directory) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pricing.normalize import (
    price_stats,
    confidence_from_count,
    iqr_filter,
    is_accessory,
)
from src.scoring.deal_scorer import DealScorer, grade_for, _grade
from src.scrapers.base import BaseScraper, RawListing
from src.agent import _demand_from_olx


_passed = 0


def check(cond: bool, msg: str) -> None:
    global _passed
    if not cond:
        raise AssertionError(msg)
    _passed += 1
    print(f"  ok  {msg}")


# ── normalize.price_stats ────────────────────────────────────────────────────

def test_price_stats():
    print("price_stats")
    empty = price_stats([])
    check(empty.count == 0 and empty.median == 0.0, "empty list -> zeros")

    st = price_stats([10.0, 20.0, 30.0, 40.0])
    check(st.count == 4, "count")
    check(st.min == 10.0 and st.max == 40.0, "min/max")
    check(st.avg == 25.0, "avg")
    check(st.median == 25.0, "median (even)")

    st2 = price_stats([5.0, 9.0, 100.0])
    check(st2.median == 9.0, "median (odd)")


def test_confidence_from_count():
    print("confidence_from_count")
    check(confidence_from_count(0) == 0.25, "base at zero count")
    check(confidence_from_count(5) == 0.5, "rises with count")
    check(confidence_from_count(1000) == 1.0, "capped at 1.0")
    check(confidence_from_count(2, base=0.3, per_item=0.07) == round(0.44, 3),
          "custom base/per_item")


def test_iqr_filter():
    print("iqr_filter")
    # Fewer than 4 points -> returned unchanged.
    check(iqr_filter([1.0, 2.0]) == [1.0, 2.0], "short list untouched")
    # An extreme outlier is dropped.
    data = [100.0, 105.0, 110.0, 95.0, 5000.0]
    kept = iqr_filter(data)
    check(5000.0 not in kept, "extreme outlier dropped")
    check(100.0 in kept, "normal value kept")


def test_is_accessory():
    print("is_accessory")
    check(is_accessory("Husa iPhone 14 silicon") is True, "case flagged")
    check(is_accessory("iPhone 14 128GB") is False, "device not flagged")


# ── scoring: liquidity gate + demand proxy ───────────────────────────────────

def _scorer():
    return DealScorer({
        "weights": {"profit": 0.40, "demand": 0.35, "confidence": 0.15, "risk": 0.10},
        "min_deal_score": 55,
    })


def test_demand_proxy():
    print("_demand_from_olx")
    check(_demand_from_olx(0) == 0.0, "zero comparables -> zero demand (was 8.0)")
    check(_demand_from_olx(1) == 10.0, "very thin market")
    check(_demand_from_olx(30) == 45.0, "ceiling capped at 45")
    check(_demand_from_olx(5) == 18.0, "mid band")


def test_liquidity_gate():
    print("scoring liquidity gate")
    sc = _scorer()

    # User's complaint: fat margin but ZERO market signal must NOT be a deal.
    r = sc.score(
        asking_price_ron=500, estimated_value_ron=1000,
        demand_score=0.0, confidence=0.5, condition="folosit",
        sold_count=0, recent_sold_30d=0, platform="olx",
    )
    check(r.total_score <= 45.0, "illiquid item score capped at 45")
    check(not r.is_good_deal, "illiquid item is not a good deal")
    check(any("No comparable market" in n for n in r.notes), "gate note present")

    # Same margin WITH a healthy market -> a real deal.
    r2 = sc.score(
        asking_price_ron=500, estimated_value_ron=1000,
        demand_score=35.0, confidence=0.7, condition="ca nou",
        sold_count=15, recent_sold_30d=0, platform="olx",
    )
    check(r2.is_good_deal, "healthy-market deal qualifies")
    check(r2.total_score > r.total_score, "market depth raises score")


def test_grade_dedup():
    print("grade_for single source of truth")
    check(_grade is grade_for, "alias points to grade_for")
    check(grade_for(90) == "S", "S")
    check(grade_for(75) == "A", "A")
    check(grade_for(60) == "B", "B")
    check(grade_for(50) == "C", "C")
    check(grade_for(10) == "D", "D")
    # Dashboard imports the same function.
    from src.dashboard.app import grade_for as dash_grade
    check(dash_grade is grade_for, "dashboard reuses scorer grade_for")


# ── scrapers: concurrent keyword fan-out preserves order ─────────────────────

class _FakeScraper(BaseScraper):
    async def _random_delay(self, base=None):
        await asyncio.sleep(0.02)

    async def search(self, keyword, cat):
        await asyncio.sleep(0.05)
        return [RawListing(
            external_id=f"{keyword}-1", platform="fake", title=keyword,
            price=10.0, currency="RON", url="u",
        )]

    async def get_listing_detail(self, url):
        return {}


def test_concurrent_scrape():
    print("concurrent scrape_all_keywords")

    async def run():
        cats = [
            {"name": "C1", "enabled": True, "keywords": list("abcdefgh")},
            {"name": "C2", "enabled": False, "keywords": ["z"]},
        ]
        s = _FakeScraper({"max_concurrent_requests": 4})
        t0 = time.time()
        res = await s.scrape_all_keywords(cats)
        return res, time.time() - t0

    res, dt = asyncio.run(run())
    check(len(res) == 8, "disabled category skipped, 8 results")
    check([r.matched_keyword for r in res] == list("abcdefgh"), "order preserved")
    check(res[0].category == "C1", "category tagged")
    # Serial would be ~8*(0.02+0.05)=0.56s; concurrency=4 should be well under.
    check(dt < 0.45, f"concurrency speedup (took {dt:.2f}s)")


# ── facebook pricer currency normalization ───────────────────────────────────

def test_fb_currency_normalization():
    print("FacebookPricer currency normalization")
    from src.pricing.facebook_pricer import FacebookPricer
    p = FacebookPricer({"eur_to_ron_rate": 5.0, "usd_to_ron_rate": 4.0})
    check(p._fx["EUR"] == 5.0 and p._fx["USD"] == 4.0, "fx rates loaded from config")
    check(round(100 * p._fx.get("EUR"), 2) == 500.0, "EUR converts to RON")
    check(p._fx.get("RON") == 1.0, "RON identity")


# \u2500\u2500 notifications \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_notifier():
    print("Notifier gating + formatting")
    from src.notifications import Notifier

    off = Notifier({"notifications": {}})
    check(off.enabled is False, "no channels -> disabled")
    check(off.qualifies(100) is False, "disabled never qualifies")

    disc = Notifier({"notifications": {"discord_webhook_url": "http://x",
                                       "notify_min_score": 75}})
    check(disc.enabled is True, "discord url -> enabled")
    check(disc.qualifies(80) is True, "score above threshold qualifies")
    check(disc.qualifies(70) is False, "score below threshold rejected")

    tg = Notifier({"notifications": {"telegram_bot_token": "t",
                                     "telegram_chat_id": "c"}})
    check(tg.enabled is True, "telegram token+chat -> enabled")
    tg_half = Notifier({"notifications": {"telegram_bot_token": "t"}})
    check(tg_half.enabled is False, "telegram without chat_id -> disabled")

    summary = Notifier._summary({
        "title": "PS5", "total_score": 82, "asking_price_ron": 800,
        "estimated_value_ron": 1500, "estimated_profit_ron": 700,
        "profit_percent": 46, "platform": "olx", "location": "Cluj",
    })
    check("PS5" in summary and "Score 82" in summary, "summary has title+score")
    check("800 RON" in summary and "+700 RON" in summary, "summary has prices")


# \u2500\u2500 config validation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_validate_config():
    print("validate_config")
    from src.agent import validate_config

    good = {
        "markets": {"olx": {}},
        "categories": [{"name": "C", "keywords": ["x"],
                        "price_range": {"min_ron": 100, "max_ron": 500}}],
        "scoring": {"weights": {"profit": 0.4, "demand": 0.35,
                                "confidence": 0.15, "risk": 0.1}},
    }
    check(validate_config(good) == [], "valid config -> no errors")

    bad_w = {**good, "scoring": {"weights": {"profit": 0.5, "demand": 0.35,
                                             "confidence": 0.15, "risk": 0.1}}}
    check(any("sum to 1.0" in e for e in validate_config(bad_w)),
          "weights not summing to 1 flagged")

    bad_pr = {"markets": {"olx": {}},
              "categories": [{"name": "C", "keywords": ["x"],
                             "price_range": {"min_ron": 500, "max_ron": 100}}]}
    check(any("min_ron" in e for e in validate_config(bad_pr)),
          "inverted price range flagged")

    check(any("category" in e for e in validate_config({"markets": {"olx": {}}})),
          "missing categories flagged")
    check(any("markets" in e for e in validate_config({"categories": good["categories"]})),
          "missing markets flagged")
    check(any("telegram_chat_id" in e for e in validate_config({
              "markets": {"olx": {}}, "categories": good["categories"],
              "notifications": {"telegram_bot_token": "t"}})),
          "telegram half-config flagged")


# \u2500\u2500 pruning \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_prune_old_listings():
    print("_prune_old_listings")
    import os
    from datetime import datetime, timedelta
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    import sqlalchemy as sa
    from src.db import models
    import src.agent as agent

    db_path = "test_prune_unit.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    eng = create_async_engine(f"sqlite+aiosqlite:///./{db_path}")
    Session = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)

    async def run():
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        now = datetime.utcnow()
        async with Session() as s:
            s.add(models.Listing(id="olx:fresh", platform="olx", external_id="1",
                                 title="fresh", price=10, url="u",
                                 last_seen_at=now, scraped_at=now))
            s.add(models.Listing(id="olx:stale", platform="olx", external_id="2",
                                 title="stale", price=10, url="u",
                                 last_seen_at=now - timedelta(days=40),
                                 scraped_at=now - timedelta(days=40)))
            s.add(models.DealScore(listing_id="olx:stale", total_score=60,
                                   asking_price_ron=10, estimated_value_ron=20,
                                   estimated_profit_ron=10, profit_percent=50))
            await s.commit()
        orig = agent.AsyncSessionLocal
        agent.AsyncSessionLocal = Session
        try:
            removed = await agent._prune_old_listings({"agent": {"max_listing_age_days": 30}})
            async with Session() as s:
                ids = [r[0] for r in (await s.execute(sa.select(models.Listing.id))).all()]
                ds = [r[0] for r in (await s.execute(sa.select(models.DealScore.listing_id))).all()]
        finally:
            agent.AsyncSessionLocal = orig
            await eng.dispose()
        return removed, ids, ds

    removed, ids, ds = asyncio.run(run())
    os.remove(db_path)
    check(removed == 1, "one stale listing pruned")
    check(ids == ["olx:fresh"], "fresh listing kept")
    check(ds == [], "orphan deal score cascaded")

    # Disabled when days <= 0.
    check(asyncio.run(agent._prune_old_listings({"agent": {"max_listing_age_days": 0}})) == 0,
          "pruning disabled at days<=0")


def main():
    tests = [
        test_price_stats, test_confidence_from_count, test_iqr_filter,
        test_is_accessory, test_demand_proxy, test_liquidity_gate,
        test_grade_dedup, test_concurrent_scrape, test_fb_currency_normalization,
        test_notifier, test_validate_config, test_prune_old_listings,
    ]
    for t in tests:
        t()
    print(f"\nAll {_passed} checks passed across {len(tests)} test groups.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
