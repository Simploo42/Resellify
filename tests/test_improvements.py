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


def main():
    tests = [
        test_price_stats, test_confidence_from_count, test_iqr_filter,
        test_is_accessory, test_demand_proxy, test_liquidity_gate,
        test_grade_dedup, test_concurrent_scrape, test_fb_currency_normalization,
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
