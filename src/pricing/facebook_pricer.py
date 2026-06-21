"""
Facebook Marketplace used-market pricer.

Reuses the FacebookMarketplaceScraper GraphQL search to fetch comparable
*used* listings for a query, then returns a normalized price distribution —
the same role OLX plays, but from a second used-market source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..scrapers.facebook import FacebookMarketplaceScraper
from .normalize import filter_comparable, price_stats, confidence_from_count


@dataclass
class FBEntry:
    title: str
    price_ron: float
    url: str = ""
    image: str = ""


@dataclass
class FBMarketData:
    query: str
    listing_count: int = 0
    min_price_ron: float = 0.0
    max_price_ron: float = 0.0
    avg_price_ron: float = 0.0
    median_price_ron: float = 0.0
    confidence: float = 0.0
    entries: list[FBEntry] = field(default_factory=list)
    error: str = ""


class FacebookPricer:
    """Stateless, cache-friendly used-market pricer backed by FB Marketplace."""

    def __init__(self, config: dict):
        # Reuse the scraper for its cookie/token/GraphQL machinery.
        self._scraper = FacebookMarketplaceScraper(config)
        self._cache: dict[str, FBMarketData] = {}
        self._available: Optional[bool] = None  # None=unknown, False=no cookies
        # FX rates so EUR/USD FB listings normalize to RON before comparison.
        self._fx = {
            "RON": 1.0,
            "EUR": float(config.get("eur_to_ron_rate", 4.97)),
            "USD": float(config.get("usd_to_ron_rate", 4.55)),
            "GBP": float(config.get("gbp_to_ron_rate", 5.75)),
        }

    async def close(self) -> None:
        await self._scraper.close()

    async def get_market_data(
        self, query: str, price_hint: float | None = None
    ) -> FBMarketData:
        if not query.strip():
            return FBMarketData(query=query, error="empty query")

        cache_key = f"{query}|{int(price_hint or 0)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Wide category config: no price filter here — we normalize locally so
        # the band is anchored to the source listing's asking price.
        cat_cfg = {"name": "", "price_range": {}}
        try:
            raw = await self._scraper.search(query, cat_cfg)
        except Exception as e:
            result = FBMarketData(query=query, error=str(e))
            self._cache[cache_key] = result
            return result

        if not raw:
            result = FBMarketData(query=query, error="no listings found")
            self._cache[cache_key] = result
            return result

        entries = [
            FBEntry(
                title=r.title,
                price_ron=round(r.price * self._fx.get((r.currency or "RON").upper(), 1.0), 2),
                url=r.url,
                image=(r.images[0] if r.images else ""),
            )
            for r in raw
            if r.price > 0
        ]

        # Normalize: accessories out, price-band around hint, IQR trim.
        entries = filter_comparable(
            entries, price_attr="price_ron", title_attr="title",
            price_hint=price_hint,
        )
        if not entries:
            result = FBMarketData(query=query, error="no comparable listings")
            self._cache[cache_key] = result
            return result

        prices = [e.price_ron for e in entries]
        st = price_stats(prices)
        n = st.count
        confidence = confidence_from_count(n)

        result = FBMarketData(
            query=query,
            listing_count=n,
            min_price_ron=st.min,
            max_price_ron=st.max,
            avg_price_ron=st.avg,
            median_price_ron=st.median,
            confidence=confidence,
            entries=entries,
        )
        self._cache[cache_key] = result
        return result
