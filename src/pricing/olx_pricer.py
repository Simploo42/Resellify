"""
OLX Romania used-market pricer.

Searches OLX for a query and returns the price distribution of active listings.
This gives the real "what similar used items sell for" — which is the correct
resell value baseline, not the eMAG new-item price.

Uses the same HTTP approach as src/scrapers/olx.py but is stateless and
cache-friendly (no per-category scraping loop).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .normalize import is_accessory

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10; SM-G981B) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Mobile Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.olx.ro/",
}

_DIACRITICS = str.maketrans({"ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t"})


def _slugify(text: str) -> str:
    text = text.lower().translate(_DIACRITICS)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    return re.sub(r"-+", "-", text)


def _iqr_filter(prices: list[float]) -> list[float]:
    if len(prices) < 4:
        return prices
    s = sorted(prices)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [p for p in s if lo <= p <= hi]


@dataclass
class OLXEntry:
    title: str
    price_ron: float
    url: str = ""
    image: str = ""


@dataclass
class OLXMarketData:
    query: str
    listing_count: int = 0
    min_price_ron: float = 0.0
    max_price_ron: float = 0.0
    avg_price_ron: float = 0.0
    median_price_ron: float = 0.0
    confidence: float = 0.0
    entries: list[OLXEntry] = field(default_factory=list)
    error: str = ""


class OLXPricer:
    """
    Estimates the used-market resell value of an item by searching
    OLX Romania and returning price statistics of active listings.

    Results are cached per query to avoid redundant requests during a scan.
    """

    BASE_URL = "https://www.olx.ro"

    def __init__(self, timeout: float = 15.0, pages: int = 2):
        self._timeout = timeout
        self._pages = pages
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: dict[str, OLXMarketData] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                headers=HEADERS,
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _search_url(self, query: str, page: int,
                    min_price: Optional[float] = None,
                    max_price: Optional[float] = None) -> str:
        slug = _slugify(query)
        url = f"{self.BASE_URL}/q-{slug}/"
        params = ["search[order]=created_at:desc"]
        if page > 1:
            params.append(f"page={page}")
        if min_price:
            params.append(f"search[filter_float_price:from]={int(min_price)}")
        if max_price:
            params.append(f"search[filter_float_price:to]={int(max_price)}")
        return url + "?" + "&".join(params)

    def _parse_listings(self, html: str) -> list[OLXEntry]:
        soup = BeautifulSoup(html, "html.parser")
        entries: list[OLXEntry] = []
        for card in soup.select('[data-cy="l-card"]'):
            # Title
            title_el = (
                card.select_one('[data-testid="ad_card-title"]')
                or card.select_one("h6")
                or card.select_one("h4")
                or card.select_one("h3")
            )
            title = title_el.get_text(strip=True) if title_el else ""

            # URL
            link_el = card.select_one('a[href*="/d/"]') or card.select_one("a[href]")
            href = link_el.get("href", "") if link_el else ""
            url = ("https://www.olx.ro" + href) if href.startswith("/") else href

            # Image (lazy-loaded src or data-src)
            img_el = card.select_one("img")
            image = ""
            if img_el:
                image = (img_el.get("src") or img_el.get("data-src")
                         or img_el.get("srcset", "").split(" ")[0] or "")

            # Price
            price_el = (
                card.select_one('[data-testid="ad-price"]')
                or card.select_one("p.price")
                or card.select_one('[class*="price"]')
            )
            if not price_el:
                continue
            text = price_el.get_text(" ", strip=True).lower()
            if "negociabil" in text and not any(c.isdigit() for c in text):
                continue
            clean = re.sub(r"\s+", "", text).replace(".", "").replace(",", ".")
            m = re.search(r"(\d+(?:\.\d+)?)", clean)
            if not m:
                continue
            try:
                val = float(m.group(1))
                if val > 10:
                    entries.append(OLXEntry(title=title, price_ron=val, url=url, image=image))
            except ValueError:
                pass
        return entries

    async def get_market_data(
        self, query: str, price_hint: Optional[float] = None
    ) -> OLXMarketData:
        """
        Fetch OLX used-market price stats for `query`.

        `price_hint` (the listing's asking price in RON) is used to build a
        price-range filter so accessories/unrelated items are excluded.
        We search within [hint × 0.30, hint × 3.0] — wide enough to catch
        all similar items while rejecting cheap accessories.
        """
        if not query.strip():
            return OLXMarketData(query=query, error="empty query")

        cache_key = f"{query}|{int(price_hint or 0)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        min_price: Optional[float] = None
        max_price: Optional[float] = None
        if price_hint and price_hint > 0:
            min_price = max(1.0, price_hint * 0.30)
            max_price = price_hint * 3.0

        client = await self._get_client()
        all_entries: list[OLXEntry] = []

        for page in range(1, self._pages + 1):
            url = self._search_url(query, page, min_price, max_price)
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as e:
                if page == 1:
                    result = OLXMarketData(query=query, error=str(e))
                    self._cache[cache_key] = result
                    return result
                break

            page_entries = self._parse_listings(resp.text)
            if not page_entries:
                break
            all_entries.extend(page_entries)

        if not all_entries:
            result = OLXMarketData(query=query, error="no listings found")
            self._cache[cache_key] = result
            return result

        all_prices = [e.price_ron for e in all_entries]
        # Drop accessories/spare-parts first so the median reflects the device.
        deviced = [e for e in all_entries if not is_accessory(e.title)]
        all_entries = deviced or all_entries

        all_prices = [e.price_ron for e in all_entries]
        filtered_prices = _iqr_filter(all_prices)
        if not filtered_prices:
            filtered_prices = all_prices

        # Entries that survived IQR filter (keep only those whose price is in the filtered set)
        price_set = set(filtered_prices)
        filtered_entries = [e for e in all_entries if e.price_ron in price_set]

        n = len(filtered_prices)
        med = statistics.median(filtered_prices)
        avg = sum(filtered_prices) / n
        confidence = min(1.0, 0.25 + n * 0.05)

        result = OLXMarketData(
            query=query,
            listing_count=n,
            min_price_ron=round(min(filtered_prices), 2),
            max_price_ron=round(max(filtered_prices), 2),
            avg_price_ron=round(avg, 2),
            median_price_ron=round(med, 2),
            confidence=round(confidence, 3),
            entries=filtered_entries[:20],
        )
        self._cache[cache_key] = result
        return result
