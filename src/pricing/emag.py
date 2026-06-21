"""
eMAG Romania price scraper — fetches current market prices in RON directly,
no currency conversion needed. Used as the primary Romanian market reference.

eMAG is the largest Romanian e-commerce platform, so its prices reflect
what buyers actually pay locally — more relevant than eBay USD for OLX arbitrage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from .normalize import filter_comparable

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.emag.ro/",
}


@dataclass
class EmagListing:
    title: str
    price_ron: float
    url: str
    is_new: bool = True
    image: str = ""


@dataclass
class EmagMarketData:
    query: str
    listing_count: int = 0
    min_price_ron: float = 0.0
    max_price_ron: float = 0.0
    avg_price_ron: float = 0.0
    median_price_ron: float = 0.0
    confidence: float = 0.0
    listings: list[EmagListing] = field(default_factory=list)
    error: str = ""


def _parse_ron(text: str) -> float:
    """Extract a RON price from text like '1.299,99 Lei' or '1299.99'."""
    # Remove thousands separator (. in Romanian format) and fix decimal (,)
    cleaned = text.replace(".", "").replace(",", ".").strip()
    digits = re.sub(r"[^\d.]", "", cleaned)
    # Take the first valid float
    m = re.match(r"(\d+(?:\.\d+)?)", digits)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def _extract_prices(soup: BeautifulSoup) -> list[EmagListing]:
    listings: list[EmagListing] = []

    # eMAG product card selectors (multiple tried in order for robustness)
    cards = (
        soup.select(".card-item")
        or soup.select(".js-product-data")
        or soup.select("[data-zone-name='offerCardV2']")
        or soup.select(".product-card")
    )

    for card in cards:
        # Title
        title_el = (
            card.select_one(".card-v2-title")
            or card.select_one("[data-zone-name='title'] a")
            or card.select_one(".product-title a")
            or card.select_one("h2 a")
            or card.select_one("h3 a")
        )
        title = title_el.get_text(strip=True) if title_el else ""
        # Prefer the link's title attribute / image alt when text is a generic CTA
        if not title or title.lower() in ("vezi detalii", "detalii", ""):
            link = card.select_one("a[title]")
            img = card.select_one("img[alt]")
            title = (
                (link.get("title", "").strip() if link else "")
                or (img.get("alt", "").strip() if img else "")
                or title
            )

        # Price — try multiple selectors
        price_el = (
            card.select_one(".product-new-price")
            or card.select_one(".card-v2-price .product-new-price")
            or card.select_one("[data-zone-name='price'] .product-new-price")
            or card.select_one(".price-wrapper .price")
            or card.select_one(".price")
        )
        price_text = price_el.get_text(strip=True) if price_el else ""
        price = _parse_ron(price_text)

        if price <= 0:
            continue

        url_el = title_el or card.select_one("a[href]")
        url = url_el.get("href", "") if url_el else ""
        if url and not url.startswith("http"):
            url = "https://www.emag.ro" + url

        # Image
        img_el = card.select_one("img")
        image = ""
        if img_el:
            image = (img_el.get("src") or img_el.get("data-src")
                     or img_el.get("data-original")
                     or img_el.get("srcset", "").split(" ")[0] or "")

        listings.append(EmagListing(title=title, price_ron=price, url=url, image=image))

    return listings


def _stats(prices: list[float]) -> tuple[float, float, float, float]:
    """Returns (min, max, avg, median)."""
    if not prices:
        return 0.0, 0.0, 0.0, 0.0
    s = sorted(prices)
    n = len(s)
    median = (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
    return s[0], s[-1], sum(s) / n, median


class EmagPricer:
    def __init__(self, timeout: float = 12.0):
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                headers=HEADERS,
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_market_data(self, query: str, price_hint: float | None = None) -> EmagMarketData:
        """
        Search eMAG Romania for `query` and return current price stats in RON.
        Uses the first page of results (typically 36 products).

        `price_hint` (the source listing's asking price) is used to drop
        accessories and out-of-band products so the median reflects the
        real device, not cases/chargers/RAM that pull the value down.
        """
        if not query.strip():
            return EmagMarketData(query=query, error="empty query")

        url = f"https://www.emag.ro/search/{query.replace(' ', '+')}"
        try:
            client = await self._get_client()
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as e:
            return EmagMarketData(query=query, error=str(e))

        soup = BeautifulSoup(resp.text, "html.parser")
        listings = _extract_prices(soup)

        if not listings:
            return EmagMarketData(query=query, error="no listings found")

        # Normalize: strip accessories/spare-parts + price-band + IQR outliers
        # so the new-retail ceiling reflects the actual product.
        listings = filter_comparable(
            listings, price_attr="price_ron", title_attr="title",
            price_hint=price_hint,
        )

        prices = [l.price_ron for l in listings]
        mn, mx, avg, med = _stats(prices)
        n = len(prices)
        confidence = min(1.0, 0.3 + n * 0.07)  # 0.3 base, +0.07 per listing, cap 1.0

        return EmagMarketData(
            query=query,
            listing_count=n,
            min_price_ron=round(mn, 2),
            max_price_ron=round(mx, 2),
            avg_price_ron=round(avg, 2),
            median_price_ron=round(med, 2),
            confidence=round(confidence, 3),
            listings=listings,
        )
