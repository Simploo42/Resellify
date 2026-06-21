"""
OLX Romania scraper via HTTP (httpx + BeautifulSoup) — no browser needed.

This is the lightweight fallback used when the olx-mcp MCP server is
unavailable. OLX serves listing cards in server-side HTML, so we can parse
them directly. Uses the system CA store, so it works behind TLS-intercepting
proxies where headless Chromium would fail.

Approach (and the IQR outlier filter) adapted from kjanus03/olx-scrapper.
"""
import re
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .base import BaseScraper, RawListing
from .filters import filter_price_outliers

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

CONDITION_MAP = {
    "nou": "nou",
    "ca nou": "ca nou",
    "folosit": "folosit",
    "deteriorat": "deteriorat",
}

_DIACRITICS = str.maketrans({"ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t"})


def _slugify(text: str) -> str:
    text = text.lower().translate(_DIACRITICS)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)


def _parse_price(text: str) -> tuple[float, str]:
    text = text.strip().replace("\xa0", " ")
    currency = "RON"
    if "€" in text or "eur" in text.lower():
        currency = "EUR"
    elif "$" in text or "usd" in text.lower():
        currency = "USD"
    # "1 599 lei" -> 1599 ; "1.299,50 €" -> 1299.50
    cleaned = text.replace(".", "").replace(" ", "").replace(",", ".")
    digits = re.sub(r"[^\d.]", "", cleaned)
    try:
        return float(digits), currency
    except ValueError:
        return 0.0, currency


_RO_MONTHS = {
    "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "mai": 5,
    "iunie": 6, "iulie": 7, "august": 8, "septembrie": 9, "octombrie": 10,
    "noiembrie": 11, "decembrie": 12,
}


def _parse_ro_date(text: str) -> Optional[datetime]:
    t = text.strip().lower()
    now = datetime.utcnow()
    if "azi" in t:
        return now
    if "ieri" in t:
        return now - timedelta(days=1)
    m = re.search(r"(\d{1,2})\s+([a-zăâîșț]+)(?:\s+(\d{4}))?", t)
    if m:
        day = int(m.group(1))
        month = _RO_MONTHS.get(m.group(2))
        year = int(m.group(3)) if m.group(3) else now.year
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                return None
    return None


class OLXScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://www.olx.ro").rstrip("/")
        self.location = config.get("location", "")
        self.max_pages = config.get("max_pages_per_keyword", 3)
        self.iqr_filter = config.get("iqr_outlier_filter", True)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                headers=HEADERS, follow_redirects=True, timeout=20
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_search_url(self, keyword: str, page: int = 1) -> str:
        kw_slug = _slugify(keyword)
        loc = f"/{_slugify(self.location)}" if self.location else ""
        url = f"{self.base_url}{loc}/q-{kw_slug}/"
        params = ["search[order]=created_at:desc"]
        if page > 1:
            params.append(f"page={page}")
        return url + "?" + "&".join(params)

    def _build_search_url_priced(
        self, keyword: str, page: int, min_price: Optional[float], max_price: Optional[float]
    ) -> str:
        url = self._build_search_url(keyword, page)
        extra = []
        if min_price:
            extra.append(f"search[filter_float_price:from]={int(min_price)}")
        if max_price:
            extra.append(f"search[filter_float_price:to]={int(max_price)}")
        if extra:
            url += "&" + "&".join(extra)
        return url

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        client = await self._get_client()
        price_range = category_config.get("price_range", {})
        min_price = price_range.get("min_ron")
        max_price = price_range.get("max_ron")
        cat_name = category_config.get("name", "")

        listings: list[RawListing] = []
        for page_num in range(1, self.max_pages + 1):
            url = self._build_search_url_priced(keyword, page_num, min_price, max_price)
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as e:
                print(f"[OLX] Error fetching page {page_num} for '{keyword}': {e}")
                break

            page_listings = self._parse_page(resp.text, keyword, cat_name)
            # Hard price-range guard (OLX filter is best-effort)
            if min_price or max_price:
                lo = min_price or 0
                hi = max_price or float("inf")
                page_listings = [l for l in page_listings if lo <= l.price <= hi]

            if not page_listings:
                break
            listings.extend(page_listings)
            if page_num < self.max_pages:
                await self._random_delay()

        # IQR outlier removal across all pages for this keyword
        if self.iqr_filter and len(listings) >= 4:
            kept, removed = filter_price_outliers(listings, lambda l: l.price)
            if removed:
                print(f"[OLX] '{keyword}': dropped {len(removed)} price outlier(s)")
            listings = kept

        return listings

    def _parse_page(self, html: str, keyword: str, category: str) -> list[RawListing]:
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select('[data-cy="l-card"]')
        results: list[RawListing] = []
        for card in cards:
            parsed = self._parse_card(card, keyword, category)
            if parsed:
                results.append(parsed)
        return results

    def _parse_card(self, card, keyword: str, category: str) -> Optional[RawListing]:
        try:
            external_id = card.get("id") or ""
            link = card.select_one("a[href]")
            if not link:
                return None
            href = link.get("href", "")
            if href.startswith("/"):
                href = self.base_url + href
            if not external_id:
                m = re.search(r"ID([A-Za-z0-9]+)\.html", href)
                external_id = m.group(1) if m else href.rsplit("/", 1)[-1]
            if not external_id:
                return None

            title_el = card.select_one('[data-cy="ad-card-title"]') or card.select_one("h4, h6")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                return None

            price_el = card.select_one('[data-testid="ad-price"]')
            if not price_el:
                return None
            price, currency = _parse_price(price_el.get_text(strip=True))
            if price <= 0:
                return None

            location = ""
            posted_at = None
            loc_el = card.select_one('[data-testid="location-date"]')
            if loc_el:
                loc_text = loc_el.get_text(strip=True)
                parts = [p.strip() for p in loc_text.split(" - ")]
                location = parts[0] if parts else loc_text
                if len(parts) > 1:
                    posted_at = _parse_ro_date(parts[-1])

            images = []
            img = card.select_one("img")
            if img:
                src = img.get("src") or img.get("data-src") or ""
                if src and src.startswith("http"):
                    images.append(src)

            condition = "unknown"
            for badge in card.select("span"):
                bt = badge.get_text(strip=True).lower()
                if bt in CONDITION_MAP:
                    condition = CONDITION_MAP[bt]
                    break

            return RawListing(
                external_id=str(external_id),
                platform="olx",
                title=title,
                price=price,
                currency=currency,
                url=href,
                location=location,
                condition=condition,
                images=images,
                category=category,
                matched_keyword=keyword,
                posted_at=posted_at,
            )
        except Exception as e:
            print(f"[OLX] Error parsing card: {e}")
            return None

    async def get_listing_detail(self, url: str) -> dict:
        client = await self._get_client()
        detail: dict = {}
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            desc_el = soup.select_one('[data-cy="ad_description"], [data-testid="ad_description"]')
            if desc_el:
                detail["description"] = desc_el.get_text(strip=True)

            seller_el = soup.select_one('[data-testid="user-profile-user-name"]')
            if seller_el:
                detail["seller_name"] = seller_el.get_text(strip=True)

            cond_el = soup.find(string=re.compile(r"Stare", re.I))
            if cond_el:
                ct = cond_el.lower()
                for key in CONDITION_MAP:
                    if key in ct:
                        detail["condition"] = CONDITION_MAP[key]
                        break

            images = []
            for img in soup.select('[data-testid="image-gallery-item"] img, .swiper-slide img'):
                src = img.get("src") or ""
                if src.startswith("http") and src not in images:
                    images.append(src)
            if images:
                detail["images"] = images
        except Exception as e:
            print(f"[OLX] Error fetching detail {url}: {e}")
        return detail
