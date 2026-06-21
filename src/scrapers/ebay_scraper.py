import re
import logging
import httpx
from datetime import datetime
from bs4 import BeautifulSoup
from .base import BaseScraper, RawListing

logger = logging.getLogger(__name__)


def _parse_ebay_price(text: str) -> tuple[float, str]:
    text = text.strip().replace(",", "")
    currency = "USD"
    if "GBP" in text or "£" in text:
        currency = "GBP"
    elif "EUR" in text or "€" in text:
        currency = "EUR"
    elif "RON" in text:
        currency = "RON"
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return float(digits), currency
    except ValueError:
        return 0.0, currency


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class EbayScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.tld = config.get("country_tld", "com")
        self.base_url = f"https://www.ebay.{self.tld}"
        self.max_pages = config.get("max_pages_per_keyword", 2)

    def _search_url(self, keyword: str, page: int = 1, sold_only: bool = False) -> str:
        encoded = keyword.replace(" ", "+")
        url = f"{self.base_url}/sch/i.html?_nkw={encoded}&_sop=10&_ipg=60"
        if sold_only:
            url += "&LH_Sold=1&LH_Complete=1"
        if page > 1:
            url += f"&_pgn={page}"
        return url

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        price_range = category_config.get("price_range", {})
        min_price = price_range.get("min_ron", 0)
        max_price = price_range.get("max_ron", 999999)
        listings: list[RawListing] = []

        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=20) as client:
            for page_num in range(1, self.max_pages + 1):
                url = self._search_url(keyword, page_num)
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    page_listings = self._parse_search_results(resp.text, keyword)
                    filtered = [l for l in page_listings if min_price <= l.price <= max_price]
                    listings.extend(filtered)
                    if page_num < self.max_pages:
                        await self._random_delay(1.5)
                except Exception as e:
                    logger.info(f"[eBay] Error page {page_num} for '{keyword}': {e}")
                    break

        return listings

    def _parse_search_results(self, html: str, keyword: str) -> list[RawListing]:
        soup = BeautifulSoup(html, "html.parser")
        listings = []

        for item in soup.select(".s-item__wrapper"):
            try:
                title_el = item.select_one(".s-item__title")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if "Shop on eBay" in title or not title:
                    continue

                link_el = item.select_one("a.s-item__link")
                href = link_el["href"] if link_el else ""
                id_match = re.search(r"/itm/(\d+)", href)
                external_id = id_match.group(1) if id_match else href[-12:]

                price_el = item.select_one(".s-item__price")
                if not price_el:
                    continue
                price_text = price_el.get_text(strip=True).split(" to ")[0]
                price, currency = _parse_ebay_price(price_text)
                if price <= 0:
                    continue

                condition_el = item.select_one(".SECONDARY_INFO, .s-item__condition")
                condition = condition_el.get_text(strip=True).lower() if condition_el else "unknown"

                location_el = item.select_one(".s-item__location")
                location = location_el.get_text(strip=True) if location_el else ""

                img_el = item.select_one(".s-item__image img")
                images = []
                if img_el:
                    src = img_el.get("src") or img_el.get("data-src") or ""
                    if src:
                        images.append(src)

                listings.append(RawListing(
                    external_id=external_id,
                    platform="ebay",
                    title=title,
                    price=price,
                    currency=currency,
                    url=href,
                    location=location,
                    condition=condition,
                    images=images,
                    matched_keyword=keyword,
                ))
            except Exception as e:
                logger.info(f"[eBay] Error parsing item: {e}")

        return listings

    async def get_listing_detail(self, url: str) -> dict:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=20) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                detail: dict = {}

                desc_el = soup.select_one("#desc_div, #viTabs_0_is")
                if desc_el:
                    detail["description"] = desc_el.get_text(strip=True)[:2000]

                seller_el = soup.select_one(".seller-persona .ux-textspans")
                if seller_el:
                    detail["seller_name"] = seller_el.get_text(strip=True)

                return detail
            except Exception as e:
                logger.info(f"[eBay] Error fetching detail {url}: {e}")
                return {}
