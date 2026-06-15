import json
import os
import re
from datetime import datetime
from typing import Optional
from playwright.async_api import async_playwright, Browser
from .base import BaseScraper, RawListing


def _parse_price(text: str) -> tuple[float, str]:
    text = text.strip().replace(",", "").replace(".", "")
    currency = "RON"
    if "€" in text or "EUR" in text:
        currency = "EUR"
    elif "$" in text:
        currency = "USD"
    digits = re.sub(r"[^\d]", "", text)
    try:
        return float(digits), currency
    except ValueError:
        return 0.0, currency


class FacebookMarketplaceScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.location = config.get("location", "Bucharest, Romania")
        self.radius_km = config.get("radius_km", 50)
        self.max_results = config.get("max_results_per_keyword", 40)
        self.cookies_file = os.environ.get("FACEBOOK_COOKIES_FILE", "./facebook_cookies.json")
        self._browser: Optional[Browser] = None
        self._playwright = None

    async def _get_browser(self) -> Browser:
        if not self._browser:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            )
        return self._browser

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    def _load_cookies(self) -> list[dict]:
        if not os.path.exists(self.cookies_file):
            print(f"[Facebook] Cookie file not found: {self.cookies_file}")
            print("[Facebook] Export cookies using the 'Cookie Editor' browser extension.")
            return []
        with open(self.cookies_file) as f:
            cookies = json.load(f)
        # Normalize to Playwright cookie format
        normalized = []
        for c in cookies:
            normalized.append({
                "name": c.get("name", ""),
                "value": c.get("value", ""),
                "domain": c.get("domain", ".facebook.com"),
                "path": c.get("path", "/"),
                "httpOnly": c.get("httpOnly", False),
                "secure": c.get("secure", True),
                "sameSite": c.get("sameSite", "None"),
            })
        return normalized

    def _build_search_url(self, keyword: str, price_range: dict) -> str:
        # Facebook Marketplace uses a city-based URL; fall back to generic search
        encoded_kw = keyword.replace(" ", "%20")
        min_price = price_range.get("min_ron", "")
        max_price = price_range.get("max_ron", "")
        # Radius in miles (FB uses miles) — 1 km ≈ 0.621 miles
        radius_miles = int(self.radius_km * 0.621)
        url = (
            f"https://www.facebook.com/marketplace/search"
            f"?query={encoded_kw}"
            f"&radius={radius_miles}"
        )
        if min_price:
            url += f"&minPrice={int(min_price)}"
        if max_price:
            url += f"&maxPrice={int(max_price)}"
        return url

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        cookies = self._load_cookies()
        if not cookies:
            return []

        price_range = category_config.get("price_range", {})
        browser = await self._get_browser()
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        await context.add_cookies(cookies)

        page = await context.new_page()
        listings: list[RawListing] = []

        try:
            url = self._build_search_url(keyword, price_range)
            await page.goto(url, wait_until="networkidle", timeout=45000)
            await page.wait_for_timeout(3000)

            # Scroll to load more listings
            for _ in range(3):
                await page.keyboard.press("End")
                await page.wait_for_timeout(1500)

            # Parse listing cards
            cards = await page.query_selector_all('[data-testid="marketplace_feed_item"]')
            if not cards:
                # Alternative selector
                cards = await page.query_selector_all('div[role="main"] a[href*="/marketplace/item/"]')

            seen_ids: set[str] = set()
            for card in cards[:self.max_results]:
                listing = await self._parse_card(card, keyword)
                if listing and listing.external_id not in seen_ids:
                    min_p = price_range.get("min_ron", 0)
                    max_p = price_range.get("max_ron", 999999)
                    if min_p <= listing.price <= max_p:
                        listings.append(listing)
                        seen_ids.add(listing.external_id)

        except Exception as e:
            print(f"[Facebook] Error searching '{keyword}': {e}")
        finally:
            await context.close()

        return listings

    async def _parse_card(self, card, keyword: str) -> Optional[RawListing]:
        try:
            href = ""
            try:
                href = await card.get_attribute("href") or ""
                if not href:
                    link_el = await card.query_selector("a[href*='/marketplace/item/']")
                    if link_el:
                        href = await link_el.get_attribute("href") or ""
            except Exception:
                pass

            if not href:
                return None
            if not href.startswith("http"):
                href = "https://www.facebook.com" + href

            id_match = re.search(r"/item/(\d+)", href)
            external_id = id_match.group(1) if id_match else href.split("/")[-1]

            title = ""
            title_el = await card.query_selector("span[class*='x1lliihq']:not([class*='x6s0dn4'])")
            if title_el:
                title = (await title_el.inner_text()).strip()
            if not title:
                all_spans = await card.query_selector_all("span")
                texts = [await s.inner_text() for s in all_spans]
                title = next((t.strip() for t in texts if len(t.strip()) > 5), "")

            price_text = ""
            price_el = await card.query_selector("span[class*='x193iq5w']")
            if price_el:
                price_text = (await price_el.inner_text()).strip()
            price, currency = _parse_price(price_text)

            location = ""
            location_el = await card.query_selector("span[class*='x1nxh6w3']")
            if location_el:
                location = (await location_el.inner_text()).strip()

            images = []
            img_el = await card.query_selector("img[src*='scontent']")
            if img_el:
                src = await img_el.get_attribute("src") or ""
                if src:
                    images.append(src)

            if not title or price <= 0:
                return None

            return RawListing(
                external_id=external_id,
                platform="facebook",
                title=title,
                price=price,
                currency=currency,
                url=href,
                location=location,
                images=images,
                matched_keyword=keyword,
            )
        except Exception as e:
            print(f"[Facebook] Error parsing card: {e}")
            return None

    async def get_listing_detail(self, url: str) -> dict:
        cookies = self._load_cookies()
        if not cookies:
            return {}

        browser = await self._get_browser()
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        await context.add_cookies(cookies)
        page = await context.new_page()
        detail: dict = {}

        try:
            await page.goto(url, wait_until="networkidle", timeout=45000)
            await page.wait_for_timeout(2000)

            desc_el = await page.query_selector('[data-testid="marketplace_listing_page_description"]')
            if desc_el:
                detail["description"] = (await desc_el.inner_text()).strip()

            seller_el = await page.query_selector('a[href*="/profile/"] span')
            if seller_el:
                detail["seller_name"] = (await seller_el.inner_text()).strip()

            img_els = await page.query_selector_all('img[src*="scontent"]')
            images = []
            for img in img_els:
                src = await img.get_attribute("src") or ""
                if src and src not in images:
                    images.append(src)
            if images:
                detail["images"] = images

        except Exception as e:
            print(f"[Facebook] Error fetching detail {url}: {e}")
        finally:
            await context.close()

        return detail
