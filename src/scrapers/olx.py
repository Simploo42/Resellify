import re
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from playwright.async_api import async_playwright, Page, Browser
from .base import BaseScraper, RawListing

# OLX Romania city slug mapping
CITY_SLUGS = {
    "bucuresti": "bucuresti",
    "cluj-napoca": "cluj-napoca",
    "timisoara": "timisoara",
    "iasi": "iasi",
    "constanta": "constanta",
    "brasov": "brasov",
    "": "",  # all Romania
}

CONDITION_MAP = {
    "nou": "nou",
    "ca nou": "ca nou",
    "folosit": "folosit",
    "deteriorat": "deteriorat",
}


def _parse_price(text: str) -> tuple[float, str]:
    text = text.strip().replace("\xa0", " ").replace(",", ".")
    currency = "RON"
    if "€" in text or "EUR" in text:
        currency = "EUR"
    elif "$" in text or "USD" in text:
        currency = "USD"
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return float(digits), currency
    except ValueError:
        return 0.0, currency


def _parse_olx_date(text: str) -> Optional[datetime]:
    text = text.strip().lower()
    now = datetime.utcnow()
    if "azi" in text or "today" in text:
        return now
    if "ieri" in text or "yesterday" in text:
        return now - timedelta(days=1)
    month_map = {
        "ian": 1, "feb": 2, "mar": 3, "apr": 4, "mai": 5, "iun": 6,
        "iul": 7, "aug": 8, "sep": 9, "oct": 10, "noi": 11, "dec": 12,
    }
    for abbr, num in month_map.items():
        if abbr in text:
            day_match = re.search(r"\d+", text)
            if day_match:
                try:
                    return datetime(now.year, num, int(day_match.group()))
                except ValueError:
                    pass
    return None


class OLXScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://www.olx.ro")
        self.location = config.get("location", "")
        self.max_pages = config.get("max_pages_per_keyword", 3)
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

    def _build_search_url(self, keyword: str, category_path: str = "", page: int = 1) -> str:
        kw_slug = keyword.strip().replace(" ", "-")
        location_part = f"/{self.location}" if self.location else ""
        if category_path:
            base = f"{self.base_url}{category_path}"
            url = f"{base}q-{kw_slug}/"
        else:
            url = f"{self.base_url}/oferte{location_part}/q-{kw_slug}/"
        if page > 1:
            url += f"?page={page}"
        return url

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        browser = await self._get_browser()
        listings: list[RawListing] = []
        price_range = category_config.get("price_range", {})
        min_price = price_range.get("min_ron", 0)
        max_price = price_range.get("max_ron", 999999)
        cat_path = category_config.get("olx_category_path", "")

        for page_num in range(1, self.max_pages + 1):
            url = self._build_search_url(keyword, cat_path, page_num)
            page_listings = await self._scrape_page(browser, url, keyword, min_price, max_price)
            if not page_listings:
                break
            listings.extend(page_listings)
            if page_num < self.max_pages:
                await self._random_delay()

        return listings

    async def _scrape_page(
        self, browser: Browser, url: str, keyword: str, min_price: float, max_price: float
    ) -> list[RawListing]:
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="ro-RO",
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()
        listings: list[RawListing] = []

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)

            # Accept cookie consent if present
            try:
                await page.click('[id*="onetrust-accept"]', timeout=3000)
                await page.wait_for_timeout(500)
            except Exception:
                pass

            cards = await page.query_selector_all('[data-cy="l-card"]')
            for card in cards:
                listing = await self._parse_card(card, keyword)
                if listing and min_price <= listing.price <= max_price:
                    listings.append(listing)

        except Exception as e:
            print(f"[OLX] Error scraping {url}: {e}")
        finally:
            await context.close()

        return listings

    async def _parse_card(self, card, keyword: str) -> Optional[RawListing]:
        try:
            # Link and ID
            link_el = await card.query_selector("a")
            if not link_el:
                return None
            href = await link_el.get_attribute("href") or ""
            if not href.startswith("http"):
                href = self.base_url + href
            external_id = href.rstrip("/").split("-")[-1].split(".")[0]

            # Title
            title_el = await card.query_selector("h6, h4, [data-cy='ad-title']")
            title = (await title_el.inner_text()).strip() if title_el else ""
            if not title:
                return None

            # Price
            price_el = await card.query_selector("[data-testid='ad-price'], p.css-10b0gli, [data-cy='ad-price']")
            price_text = (await price_el.inner_text()).strip() if price_el else "0"
            price, currency = _parse_price(price_text)
            if price <= 0:
                return None

            # Location and date
            location = ""
            posted_at = None
            footer_el = await card.query_selector("[data-testid='location-date'], p.css-veheph")
            if footer_el:
                footer_text = await footer_el.inner_text()
                parts = [p.strip() for p in footer_text.split("-")]
                if len(parts) >= 2:
                    location = parts[0]
                    posted_at = _parse_olx_date(parts[-1])
                elif len(parts) == 1:
                    location = parts[0]

            # Image
            images = []
            img_el = await card.query_selector("img")
            if img_el:
                src = await img_el.get_attribute("src") or await img_el.get_attribute("data-src") or ""
                if src:
                    images.append(src)

            # Condition badge (sometimes present on cards)
            condition = "unknown"
            badge_el = await card.query_selector("[data-testid='ad-badge-label'], span.css-1dbe8x")
            if badge_el:
                badge_text = (await badge_el.inner_text()).strip().lower()
                for key in CONDITION_MAP:
                    if key in badge_text:
                        condition = CONDITION_MAP[key]
                        break

            return RawListing(
                external_id=external_id,
                platform="olx",
                title=title,
                price=price,
                currency=currency,
                url=href,
                location=location,
                condition=condition,
                images=images,
                matched_keyword=keyword,
                posted_at=posted_at,
            )
        except Exception as e:
            print(f"[OLX] Error parsing card: {e}")
            return None

    async def get_listing_detail(self, url: str) -> dict:
        browser = await self._get_browser()
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="ro-RO",
        )
        page = await context.new_page()
        detail: dict = {}
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)

            desc_el = await page.query_selector("[data-cy='ad_description']")
            if desc_el:
                detail["description"] = (await desc_el.inner_text()).strip()

            seller_el = await page.query_selector("[data-cy='seller-card'] h4")
            if seller_el:
                detail["seller_name"] = (await seller_el.inner_text()).strip()

            condition_el = await page.query_selector("[data-testid='key-value-Stan']")
            if condition_el:
                cond_text = (await condition_el.inner_text()).strip().lower()
                for key in CONDITION_MAP:
                    if key in cond_text:
                        detail["condition"] = CONDITION_MAP[key]
                        break

            # All images
            img_els = await page.query_selector_all("[data-testid='image-gallery-img'] img, .swiper-slide img")
            images = []
            for img in img_els:
                src = await img.get_attribute("src") or ""
                if src and src not in images:
                    images.append(src)
            if images:
                detail["images"] = images

        except Exception as e:
            print(f"[OLX] Error fetching detail {url}: {e}")
        finally:
            await context.close()

        return detail
