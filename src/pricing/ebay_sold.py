"""
Fetches eBay completed/sold listings to estimate real market value and demand.
"""
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class SoldEntry:
    title: str
    price_usd: float
    sold_date: datetime
    condition: str
    url: str


@dataclass
class EbayMarketData:
    query: str
    sold_count: int = 0
    avg_price_usd: float = 0.0
    min_price_usd: float = 0.0
    max_price_usd: float = 0.0
    median_price_usd: float = 0.0
    # Demand: how many sold in the lookback window
    recent_sold_30d: int = 0
    recent_sold_7d: int = 0
    avg_days_to_sell: float = 0.0
    confidence: float = 0.0
    entries: list[SoldEntry] = field(default_factory=list)
    error: str = ""


def _parse_price(text: str) -> float:
    text = text.replace(",", "").strip()
    # Take the first price if it's a range
    text = text.split(" to ")[0]
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return float(digits)
    except ValueError:
        return 0.0


def _parse_sold_date(text: str) -> datetime:
    """Parse eBay 'Sold <date>' text."""
    text = text.lower().strip()
    now = datetime.utcnow()

    if "sold" in text:
        text = text.replace("sold", "").strip()

    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for abbr, num in months.items():
        if abbr in text:
            day_match = re.search(r"(\d+)", text)
            if day_match:
                day = int(day_match.group(1))
                year = now.year
                try:
                    dt = datetime(year, num, day)
                    if dt > now:
                        dt = datetime(year - 1, num, day)
                    return dt
                except ValueError:
                    pass

    return now - timedelta(days=30)


class EbaySoldPricer:
    def __init__(self, config: dict, usd_to_ron: float = 4.55, eur_to_ron: float = 4.97):
        self.tld = config.get("country_tld", "com")
        self.base_url = f"https://www.ebay.{self.tld}"
        self.lookback_days = config.get("ebay_sold_lookback_days", 90)
        self.usd_to_ron = usd_to_ron
        self.eur_to_ron = eur_to_ron

    def _sold_url(self, query: str, page: int = 1) -> str:
        encoded = query.replace(" ", "+")
        url = (
            f"{self.base_url}/sch/i.html"
            f"?_nkw={encoded}&LH_Sold=1&LH_Complete=1&_sop=13&_ipg=60"
        )
        if page > 1:
            url += f"&_pgn={page}"
        return url

    async def get_market_data(self, query: str, max_pages: int = 2) -> EbayMarketData:
        data = EbayMarketData(query=query)
        cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)
        all_entries: list[SoldEntry] = []

        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=25) as client:
            for page_num in range(1, max_pages + 1):
                url = self._sold_url(query, page_num)
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    entries = self._parse_sold_page(resp.text)
                    if not entries:
                        break
                    all_entries.extend(entries)
                    if page_num < max_pages:
                        await asyncio.sleep(1.2)
                except Exception as e:
                    data.error = str(e)
                    break

        # Filter to lookback window
        entries_in_window = [e for e in all_entries if e.sold_date >= cutoff]
        prices = [e.price_usd for e in entries_in_window if e.price_usd > 0]

        if not prices:
            data.confidence = 0.0
            return data

        prices.sort()
        # Remove top/bottom 5% outliers if enough data
        if len(prices) >= 10:
            trim = max(1, len(prices) // 20)
            prices = prices[trim:-trim]

        now = datetime.utcnow()
        data.sold_count = len(entries_in_window)
        data.avg_price_usd = sum(prices) / len(prices)
        data.min_price_usd = prices[0]
        data.max_price_usd = prices[-1]
        data.median_price_usd = prices[len(prices) // 2]
        data.recent_sold_30d = sum(1 for e in entries_in_window if (now - e.sold_date).days <= 30)
        data.recent_sold_7d = sum(1 for e in entries_in_window if (now - e.sold_date).days <= 7)
        data.entries = entries_in_window[:20]

        # Confidence based on sold count
        if data.sold_count >= 10:
            data.confidence = 0.90
        elif data.sold_count >= 5:
            data.confidence = 0.75
        elif data.sold_count >= 3:
            data.confidence = 0.60
        elif data.sold_count >= 1:
            data.confidence = 0.40
        else:
            data.confidence = 0.0

        return data

    def _parse_sold_page(self, html: str) -> list[SoldEntry]:
        soup = BeautifulSoup(html, "html.parser")
        entries = []

        for item in soup.select(".s-item__wrapper"):
            try:
                title_el = item.select_one(".s-item__title")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if "Shop on eBay" in title or not title:
                    continue

                price_el = item.select_one(".s-item__price")
                if not price_el:
                    continue
                price = _parse_price(price_el.get_text())
                if price <= 0:
                    continue

                date_el = item.select_one(".s-item__ended-date, .POSITIVE")
                sold_date = _parse_sold_date(date_el.get_text()) if date_el else datetime.utcnow()

                link_el = item.select_one("a.s-item__link")
                url = link_el["href"] if link_el else ""

                condition_el = item.select_one(".SECONDARY_INFO")
                condition = condition_el.get_text(strip=True) if condition_el else "unknown"

                entries.append(SoldEntry(
                    title=title,
                    price_usd=price,
                    sold_date=sold_date,
                    condition=condition,
                    url=url,
                ))
            except Exception:
                continue

        return entries

    def to_ron(self, usd_amount: float) -> float:
        return round(usd_amount * self.usd_to_ron, 2)

    def demand_score(self, data: EbayMarketData) -> float:
        """Score 0-100 representing how in-demand this item is."""
        score = 0.0
        # Recency weighted: more sold recently = higher demand
        if data.recent_sold_7d >= 10:
            score = 100.0
        elif data.recent_sold_7d >= 5:
            score = 85.0
        elif data.recent_sold_7d >= 2:
            score = 70.0
        elif data.recent_sold_30d >= 15:
            score = 65.0
        elif data.recent_sold_30d >= 8:
            score = 55.0
        elif data.recent_sold_30d >= 3:
            score = 40.0
        elif data.sold_count >= 1:
            score = 25.0
        else:
            score = 10.0
        return score
