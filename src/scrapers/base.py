from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import asyncio
import random

logger = logging.getLogger(__name__)


@dataclass
class RawListing:
    external_id: str
    platform: str
    title: str
    price: float
    currency: str
    url: str
    location: str = ""
    description: str = ""
    condition: str = "unknown"
    images: list[str] = field(default_factory=list)
    seller_name: str = ""
    seller_url: str = ""
    category: str = ""
    matched_keyword: str = ""
    posted_at: Optional[datetime] = None

    @property
    def listing_id(self) -> str:
        return f"{self.platform}:{self.external_id}"


class BaseScraper(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.request_delay = config.get("request_delay_seconds", 2.5)

    async def _random_delay(self, base: float | None = None):
        delay = base or self.request_delay
        jitter = delay * 0.4
        await asyncio.sleep(delay + random.uniform(-jitter, jitter))

    @abstractmethod
    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        """Search for listings matching a keyword within a category config."""
        ...

    @abstractmethod
    async def get_listing_detail(self, url: str) -> dict:
        """Fetch additional detail for a single listing URL."""
        ...

    async def scrape_all_keywords(self, categories: list[dict]) -> list[RawListing]:
        all_listings: list[RawListing] = []
        for cat in categories:
            if not cat.get("enabled", True):
                continue
            for keyword in cat.get("keywords", []):
                try:
                    results = await self.search(keyword, cat)
                    for r in results:
                        r.matched_keyword = keyword
                        r.category = cat.get("name", "")
                    all_listings.extend(results)
                    await self._random_delay()
                except Exception as e:
                    logger.info(f"[{self.__class__.__name__}] Error searching '{keyword}': {e}")
        return all_listings
