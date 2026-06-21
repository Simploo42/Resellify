"""Live test of the HTTP-based OLX scraper + IQR outlier filtering."""
import asyncio
import json
from src.scrapers.olx import OLXScraper
from src.scrapers.filters import iqr_bounds

CONFIG = {
    "location": "bucuresti",
    "max_pages_per_keyword": 1,
    "request_delay_seconds": 2,
    "iqr_outlier_filter": True,
}
CATEGORY = {
    "name": "Smartphones",
    "keywords": ["iphone 13"],
    "price_range": {"min_ron": 300, "max_ron": 6000},
}


async def main():
    scraper = OLXScraper(CONFIG)
    try:
        # Scrape WITHOUT filtering first to show raw prices
        scraper.iqr_filter = False
        raw = await scraper.search("iphone 13", CATEGORY)
        prices = sorted(l.price for l in raw)
        lo, hi = iqr_bounds(prices)
        print(f"Raw listings: {len(raw)}")
        print(f"Price spread: {prices[0]:.0f} .. {prices[-1]:.0f} RON")
        print(f"IQR accept band: {lo:.0f} .. {hi:.0f} RON\n")

        kept = [l for l in raw if lo <= l.price <= hi]
        removed = [l for l in raw if not (lo <= l.price <= hi)]
        print(f"IQR kept: {len(kept)}  |  dropped as outliers: {len(removed)}")
        if removed:
            print("Dropped outliers:")
            for l in removed:
                print(f"   {l.price:>7.0f} RON  {l.title[:55]}")
        print("\n" + "=" * 66)
        print("Sample kept listings:")
        for i, l in enumerate(kept[:6], 1):
            print(f"[{i}] {l.price:>6.0f} {l.currency}  {l.title[:50]}")
            print(f"     {l.location}  | posted: {l.posted_at}")
        print("=" * 66)
        print("\nFirst listing as JSON:")
        if kept:
            f = kept[0]
            print(json.dumps({
                "id": f.listing_id, "title": f.title, "price": f.price,
                "currency": f.currency, "location": f.location,
                "condition": f.condition, "url": f.url[:75],
                "image": (f.images[0][:70] if f.images else None),
                "posted_at": str(f.posted_at),
            }, indent=2, ensure_ascii=False))
    finally:
        await scraper.close()


asyncio.run(main())
