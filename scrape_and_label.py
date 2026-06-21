"""
Scrape ~5k OLX Romania titles, dedup them, run through the NER labeling pipeline.

Usage:
  python scrape_and_label.py           # scrape fresh + label
  python scrape_and_label.py --label-only  # re-run pipeline on existing titles.jsonl
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from src.scrapers.olx import OLXScraper
from src.title_engine.pipeline import TitleEnginePipeline

TITLES_FILE = Path("data/title_engine/raw_titles.jsonl")

# Wide keyword set across Romanian resell categories — no location filter
# so we pull all Romania, maximising unique listings.
KEYWORDS = [
    # Phones
    "iphone", "samsung galaxy", "xiaomi", "huawei", "oneplus",
    "pixel", "motorola", "redmi", "oppo", "nokia",
    # Laptops
    "laptop", "macbook", "thinkpad", "dell xps", "asus rog",
    "hp laptop", "acer laptop", "msi laptop", "lenovo",
    # Gaming
    "ps5", "playstation 5", "xbox series", "nintendo switch",
    "steam deck", "controller", "gpu rtx", "gtx 1080", "rx 6800",
    # Audio
    "airpods", "sony wh", "bose headphones", "jbl speaker",
    "casti bluetooth", "boxe", "amplificator",
    # Cameras
    "camera foto", "nikon", "canon eos", "sony alpha",
    "fujifilm", "obiectiv foto", "gopro",
    # Tablets / wearables
    "ipad", "tablet", "smartwatch", "apple watch", "samsung watch",
    # TV / display
    "televizor", "monitor gaming", "oled tv", "qled",
    # Components
    "placa video", "procesor", "memorie ram", "ssd nvme", "motherboard",
    # General electronics
    "telefon", "electronice", "consola jocuri",
]

SCRAPER_CONFIG = {
    "location": "",           # all Romania
    "max_pages_per_keyword": 2,
    "request_delay_seconds": 1.5,
    "iqr_outlier_filter": False,  # keep raw; pipeline deduper handles it
}


async def scrape_titles(target: int = 5000) -> list[dict]:
    TITLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    scraper = OLXScraper(SCRAPER_CONFIG)
    seen_ids: set[str] = set()
    titles: list[dict] = []

    try:
        for i, kw in enumerate(KEYWORDS):
            if len(titles) >= target:
                break
            cat = {"name": "General", "price_range": {}}
            try:
                listings = await scraper.search(kw, cat)
            except Exception as e:
                print(f"  [scrape] error on '{kw}': {e}")
                continue

            new = 0
            for l in listings:
                if l.listing_id not in seen_ids:
                    seen_ids.add(l.listing_id)
                    titles.append({"id": l.listing_id, "raw": l.title})
                    new += 1

            print(f"  [{i+1:02d}/{len(KEYWORDS)}] '{kw}': +{new} new | total {len(titles)}")

            if len(titles) >= target:
                break
    finally:
        await scraper.close()

    # Persist raw titles
    with open(TITLES_FILE, "w") as f:
        for t in titles:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n[scrape] Wrote {len(titles)} raw titles → {TITLES_FILE}")
    return titles


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-only", action="store_true",
                        help="Skip scraping; re-use existing raw_titles.jsonl")
    parser.add_argument("--target", type=int, default=5000,
                        help="Target number of unique titles to scrape")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.label_only and TITLES_FILE.exists():
        print(f"[main] Loading existing titles from {TITLES_FILE}")
        titles = [json.loads(l) for l in TITLES_FILE.read_text().splitlines() if l.strip()]
    else:
        print(f"[main] Scraping up to {args.target} titles from OLX Romania…")
        t0 = time.time()
        titles = await scrape_titles(args.target)
        print(f"[main] Scrape done in {time.time()-t0:.1f}s")

    print(f"\n[main] Running NER pipeline on {len(titles)} titles…")
    pipeline = TitleEnginePipeline(batch_size=args.batch_size)
    stats = await pipeline.run(titles)

    print("\n── Summary ──────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print(f"\nOutputs:")
    print(f"  data/title_engine/train_v0.jsonl")
    print(f"  data/title_engine/spot_check_queue.jsonl")

asyncio.run(main())
