"""
Shared price-normalization helpers used by every pricer (eMAG, OLX, Facebook).

Goal: keep only *comparable* products in the price distribution so the median
reflects the real item value — not accessories (cases, chargers, cables, RAM
sticks, screen protectors, …) that drag the value down, nor wildly
mispriced outliers.
"""
from __future__ import annotations

import re

# Accessory / spare-part keywords (Romanian + English). If a product title is
# dominated by these and lacks the core product noun, it is almost certainly an
# accessory rather than the device itself.
_ACCESSORY_KEYWORDS = (
    # cases / covers / protection
    "husa", "huse", "carcasa", "carcase", "folie", "folii", "sticla", "tempered",
    "screen protector", "case", "cover", "bumper", "skin",
    # power / cables
    "incarcator", "incarcatoare", "cablu", "cabluri", "adaptor", "adaptoare",
    "charger", "cable", "usb-c", "type-c", "priza",
    # spare parts / components sold alone
    "baterie", "acumulator", "display ", "ecran ", "lcd", "touchscreen",
    "placa de baza", "motherboard", "tastatura", "keyboard ", "balama",
    "conector", "mufa", "flex", "banda", "ribbon",
    # memory / storage sold alone (common eMAG noise for laptop searches)
    "memorie", "memory ", "ram ddr", "ddr3", "ddr4", "ddr5", "sodimm", "so-dimm",
    "ssd ", "hdd ", "hard disk", "stick", "card memorie", "microsd", "sd card",
    # generic accessory nouns
    "suport", "stand", "geanta", "rucsac", "borseta", "mouse", "casti",
    "earbuds", "airpods case", "docking", "hub", "splitter", "set ",
    "kit ", "accesoriu", "accesorii", "accessory", "accessories", "pack",
    "abonament", "garantie extinsa", "asigurare",
)

# Strong product nouns — if present, the item is likely the real device even
# when an accessory word also appears (e.g. "laptop cu husa cadou").
_PRODUCT_NOUNS = (
    "laptop", "notebook", "macbook", "telefon", "smartphone", "iphone",
    "samsung galaxy", "tableta", "tablet", "ipad", "consola", "console",
    "playstation", "ps4", "ps5", "xbox", "nintendo", "monitor", "televizor",
    "tv ", "camera foto", "aparat foto", "obiectiv", "drona", "drone",
    "ceas", "watch", "smartwatch", "boxa", "boxe", "soundbar", "casca vr",
    "desktop", "calculator", "sistem pc", "all-in-one",
)

_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)")


def is_accessory(title: str) -> bool:
    """True when the title looks like an accessory/spare part, not the device."""
    if not title:
        return False
    t = title.lower()
    # Earliest position of any accessory keyword and any product noun.
    acc_pos = min((t.find(k) for k in _ACCESSORY_KEYWORDS if k in t), default=-1)
    if acc_pos < 0:
        return False
    prod_pos = min((t.find(n) for n in _PRODUCT_NOUNS if n in t), default=-1)
    # No product noun at all -> accessory.
    if prod_pos < 0:
        return True
    # Both present: it's the device only if the product noun leads (e.g.
    # "Laptop ... cu husa cadou"). If the accessory word comes first
    # ("Husa laptop ...", "Incarcator MacBook ...") it's an accessory.
    return acc_pos < prod_pos


def iqr_filter(prices: list[float]) -> list[float]:
    """Drop statistical outliers using the 1.5×IQR rule (needs >=4 points)."""
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


def price_band(price_hint: float | None,
               lo_mult: float = 0.30,
               hi_mult: float = 3.0) -> tuple[float | None, float | None]:
    """
    Build a [min, max] band around the listing's asking price so cheap
    accessories and absurdly priced bundles fall outside the comparable set.
    """
    if not price_hint or price_hint <= 0:
        return None, None
    return max(1.0, price_hint * lo_mult), price_hint * hi_mult


def filter_comparable(
    items: list,
    price_attr: str = "price_ron",
    title_attr: str = "title",
    price_hint: float | None = None,
    lo_mult: float = 0.30,
    hi_mult: float = 3.0,
) -> list:
    """
    Return the subset of `items` that are genuine comparables:
      1. not classified as accessories,
      2. within the price band around `price_hint` (when provided),
      3. survivors of an IQR outlier trim on the remaining prices.

    Falls back gracefully (never returns empty if any items exist) so a noisy
    page still yields an estimate rather than nothing.
    """
    if not items:
        return []

    # 1. accessory classification
    non_acc = [it for it in items if not is_accessory(getattr(it, title_attr, "") or "")]
    pool = non_acc or items  # don't wipe everything if classifier is too aggressive

    # 2. price band
    lo, hi = price_band(price_hint, lo_mult, hi_mult)
    if lo is not None:
        banded = [it for it in pool if lo <= getattr(it, price_attr, 0.0) <= hi]
        pool = banded or pool

    # 3. IQR trim
    prices = [getattr(it, price_attr, 0.0) for it in pool]
    keep = set(iqr_filter(prices))
    trimmed = [it for it in pool if getattr(it, price_attr, 0.0) in keep]
    return trimmed or pool
