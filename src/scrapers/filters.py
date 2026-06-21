"""
Shared listing filters: IQR price-outlier removal and relevance helpers.
Adapted from the heuristics in kjanus03/olx-scrapper.
"""
from __future__ import annotations
from typing import TypeVar, Callable, Sequence

T = TypeVar("T")


def iqr_bounds(values: Sequence[float], k: float = 1.5) -> tuple[float, float]:
    """
    Return (lower, upper) acceptable bounds using the interquartile range.
    Values outside [Q1 - k*IQR, Q3 + k*IQR] are considered outliers.
    """
    if len(values) < 4:
        # Not enough data for a meaningful IQR — accept everything.
        return float("-inf"), float("inf")

    s = sorted(values)
    n = len(s)

    def _quantile(q: float) -> float:
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    q1 = _quantile(0.25)
    q3 = _quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return float("-inf"), float("inf")
    return q1 - k * iqr, q3 + k * iqr


def filter_price_outliers(
    items: Sequence[T],
    price_getter: Callable[[T], float],
    k: float = 1.5,
) -> tuple[list[T], list[T]]:
    """
    Split items into (kept, removed) by IQR on their prices.
    Removes fake/extreme listings (e.g. a 1-RON 'iPhone' or a 99,999-RON typo).
    """
    prices = [price_getter(i) for i in items if price_getter(i) > 0]
    if len(prices) < 4:
        return list(items), []

    lower, upper = iqr_bounds(prices, k=k)
    kept: list[T] = []
    removed: list[T] = []
    for it in items:
        p = price_getter(it)
        if p <= 0 or lower <= p <= upper:
            kept.append(it)
        else:
            removed.append(it)
    return kept, removed


# Accessory/non-product keywords — items whose titles strongly suggest an
# accessory rather than the product itself. (Romanian + English.)
ACCESSORY_KEYWORDS = {
    "husa", "huse", "case", "carcasa", "folie", "sticla", "tempered",
    "incarcator", "charger", "cablu", "cable", "adaptor", "adapter",
    "casti", "earbuds", "suport", "stand", "dock", "screen protector",
    "protector", "geam", "display only", "doar display", "placa de baza",
    "piese", "for parts", "dezmembrare", "defect",
}


def looks_like_accessory(title: str) -> bool:
    """Heuristic: True if the title looks like an accessory, not the product."""
    t = title.lower()
    return any(kw in t for kw in ACCESSORY_KEYWORDS)
