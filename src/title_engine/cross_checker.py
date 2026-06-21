"""
Cross-checker — spec §5.

Compares LLM tags against two cheap ground-truth sources:
  1. Regex patterns for DIMENSION and VARIANT
  2. Brand gazetteer for BRAND

Disagreements are flagged as issues that route the record to spot-check.
Also pre-fills spans BEFORE labeling so the LLM labels fewer tokens from scratch
(cheaper and more consistent).
"""
import os
import re
from pathlib import Path

from .tokenizer import tokenize, TokenSpan, spans_to_bio, bio_to_spans

# ── Gazetteer ─────────────────────────────────────────────────────────────────

_BRANDS_FILE = Path(__file__).parent / "data" / "brands.txt"


def _load_brands() -> frozenset[str]:
    if not _BRANDS_FILE.exists():
        return frozenset()
    lines = _BRANDS_FILE.read_text().splitlines()
    return frozenset(l.strip().lower() for l in lines if l.strip() and not l.startswith("#"))


_BRANDS: frozenset[str] = _load_brands()


def reload_brands():
    """Call after editing brands.txt at runtime."""
    global _BRANDS
    _BRANDS = _load_brands()


# ── Regex patterns ────────────────────────────────────────────────────────────

# DIMENSION: measured physical/electrical quantities
_DIM_PATTERNS = [
    # Storage (used as dimension when it's system RAM / screen/module spec, not phone storage trim)
    # We flag both and let VARIANT/DIMENSION disambiguation happen in the LLM.
    re.compile(r'^\d+\s*(gb|tb|mb|kb)$', re.I),            # 32GB, 512GB
    re.compile(r'^\d+\s*(mah)$', re.I),                     # 5000mAh
    re.compile(r'^\d+(\.\d+)?\s*(w|watts?)$', re.I),        # 65W, 1000 watts
    re.compile(r'^\d+(\.\d+)?\s*(hz|mhz|ghz|khz)$', re.I), # 120Hz, 2.4GHz
    re.compile(r'^\d+(\.\d+)?\s*(mm|cm|m|in|inch|")$', re.I), # 6.1in, 15.6"
    re.compile(r'^\d+\s*(mp|megapixels?)$', re.I),           # 50MP
    re.compile(r'^\d+\s*(v|volts?)$', re.I),                 # 12V
    re.compile(r'^\d+k$', re.I),                              # 4K
    re.compile(r'^(full\s*hd|fhd|qhd|uhd|4k|8k|2k)$', re.I),
    re.compile(r'^ddr[0-9]?$', re.I),                        # DDR5
    re.compile(r'^\d{3,4}p$', re.I),                         # 1080p, 720p
]

# VARIANT: product trim/edition keywords
_VARIANT_TOKENS = frozenset([
    "pro", "max", "ultra", "plus", "mini", "lite", "se", "neo", "edge",
    "fold", "flip", "note", "air", "slim", "sport", "gaming", "extreme",
    "fe", "active", "go", "view", "refresh", "turbo", "prime", "one",
    "standard", "classic", "basic", "advanced", "elite", "supreme",
    # Romanian
    "nou", "sigilat", "resigilat", "recondiționat",
])

# CONDITION tokens
_CONDITION_TOKENS = frozenset([
    "nou", "sigilat", "resigilat", "folosit", "utilizat", "stricat",
    "defect", "functional", "impecabil", "excelent",
    "ca", "nou",   # "ca nou" is a 2-token span
    "pentru", "piese",  # "pentru piese" = for parts
    "new", "used", "refurbished", "broken", "faulty",
])

# ACCESSORY nouns (single-token)
_ACCESSORY_TOKENS = frozenset([
    "husa", "huse", "folie", "sticla", "cablu", "incarcator", "acumulator",
    "carcasa", "suport", "dock", "stand", "adaptor", "case", "charger",
    "cable", "cover", "protector", "sleeve", "pouch", "strap",
])


def _regex_spans(tokens: list[str]) -> list[TokenSpan]:
    """Return regex-confident spans. Only tags tokens individually (no multi-token regex spans)."""
    spans: list[TokenSpan] = []
    for i, tok in enumerate(tokens):
        if any(p.match(tok) for p in _DIM_PATTERNS):
            spans.append(TokenSpan(i, i + 1, "DIMENSION"))
        elif tok in _ACCESSORY_TOKENS:
            spans.append(TokenSpan(i, i + 1, "ACCESSORY"))
    return spans


def _gazetteer_spans(tokens: list[str]) -> list[TokenSpan]:
    """
    Find BRAND spans using the gazetteer.
    Tries longest-match first (multi-word brands like "ground zero").
    """
    spans: list[TokenSpan] = []
    i = 0
    while i < len(tokens):
        matched = False
        # Try 3-gram, 2-gram, 1-gram
        for length in (3, 2, 1):
            if i + length <= len(tokens):
                phrase = " ".join(tokens[i:i + length])
                if phrase in _BRANDS:
                    spans.append(TokenSpan(i, i + length, "BRAND"))
                    i += length
                    matched = True
                    break
        if not matched:
            i += 1
    return spans


def prefill(tokens: list[str]) -> list[str]:
    """
    Build a pre-filled tag list from regex + gazetteer.
    O = unknown (LLM should fill). Non-O = confident pre-fill.
    This is sent alongside the tokens in the labeling prompt.
    """
    tags = ["O"] * len(tokens)
    for span in _regex_spans(tokens):
        for i in range(span.start, span.end):
            tags[i] = f"{'B' if i == span.start else 'I'}-{span.label}"
    for span in _gazetteer_spans(tokens):
        # Don't overwrite a regex tag
        if all(tags[i] == "O" for i in range(span.start, span.end)):
            for i in range(span.start, span.end):
                tags[i] = f"{'B' if i == span.start else 'I'}-{span.label}"
    return tags


def cross_check(tokens: list[str], llm_tags: list[str]) -> list[str]:
    """
    Compare LLM tags against regex + gazetteer.
    Returns a list of disagreement descriptions (empty = no issues).
    """
    issues: list[str] = []
    pre = prefill(tokens)

    for i, (p, l) in enumerate(zip(pre, llm_tags)):
        if p == "O" or p == l:
            continue
        p_label = p[2:] if len(p) > 2 else p
        l_label = l[2:] if len(l) > 2 else l
        if p_label != l_label:
            issues.append(
                f"idx {i} token={tokens[i]!r}: "
                f"regex/gazetteer={p!r} vs llm={l!r}"
            )
    return issues
