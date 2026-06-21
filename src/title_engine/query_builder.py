"""
Canonical eBay query builder from NER-prefilled or fully-labeled tokens.

Produces a clean, English-normalized search query from a Romanian/mixed-language
listing title, suitable for eBay sold-listings lookup.

Usage (at runtime — no LLM required):
    from .tokenizer import tokenize
    from .cross_checker import prefill
    from .query_builder import build_ebay_query

    tokens = tokenize(raw_title)
    tags   = prefill(tokens)          # regex + gazetteer only, ~instant
    query  = build_ebay_query(tokens, tags)

The function also works with full NER-labeled tags when available.

Assembly strategy
-----------------
True in-order traversal: tokens are emitted in their original sequence.
  - BRAND / MODEL / VARIANT span tokens → emitted inline at their position.
  - O tokens: emitted if they pass the signal heuristic (_is_signal_token).
  - Storage DIMENSION tokens (256GB, 1TB) → collected and appended at the end.
  - All other tagged tokens (CONDITION, ACCESSORY, non-storage DIMENSION) → skipped.
  - Noise / filler O tokens → skipped.

Whole-phrase shortcuts (ps5 → PlayStation 5, etc.) are resolved before
tag-based logic and short-circuit the entire assembly.
"""
from __future__ import annotations

import re

from .tokenizer import bio_to_spans, TokenSpan

# ── Noise / filler words ──────────────────────────────────────────────────────
# These are stripped from O-tokens during both assembly and the fallback path.
_FILLER: frozenset[str] = frozenset([
    # Romanian discourse / prepositions
    "vand", "vanzare", "cumpar", "schimb", "de", "la", "cu", "si", "sau",
    "pentru", "in", "pe", "din", "ca", "mai", "nu", "se", "sunt", "este",
    "am", "un", "o", "al", "ai", "ale", "ii", "le", "mi", "iti", "ne", "va",
    # Listing-level noise
    "stare", "buna", "perfecta", "impecabila", "excelenta", "foarte",
    "urgenta", "urgent", "negociabil", "negociabila", "pret", "fix",
    "original", "originala", "cutie", "accesorii", "garantie", "factura",
    "bon", "complet", "full", "set", "kit", "bundle", "edition", "limited",
    # Condition words (irrelevant for eBay model matching)
    "nou", "sigilat", "resigilat", "folosit", "utilizat", "stricat",
    "defect", "functional", "impecabil",
    # Romanian colour words
    "negru", "alb", "gri", "argintiu", "auriu", "rosu", "albastru",
    "verde", "galben", "roz", "violet", "portocaliu",
    # English colour words (including specific Apple / Samsung colour names)
    "black", "white", "silver", "gold", "blue", "red", "green", "grey",
    "gray", "pink", "purple", "orange", "yellow", "bronze",
    "obsidian", "midnight", "starlight", "champagne", "coral", "sierra",
    "graphite", "titanium", "natural", "space", "alpine", "violet",
    "phantom", "mystic", "aqua", "lavender", "cream", "sage", "fog",
    # English filler
    "sale", "sell", "selling", "working", "second", "hand",
    "like", "good", "condition", "used", "new", "open", "sealed",
    "body", "bodies", "only",
    # Category words that carry no eBay model-match value
    "gpu", "cpu", "console", "laptop", "phone", "smartphone", "tablet",
    "foto", "monitor", "desktop", "pc", "gaming",
    # Accessory / peripheral noise
    "husa", "folie", "cablu", "incarcator", "acumulator", "carcasa",
    "manete", "controller", "casti", "headphones", "earphones",
    "noise", "cancelling", "wireless", "bluetooth",
    # Overspecific GPU SKU suffixes that narrow eBay results too much
    "trio", "trio!", "oc", "oc!", "eagle", "windforce", "ventus", "mech",
    "sapphire", "nitro", "pulse", "reference", "founders",
    # Punctuation tokens
    "+", "-", "/", "\\", "|", "&", ".", ",", "!", "?", "x",
])

# Known product-line keywords (O-tagged by prefill, but signal not noise).
_PRODUCT_TOKENS: frozenset[str] = frozenset([
    # Samsung
    "galaxy", "tab",
    # Apple lines (also in _NORMALIZE for display)
    "iphone", "ipad", "macbook", "airpods", "imac", "ipod",
    # Sony / camera lines
    "alpha", "eos", "dslr",
    # Laptop model lines
    "xps", "inspiron", "latitude", "vostro", "precision",   # Dell
    "yoga",                                                   # Lenovo
    "spectre", "envy", "pavilion", "elitebook", "probook",  # HP
    "swift", "aspire", "spin",                               # Acer
    "vivobook", "zenbook",                                   # ASUS
    # Gaming
    "switch",                                                # Nintendo
    "playstation", "dualsense", "dualshock",                 # Sony
    # GPU sub-brands
    "geforce", "radeon",
    # Display / panel tech
    "oled", "amoled", "retina", "qled",
])

# Known variant-modifier tokens.
_VARIANT_TOKENS: frozenset[str] = frozenset([
    "pro", "max", "ultra", "plus", "mini", "lite", "se", "neo",
    "air", "slim", "sport", "fe", "active", "go",
    "fold", "flip", "edge", "turbo", "prime",
    "standard", "classic", "refresh", "extreme",
    "note",   # Samsung Note, but also product line — acceptable either way
    # GPU / CPU qualifiers
    "ti", "xt", "super",
])

# ── Display-form normalizations ───────────────────────────────────────────────
_NORMALIZE: dict[str, str] = {
    # Brands
    "apple": "Apple", "samsung": "Samsung", "huawei": "Huawei",
    "xiaomi": "Xiaomi", "redmi": "Redmi", "poco": "POCO", "oppo": "OPPO",
    "oneplus": "OnePlus", "realme": "Realme", "nokia": "Nokia",
    "motorola": "Motorola", "lenovo": "Lenovo", "asus": "ASUS", "rog": "ROG",
    "google": "Google", "pixel": "Pixel", "sony": "Sony", "lg": "LG",
    "htc": "HTC", "honor": "Honor", "dell": "Dell", "hp": "HP",
    "acer": "Acer", "msi": "MSI", "razer": "Razer", "toshiba": "Toshiba",
    "nvidia": "NVIDIA", "geforce": "GeForce", "amd": "AMD", "radeon": "Radeon",
    "intel": "Intel", "canon": "Canon", "nikon": "Nikon",
    "fujifilm": "Fujifilm", "olympus": "Olympus", "panasonic": "Panasonic",
    "nintendo": "Nintendo", "playstation": "PlayStation", "xbox": "Xbox",
    "corsair": "Corsair", "logitech": "Logitech", "bose": "Bose",
    "jbl": "JBL", "sennheiser": "Sennheiser", "valve": "Valve",
    "surface": "Surface", "thinkpad": "ThinkPad", "ideapad": "IdeaPad",
    "legion": "Legion",
    # Apple product lines
    "iphone": "iPhone", "ipad": "iPad", "macbook": "MacBook",
    "airpods": "AirPods", "imac": "iMac", "ipod": "iPod",
    # GPU shorthands
    "rtx": "RTX", "gtx": "GTX", "rx": "RX", "arc": "Arc",
    # Units
    "gb": "GB", "tb": "TB", "mb": "MB", "mah": "mAh",
    "ghz": "GHz", "mhz": "MHz", "hz": "Hz", "mp": "MP",
    # Variant keywords
    "pro": "Pro", "max": "Max", "ultra": "Ultra", "plus": "Plus",
    "mini": "Mini", "lite": "Lite", "se": "SE", "neo": "Neo",
    "air": "Air", "slim": "Slim", "sport": "Sport",
    "fe": "FE", "active": "Active", "go": "Go", "fold": "Fold",
    "flip": "Flip", "note": "Note", "edge": "Edge", "turbo": "Turbo",
    "prime": "Prime", "ti": "Ti", "xt": "XT", "super": "Super",
    # Product / model line keywords
    "switch": "Switch", "oled": "OLED", "amoled": "AMOLED", "qled": "QLED",
    "galaxy": "Galaxy", "alpha": "Alpha",
    "xps": "XPS", "yoga": "Yoga", "spectre": "Spectre", "envy": "Envy",
    "pavilion": "Pavilion", "swift": "Swift", "aspire": "Aspire",
    "vivobook": "VivoBook", "zenbook": "ZenBook",
    "dualsense": "DualSense", "dualshock": "DualShock",
    "eos": "EOS", "dslr": "DSLR",
    # Misc model words
    "mark": "Mark", "gen": "Gen",
}

# Whole-phrase shortcuts — short-circuit the entire assembly.
_MODEL_SHORTCUTS: dict[str, str] = {
    "ps5": "PlayStation 5",
    "ps4": "PlayStation 4",
    "ps3": "PlayStation 3",
    "ps2": "PlayStation 2",
    "ps1": "PlayStation 1",
    "ps5 slim": "PlayStation 5 Slim",
    "ps4 pro": "PlayStation 4 Pro",
    "ps4 slim": "PlayStation 4 Slim",
    "nsw": "Nintendo Switch",
}

# Patterns compiled once.
_STORAGE_RE = re.compile(r'^\d+(gb|tb)$', re.I)
_ATTACHED_STORAGE_RE = re.compile(r'^(\d+)(gb|tb)$', re.I)
_ALPHA_NUM_CODE_RE = re.compile(r'^([a-zA-Z]{1,4})(\d+[a-zA-Z]?\d*)$')   # S24, M3, XM5, i7
_NUM_ALPHA_CODE_RE = re.compile(r'^\d+[a-zA-Z]+\d*$')                     # 5D, 5DS, 3DS
_HYPHEN_MODEL_RE = re.compile(r'^[a-zA-Z]{1,4}-[a-zA-Z0-9]', re.I)        # WH-1000XM5, EF-S


def _display(token: str) -> str:
    """Preferred display form of a single (lowercase) token."""
    low = token.lower()
    if low in _MODEL_SHORTCUTS:
        return _MODEL_SHORTCUTS[low]
    if low in _NORMALIZE:
        return _NORMALIZE[low]
    # Attached storage: "256gb" → "256GB"
    m = _ATTACHED_STORAGE_RE.match(low)
    if m:
        return m.group(1) + m.group(2).upper()
    # Pure digits stay as-is
    if token.isdigit():
        return token
    # Alpha+num codes: s24 → S24, m3 → M3, i7 → I7, xm5 → XM5
    mc = _ALPHA_NUM_CODE_RE.match(low)
    if mc:
        return mc.group(1).upper() + mc.group(2).upper()
    # Num+alpha codes: 5d → 5D, 3ds → 3DS
    if _NUM_ALPHA_CODE_RE.match(low):
        return token.upper()
    # Short all-alpha identifiers
    if len(token) <= 4 and token.isalpha():
        return token.upper()
    # Hyphenated model codes: wh-1000xm5 → WH-1000XM5
    if _HYPHEN_MODEL_RE.match(low):
        return token.upper()
    return token


def _span_text(tokens: list[str], span: TokenSpan) -> str:
    return " ".join(_display(t) for t in tokens[span.start : span.end])


def _is_storage_dim(tokens: list[str], span: TokenSpan) -> bool:
    return all(_STORAGE_RE.match(t) for t in tokens[span.start : span.end])


def _is_signal_token(tok: str) -> bool:
    """
    True if an O-tagged token is product signal rather than noise.
    Note: digit check comes before the length guard so single-digit
    version numbers (Pixel 8, iPhone 8) are not filtered out.
    """
    low = tok.lower()
    if low in _FILLER:
        return False
    # All-digit version numbers (8, 15, 4080, …)
    if tok.isdigit():
        return True
    if len(tok) <= 1:
        return False
    if low in _PRODUCT_TOKENS or low in _VARIANT_TOKENS:
        return True
    if _ATTACHED_STORAGE_RE.match(low):
        return True
    if _ALPHA_NUM_CODE_RE.match(low):   # S24, M3, i7
        return True
    if _NUM_ALPHA_CODE_RE.match(low):   # 5D, 3DS
        return True
    if _HYPHEN_MODEL_RE.match(low):     # WH-1000XM5
        return True
    # Short all-alpha tokens not caught above
    if tok.isalpha() and 2 <= len(tok) <= 8:
        return True
    return False


def build_ebay_query(
    tokens: list[str],
    tags: list[str],
    max_words: int = 8,
) -> str:
    """
    Build a clean eBay search query from tokenized title + BIO tags.

    Tags may come from full NER labeling OR the fast regex/gazetteer prefill.
    Both work; full NER produces better MODEL/VARIANT coverage.

    Tokens are emitted in their original sequence order, so model numbers
    that appear between brand tokens are preserved correctly.

    Args:
        tokens:    Lowercase tokens from ``tokenizer.tokenize()``.
        tags:      BIO tag list of the same length as tokens.
        max_words: Maximum number of words in the returned query.

    Returns:
        A clean string for ``EbaySoldPricer.get_market_data()``.
    """
    if len(tokens) != len(tags):
        return _fallback(tokens, max_words)

    # ── 0. Whole-phrase shortcuts (ps5, ps4, nsw …) ───────────────────────────
    raw_phrase = " ".join(tokens)
    if raw_phrase in _MODEL_SHORTCUTS:
        return _MODEL_SHORTCUTS[raw_phrase]
    sig_toks = [t for t in tokens if t not in _FILLER and len(t) > 1]
    two_tok = " ".join(sig_toks[:2])
    if two_tok in _MODEL_SHORTCUTS:
        return _MODEL_SHORTCUTS[two_tok]
    if sig_toks and sig_toks[0] in _MODEL_SHORTCUTS:
        return _MODEL_SHORTCUTS[sig_toks[0]]

    spans = bio_to_spans(tags)

    # Quick guard — no BRAND or MODEL anywhere → pure fallback
    if not any(s.label in ("BRAND", "MODEL") for s in spans):
        return _fallback(tokens, max_words)

    # ── 1. Per-span MODEL shortcut check ─────────────────────────────────────
    for span in spans:
        if span.label == "MODEL":
            phrase = " ".join(tokens[span.start : span.end])
            if phrase in _MODEL_SHORTCUTS:
                return _MODEL_SHORTCUTS[phrase]

    # ── 2. Build span lookup structures ──────────────────────────────────────
    span_at: dict[int, TokenSpan] = {}  # start index → span
    covered: set[int] = set()
    for s in spans:
        span_at[s.start] = s
        for idx in range(s.start, s.end):
            covered.add(idx)

    # ── 3. In-order assembly ──────────────────────────────────────────────────
    # Traverse tokens left-to-right.  Brand/Model/Variant span content is
    # emitted inline.  Signal O-tokens are emitted inline.  Storage DIMENSION
    # tokens are deferred and appended at the end.
    inline: list[str] = []
    storage: list[str] = []

    i = 0
    while i < len(tokens):
        if i in span_at:
            span = span_at[i]
            if span.label in ("BRAND", "MODEL", "VARIANT"):
                inline.append(_span_text(tokens, span))
            elif span.label == "DIMENSION" and _is_storage_dim(tokens, span):
                storage.append(_span_text(tokens, span))
            # Other spans (CONDITION, ACCESSORY, non-storage DIMENSION) → skip
            i = span.end
        elif i in covered:
            # Interior of a multi-token span — already handled at its start
            i += 1
        else:
            tok = tokens[i]
            tag = tags[i]
            # Any non-O tag here means a span we chose to skip above
            if tag != "O":
                i += 1
                continue
            if _is_signal_token(tok):
                # Route untagged storage to deferred list; everything else inline
                if _STORAGE_RE.match(tok) or _ATTACHED_STORAGE_RE.match(tok):
                    if not storage:  # keep only the first untagged storage token
                        storage.append(_display(tok))
                else:
                    inline.append(_display(tok))
            i += 1

    all_words = (" ".join(inline + storage)).split()
    return " ".join(all_words[:max_words])


def _fallback(tokens: list[str], max_words: int = 8) -> str:
    """
    Filler-filtered fallback: strip noise, normalize, and join.
    Used when no BRAND/MODEL spans exist at all.
    """
    kept: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _FILLER or len(tok) <= 1:
            continue
        kept.append(_display(tok))
        if len(kept) >= max_words:
            break
    return " ".join(kept)
