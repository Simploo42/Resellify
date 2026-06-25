"""OCR processing and card parsing for courier PDA screenshots."""

import re
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded EasyOCR reader
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # noqa: import deferred to avoid startup cost
        logger.info("Loading EasyOCR model (first run downloads ~100 MB)…")
        _reader = easyocr.Reader(["ro", "en"], gpu=False)
        logger.info("EasyOCR ready.")
    return _reader


# ── Patterns ──────────────────────────────────────────────────────────────────

# Romanian phone: 07xx xxxxxx, 02xx xxxxxx, +40 …
_PHONE_RE = re.compile(r"\b(?:\+40|0)[0-9]{9}\b")

# AWB: 8-20 alphanumeric chars (digits or uppercase + digits combos)
_AWB_RE = re.compile(r"\b[A-Z0-9]{8,20}\b")

# Field-label keywords (lower-case substrings)
_AWB_LABELS    = {"awb", "colet", "barcode", "barcod", "expeditie", "nr.colet",
                  "nr colet", "cod exp", "spedition", "nr.exp"}
_NAME_LABELS   = {"destinatar", "beneficiar", "nume", "client", "persoana",
                  "destinator", "receiver", "dest."}
_ADDR_LABELS   = {"adresa", "strada", "adresa livrare", "loc.livrare",
                  "adresa dest", "adresa:", "loc.", "localitate"}
# Note: "gsm" intentionally excluded — Romanian PDAs use it as a building code in addresses
_PHONE_LABELS  = {"telefon", "tel.", "tel:", "mobil", "nr.tel", "nr tel", "phone"}
# Lines whose label should be silently skipped (not part of the card fields we need)
_SKIP_LABELS   = {"interval", "interval:", "greutate", "volum", "serviciu",
                  "obs", "observ", "mentiune", "continut", "valoare"}


def _has_label(line_lower: str, labels: set[str]) -> bool:
    return any(lbl in line_lower for lbl in labels)


def _value_after_colon(text: str) -> Optional[str]:
    if ":" in text:
        val = text.split(":", 1)[1].strip()
        return val or None
    return None


def _norm_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("40") and len(digits) == 11:
        return "0" + digits[2:]
    return digits


def _join_addr(parts: list[str]) -> str:
    """Join multi-line address parts, stripping bare '/' continuation markers."""
    cleaned = [p.rstrip("/ ").strip() for p in parts if p.strip().strip("/")]
    return ", ".join(cleaned)


def _extract_awb(text: str) -> Optional[str]:
    """Try colon-value first, then any 8+-digit/char run."""
    val = _value_after_colon(text)
    if val:
        m = _AWB_RE.search(val.upper())
        if m:
            return m.group()
    m = _AWB_RE.search(text.upper())
    return m.group() if m else None


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return denoised


# ── Card parser ───────────────────────────────────────────────────────────────

def _parse_cards(text_lines: list[str]) -> list[dict]:
    """
    State-machine parser: iterates OCR lines top→bottom.
    Emits a card only when all four fields are present.
    """
    cards: list[dict] = []
    cur: dict = {}
    addr_parts: list[str] = []
    in_addr = False

    def _flush():
        nonlocal addr_parts, in_addr
        if addr_parts and "address" not in cur:
            cur["address"] = _join_addr(addr_parts)
            addr_parts.clear()
        in_addr = False
        if all(k in cur for k in ("awb", "name", "address", "phone")):
            if all(str(cur[k]).strip() for k in ("awb", "name", "address", "phone")):
                cards.append(dict(cur))

    for raw in text_lines:
        line = raw.strip()
        if not line:
            continue
        ll = line.lower()

        # ── Skip irrelevant labelled fields (Interval, Greutate, etc.) ───────
        if _has_label(ll, _SKIP_LABELS) and ":" in line:
            in_addr = False
            if addr_parts and "address" not in cur:
                cur["address"] = _join_addr(addr_parts)
                addr_parts.clear()
            continue

        # ── AWB ──────────────────────────────────────────────────────────────
        if _has_label(ll, _AWB_LABELS):
            _flush()
            cur = {}
            val = _extract_awb(line)
            if val:
                cur["awb"] = val
            continue

        # ── Name ─────────────────────────────────────────────────────────────
        if _has_label(ll, _NAME_LABELS) and "name" not in cur:
            in_addr = False
            if addr_parts and "address" not in cur:
                cur["address"] = _join_addr(addr_parts)
                addr_parts.clear()
            val = _value_after_colon(line)
            if val:
                cur["name"] = val
            continue

        # ── Address ───────────────────────────────────────────────────────────
        if _has_label(ll, _ADDR_LABELS) and "address" not in cur:
            in_addr = True
            addr_parts.clear()
            val = _value_after_colon(line)
            if val:
                addr_parts.append(val)
            continue

        # ── Phone ─────────────────────────────────────────────────────────────
        ph_match = _PHONE_RE.search(line)
        if ph_match and "phone" not in cur:
            if in_addr:
                cur["address"] = _join_addr(addr_parts)
                addr_parts.clear()
                in_addr = False
            cur["phone"] = _norm_phone(ph_match.group())
            continue

        if _has_label(ll, _PHONE_LABELS) and "phone" not in cur:
            if in_addr:
                cur["address"] = _join_addr(addr_parts)
                addr_parts.clear()
                in_addr = False
            val = _value_after_colon(line)
            if val:
                m = _PHONE_RE.search(val)
                if m:
                    cur["phone"] = _norm_phone(m.group())
            continue

        # ── Address continuation ──────────────────────────────────────────────
        if in_addr and "address" not in cur:
            if not _has_label(ll, _AWB_LABELS | _NAME_LABELS | _PHONE_LABELS):
                addr_parts.append(line)

    _flush()
    return cards


# ── Public API ────────────────────────────────────────────────────────────────

def process_image(image_bytes: bytes) -> list[dict]:
    """Run OCR on raw image bytes, return list of complete card dicts."""
    img = _preprocess(image_bytes)
    reader = _get_reader()
    results = reader.readtext(img)

    # Sort top→bottom by bounding-box y of top-left corner
    results.sort(key=lambda r: r[0][0][1])
    # Keep only results with reasonable confidence
    lines = [r[1] for r in results if r[2] > 0.25]

    return _parse_cards(lines)
