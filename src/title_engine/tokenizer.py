"""
Shared tokenizer — MUST be imported by labeler, trainer, and inference.
Never copy-paste this logic; always import from here.

Spec §1: "the tokenizer used here must be byte-for-byte the same one used
at training and inference."
"""
import re
from typing import NamedTuple

TOKENIZER_VERSION = "v1"
SCHEMA_VERSION = "v1"

# ── Stage-A sanitizer ─────────────────────────────────────────────────────────

# Collapse visual separators to a single space (includes em/en dashes)
_SEP_RE = re.compile(r'[_]{1,}|/{1,}|\|{1,}|•|—|–')

# Pad "x"/"×" ONLY when it sits between two digits: "128x64" → "128 x 64".
# Does NOT fire on words like "Galaxy", "Max", "Xbox".
_DIGIT_X_DIGIT_RE = re.compile(r'(\d)\s*[xX×]\s*(\d)')

# Pad punctuation ONLY when it is NOT between two digits.
# Negative lookbehind/lookahead for digit so "6.1" / "66,34" survive intact.
_PUNCT_NOT_BETWEEN_DIGITS = re.compile(r'(?<!\d)[.,](?!\d)|(?<=\d)[.,](?!\d)|(?<!\d)[.,](?=\d)')

# Collapse runs of whitespace
_WS_RE = re.compile(r'\s+')


def sanitize(text: str) -> str:
    """Stage-A sanitizer. Returns a cleaned string, NOT yet tokenized."""
    text = _SEP_RE.sub(' ', text)
    text = _DIGIT_X_DIGIT_RE.sub(r'\1 x \2', text)
    text = _PUNCT_NOT_BETWEEN_DIGITS.sub(' ', text)
    return _WS_RE.sub(' ', text).strip()


def tokenize(text: str) -> list[str]:
    """
    Canonical tokenization: sanitize then lowercase whitespace-split.
    Each space-separated unit = one token. This is the frozen v1 behaviour.
    """
    return sanitize(text).lower().split()


# ── Token span helpers ────────────────────────────────────────────────────────

class TokenSpan(NamedTuple):
    start: int   # inclusive token index
    end: int     # exclusive token index
    label: str   # entity type (no B-/I- prefix)


def spans_to_bio(num_tokens: int, spans: list[TokenSpan]) -> list[str]:
    """Convert a list of TokenSpan objects to a BIO tag list of length num_tokens."""
    tags = ["O"] * num_tokens
    for span in spans:
        for i in range(span.start, min(span.end, num_tokens)):
            tags[i] = f"{'B' if i == span.start else 'I'}-{span.label}"
    return tags


def bio_to_spans(tags: list[str]) -> list[TokenSpan]:
    """Convert a BIO tag list back to a list of TokenSpan objects."""
    spans: list[TokenSpan] = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t.startswith("B-"):
            label = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                j += 1
            spans.append(TokenSpan(i, j, label))
            i = j
        else:
            i += 1
    return spans


# ── BIO validation ────────────────────────────────────────────────────────────

VALID_LABELS = {
    "BRAND", "MODEL", "VARIANT", "DIMENSION", "COLOR",
    "MATERIAL", "CONDITION", "IDENTIFIER", "ACCESSORY", "QUANTITY",
}


def repair_bio(tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Auto-repair simple illegal BIO sequences.
    Returns (repaired_tags, list_of_repairs_made).

    Repairs:
    - Orphan I-X (no preceding B-X / I-X of same type) → promote to B-X
    - Unknown label → replace with O

    Anything else is left for the caller to reject.
    """
    repaired = list(tags)
    repairs: list[str] = []

    for i, tag in enumerate(repaired):
        if tag == "O":
            continue
        if not (tag.startswith("B-") or tag.startswith("I-")):
            repaired[i] = "O"
            repairs.append(f"idx {i}: unknown tag {tag!r} → O")
            continue
        prefix, label = tag[:2], tag[2:]
        if label not in VALID_LABELS:
            repaired[i] = "O"
            repairs.append(f"idx {i}: unknown label {label!r} → O")
            continue
        if prefix == "I-":
            prev = repaired[i - 1] if i > 0 else "O"
            if prev not in (f"B-{label}", f"I-{label}"):
                repaired[i] = f"B-{label}"
                repairs.append(f"idx {i}: orphan I-{label} → B-{label}")

    return repaired, repairs
