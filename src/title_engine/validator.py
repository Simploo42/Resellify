"""
Validation gates — spec §4.

Gate 1: length mismatch        → hard reject, re-request once
Gate 2: illegal BIO            → auto-repair simple cases, reject rest
Gate 3: schema sanity          → zero non-O, or 3+ distinct BRAND spans
Gate 4: confidence floor       → route to spot-check queue
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from .tokenizer import repair_bio, bio_to_spans, VALID_LABELS

if TYPE_CHECKING:
    from .labeler import LabelRecord


class Disposition(Enum):
    TRAIN = auto()          # safe to include in training set
    SPOT_CHECK = auto()     # needs human review before training
    REJECT = auto()         # hard reject, discard


@dataclass
class ValidationResult:
    disposition: Disposition
    issues: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.disposition != Disposition.REJECT


def validate(record: "LabelRecord", confidence_floor: float = 0.75) -> ValidationResult:
    issues: list[str] = []
    repairs: list[str] = []
    disposition = Disposition.TRAIN

    # ── Gate 1: length mismatch ───────────────────────────────────────────────
    if len(record.tags) != len(record.tokens):
        issues.append(
            f"length mismatch: {len(record.tags)} tags vs {len(record.tokens)} tokens"
        )
        return ValidationResult(Disposition.REJECT, issues)

    # ── Gate 2: illegal BIO ────────────────────────────────────────────────────
    repaired_tags, bio_repairs = repair_bio(record.tags)
    if bio_repairs:
        repairs.extend(bio_repairs)
        record.tags = repaired_tags
        record.repairs.extend(bio_repairs)

    # Check for any tags that couldn't be repaired (still malformed after repair pass)
    for i, tag in enumerate(record.tags):
        if tag == "O":
            continue
        if not (tag.startswith("B-") or tag.startswith("I-")):
            issues.append(f"unrepairable tag at idx {i}: {tag!r}")
            return ValidationResult(Disposition.REJECT, issues, repairs)
        label = tag[2:]
        if label not in VALID_LABELS:
            issues.append(f"unknown label at idx {i}: {label!r}")
            return ValidationResult(Disposition.REJECT, issues, repairs)

    # ── Gate 3: schema sanity ─────────────────────────────────────────────────
    non_o = [t for t in record.tags if t != "O"]
    if not non_o:
        issues.append("all tokens tagged O — likely labeling failure or junk title")
        disposition = Disposition.SPOT_CHECK

    brand_spans = [s for s in bio_to_spans(record.tags) if s.label == "BRAND"]
    if len(brand_spans) >= 3:
        issues.append(f"{len(brand_spans)} distinct BRAND spans — multi-product/junk title")
        disposition = Disposition.SPOT_CHECK

    # ── Gate 4: confidence floor + uncertain_idx ──────────────────────────────
    if record.confidence < confidence_floor:
        issues.append(f"confidence {record.confidence:.2f} below floor {confidence_floor:.2f}")
        disposition = Disposition.SPOT_CHECK

    if record.uncertain_idx:
        issues.append(f"uncertain tokens at idx {record.uncertain_idx}")
        disposition = Disposition.SPOT_CHECK

    record.needs_review = (disposition == Disposition.SPOT_CHECK)
    return ValidationResult(disposition, issues, repairs)
