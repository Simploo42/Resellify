"""
Title Engine pipeline — spec §6 / §7.

Flow:
  raw titles → dedup → prefill → label (Haiku) → validate → cross-check
           → train_v0.jsonl   (validated, spot-checked clean)
           → spot_check_queue.jsonl  (needs human review)

CLI:
  python -m src.title_engine.pipeline --input listings.jsonl [--output-dir data/]
  python -m src.title_engine.pipeline --from-db              [--limit 3000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .deduper import Deduper
from .labeler import HaikuLabeler, LabelRecord, CONFIDENCE_FLOOR
from .cross_checker import prefill, cross_check
from .tokenizer import tokenize
from .validator import validate, Disposition

DEFAULT_OUTPUT_DIR = Path("data/title_engine")

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(records: list[LabelRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(r.to_jsonl() + "\n")
    print(f"[Pipeline] Wrote {len(records)} records → {path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

class TitleEnginePipeline:
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        dedup_threshold: float = 0.85,
        confidence_floor: float = CONFIDENCE_FLOOR,
        batch_size: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.deduper = Deduper(threshold=dedup_threshold)
        self.labeler = HaikuLabeler(batch_size=batch_size)
        self.confidence_floor = confidence_floor

    async def run(self, raw_titles: list[dict]) -> dict:
        """
        raw_titles: list of {id, raw}. id should be the listing_id (e.g. "olx:12345").

        Returns a stats dict.
        """
        start = datetime.utcnow()
        print(f"[Pipeline] Starting. Input: {len(raw_titles)} titles")

        # ── Step 1: dedup ──────────────────────────────────────────────────────
        unique = self.deduper.deduplicate(raw_titles)
        print(f"[Pipeline] After dedup: {len(unique)} unique titles")

        # ── Step 2: prefill spans from regex + gazetteer ───────────────────────
        for item in unique:
            tokens = tokenize(item["raw"])
            item["prefilled_tags"] = prefill(tokens)

        # ── Step 3: label with Haiku ───────────────────────────────────────────
        print(f"[Pipeline] Labeling {len(unique)} titles with {self.labeler.model}…")
        records = await self.labeler.label_titles(unique)
        print(f"[Pipeline] Labeling done. {len(records)} records received.")

        # ── Step 4: validate + cross-check ────────────────────────────────────
        train_set: list[LabelRecord] = []
        spot_check: list[LabelRecord] = []
        rejected: list[LabelRecord] = []

        for record in records:
            val = validate(record, confidence_floor=self.confidence_floor)

            if val.disposition == Disposition.REJECT:
                print(f"[Pipeline] REJECT {record.id}: {'; '.join(val.issues)}")
                rejected.append(record)
                continue

            # Cross-check LLM tags vs regex/gazetteer
            cc_issues = cross_check(record.tokens, record.tags)
            if cc_issues:
                for issue in cc_issues:
                    print(f"[Pipeline] CROSS-CHECK {record.id}: {issue}")
                record.needs_review = True
                if val.disposition == Disposition.TRAIN:
                    val.issues.extend(cc_issues)
                    record.needs_review = True

            if record.needs_review or val.disposition == Disposition.SPOT_CHECK:
                spot_check.append(record)
            else:
                train_set.append(record)

        # ── Step 5: write outputs ──────────────────────────────────────────────
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        write_jsonl(train_set, self.output_dir / "train_v0.jsonl")
        write_jsonl(spot_check, self.output_dir / "spot_check_queue.jsonl")
        if rejected:
            write_jsonl(rejected, self.output_dir / f"rejected_{ts}.jsonl")

        elapsed = (datetime.utcnow() - start).total_seconds()
        stats = {
            "input": len(raw_titles),
            "after_dedup": len(unique),
            "labeled": len(records),
            "train_set": len(train_set),
            "spot_check": len(spot_check),
            "rejected": len(rejected),
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"[Pipeline] Done in {elapsed:.1f}s: {stats}")
        return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

async def _main():
    parser = argparse.ArgumentParser(description="Title Engine NER labeling pipeline")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--input", help="Path to JSONL file with {id, raw} records")
    src_group.add_argument("--from-db", action="store_true", help="Load titles from Resellify SQLite DB")
    parser.add_argument("--limit", type=int, default=3000, help="Max titles to process (default 3000)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--confidence-floor", type=float, default=CONFIDENCE_FLOOR)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    raw_titles: list[dict] = []

    if args.input:
        raw_titles = load_jsonl(args.input)
        print(f"[Pipeline] Loaded {len(raw_titles)} titles from {args.input}")

    elif args.from_db:
        from sqlalchemy import select
        from ..db.models import Listing, AsyncSessionLocal, init_db
        await init_db()
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Listing.id, Listing.title)
                .where(Listing.is_active == True)
                .limit(args.limit)
            )
            rows = result.all()
        raw_titles = [{"id": r[0], "raw": r[1]} for r in rows if r[1]]
        print(f"[Pipeline] Loaded {len(raw_titles)} titles from DB")

    if not raw_titles:
        print("[Pipeline] No titles to process.")
        return

    raw_titles = raw_titles[:args.limit]

    pipeline = TitleEnginePipeline(
        output_dir=Path(args.output_dir),
        dedup_threshold=args.dedup_threshold,
        confidence_floor=args.confidence_floor,
        batch_size=args.batch_size,
    )
    await pipeline.run(raw_titles)


if __name__ == "__main__":
    asyncio.run(_main())
