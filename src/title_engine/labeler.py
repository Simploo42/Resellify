"""
Haiku BIO labeler — sends batches of pre-tokenized titles to Claude Haiku,
gets back BIO tags + confidence + uncertain_idx per title.

Spec §2: temperature 0, JSON-only, uncertain_idx field, pre-filled regex/gazetteer spans.
"""
import json
import os
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import anthropic

from .tokenizer import tokenize, TOKENIZER_VERSION, SCHEMA_VERSION, VALID_LABELS

LABELER_MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 8          # titles per API call
CONFIDENCE_FLOOR = 0.75 # below this → spot-check queue


@dataclass
class LabelRecord:
    id: str
    raw: str
    tokens: list[str]
    tags: list[str]
    confidence: float
    uncertain_idx: list[int]
    prefilled_spans: list[dict]   # spans pre-filled from regex/gazetteer
    labeler: str = LABELER_MODEL
    labeler_version: str = ""
    schema_version: str = SCHEMA_VERSION
    repairs: list[str] = field(default_factory=list)
    needs_review: bool = False    # confidence < floor OR uncertain_idx non-empty

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelRecord":
        return cls(**d)


_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise sequence-labeling tool for second-hand marketplace
    product titles (Romanian and English, often mixed). You are given product
    titles already split into tokens. Label EACH token with exactly one BIO
    tag from the schema below. Output JSON only — no prose, no markdown.

    ENTITY TYPES:
    - BRAND: manufacturer/make (Apple, Samsung, Dacia, Ground Zero)
    - MODEL: product line/model (iPhone 13, Edge 70 Fusion, Legion GO)
    - VARIANT: storage/trim/generation/edition (128GB, Pro Max, Z1 Extreme)
    - DIMENSION: size/capacity/power/screen (6.1in, 50", 1000W, 32GB, DDR5)
    - COLOR: colour (negru, blue, orange-as-colour)
    - MATERIAL: material (piele, sticla, aluminiu)
    - CONDITION: state (nou, utilizat, sigilat, pentru piese, ca nou)
    - IDENTIFIER: serial/part-number/model-code (A07, QE50Q77AAU, B1)
    - ACCESSORY: accessory/part noun (husa, folie, cablu, incarcator,
      acumulator, carcasa, suport) — tag ONLY the accessory noun itself
    - QUANTITY: count words/numbers (buc, pereche, set, 2x)
    - O: anything else (filler, verbs, "vand", prepositions)

    BIO RULES:
    - B-X marks the FIRST token of an entity span, I-X each subsequent token
      of the SAME entity span. Single-token entity = B-X only.
    - "iphone 13" → B-MODEL I-MODEL. "pro max" → B-VARIANT I-VARIANT.
    - Output one tag per input token, in the same order, same length.

    DISAMBIGUATION (critical):
    - A brand word used as a colour/fruit/common word is NOT a brand.
      "orange" as a colour → B-COLOR. "apple iphone" → apple is B-BRAND.
    - If a token could be VARIANT or DIMENSION, prefer DIMENSION for
      measured units (W, mm, ", Hz, GB-of-RAM) and VARIANT for product
      storage/trim (the phone's "128GB", "Pro Max").
    - When genuinely unsure, tag O and lower your confidence.

    PRE-FILLED SPANS: Where provided, some tokens are already tagged by
    regex/gazetteer. Trust these unless clearly wrong; adjust only if you
    have high confidence they are mislabeled.

    OUTPUT FORMAT (for a batch of N titles):
    [
      {"tags": ["O","B-BRAND",...], "confidence": 0.0-1.0, "uncertain_idx": []},
      ...
    ]
    Exactly N objects, in input order. No other keys, no markdown.\
""")


def _build_user_message(batch: list[dict]) -> str:
    """batch = list of {tokens, prefilled_tags} dicts."""
    lines = []
    for i, item in enumerate(batch):
        tokens = item["tokens"]
        prefilled = item.get("prefilled_tags", ["O"] * len(tokens))
        pre_str = json.dumps(prefilled)
        lines.append(
            f"Title {i+1} ({len(tokens)} tokens):\n"
            f"  TOKENS: {json.dumps(tokens)}\n"
            f"  PRE-FILLED: {pre_str}"
        )
    return "\n\n".join(lines)


class HaikuLabeler:
    def __init__(self, model: str = LABELER_MODEL, batch_size: int = BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._version_tag = datetime.utcnow().strftime("%Y-%m")

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if not self._client:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def label_titles(
        self,
        titles: list[dict],   # list of {id, raw, prefilled_tags?}
    ) -> list[LabelRecord]:
        """
        Label a list of titles. Each item must have 'id' and 'raw'.
        Optionally 'prefilled_tags' (list[str]) from cross_checker pre-fill.
        Returns one LabelRecord per input title, in order.
        """
        # Tokenize all inputs
        tokenized = []
        for t in titles:
            toks = tokenize(t["raw"])
            pre = t.get("prefilled_tags", ["O"] * len(toks))
            tokenized.append({
                "id": t["id"],
                "raw": t["raw"],
                "tokens": toks,
                "prefilled_tags": pre,
            })

        records: list[LabelRecord] = []
        for i in range(0, len(tokenized), self.batch_size):
            batch = tokenized[i : i + self.batch_size]
            batch_records = await self._label_batch(batch)
            records.extend(batch_records)

        return records

    async def _label_batch(self, batch: list[dict]) -> list[LabelRecord]:
        client = self._get_client()
        user_msg = _build_user_message(batch)

        try:
            resp = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = resp.content[0].text.strip()
            # Strip accidental markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:]
            parsed = json.loads(raw_text.strip())
        except Exception as e:
            print(f"[Labeler] API/parse error on batch of {len(batch)}: {e}")
            # Return placeholder records for the whole batch so the pipeline
            # routes them to the spot-check queue rather than crashing.
            return [
                LabelRecord(
                    id=item["id"],
                    raw=item["raw"],
                    tokens=item["tokens"],
                    tags=["O"] * len(item["tokens"]),
                    confidence=0.0,
                    uncertain_idx=list(range(len(item["tokens"]))),
                    prefilled_spans=[],
                    labeler_version=self._version_tag,
                    needs_review=True,
                )
                for item in batch
            ]

        if not isinstance(parsed, list) or len(parsed) != len(batch):
            print(f"[Labeler] Response length mismatch: got {len(parsed) if isinstance(parsed, list) else '?'}, expected {len(batch)}")
            parsed = parsed[:len(batch)] if isinstance(parsed, list) else []
            while len(parsed) < len(batch):
                parsed.append({"tags": [], "confidence": 0.0, "uncertain_idx": []})

        records: list[LabelRecord] = []
        for item, result in zip(batch, parsed):
            tags = result.get("tags", [])
            confidence = float(result.get("confidence", 0.0))
            uncertain_idx = [int(x) for x in result.get("uncertain_idx", [])]
            needs_review = confidence < CONFIDENCE_FLOOR or bool(uncertain_idx)

            records.append(LabelRecord(
                id=item["id"],
                raw=item["raw"],
                tokens=item["tokens"],
                tags=tags,
                confidence=confidence,
                uncertain_idx=uncertain_idx,
                prefilled_spans=item.get("prefilled_tags", []),
                labeler_version=self._version_tag,
                needs_review=needs_review,
            ))

        return records
