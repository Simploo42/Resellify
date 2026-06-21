# Labeler error patterns

Running list of disambiguation corrections fed back into the prompt's
DISAMBIGUATION section (spec §4). Add an entry each time a systematic
labeling mistake is caught in spot-check.

Format: `## YYYY-MM-DD — <brief description>` + correction rule.

---

## Seed entries

- "orange" as a colour (Romanian operator brand vs colour): if the title
  contains a phone/product context, "orange" → B-COLOR. If it appears
  alone or next to "abonament / retea / SIM", it is a BRAND (mobile network).

- "apple" as fruit vs brand: always B-BRAND when adjacent to a model token
  (iphone, ipad, macbook, watch). Standalone "apple" in a food/produce title → O.

- "samsung" storage drives: when title contains "SSD" or "HDD" or "T7/T5",
  Samsung is still B-BRAND.

- Condition spans: "ca nou" is a 2-token CONDITION span → B-CONDITION I-CONDITION.
  "pentru piese" is a 2-token CONDITION span → B-CONDITION I-CONDITION.

- "Pro Max" vs "Pro" + "Max": always a single VARIANT span (B-VARIANT I-VARIANT).
  Never split into two separate spans.

- Numbers before brand names: "2x iPhone" — "2x" → B-QUANTITY, "iphone" → B-MODEL.

---

## 2026-06-21 — iphone/ipad/macbook always MODEL, never BRAND

Systematic error: LLM tagged "iphone" as B-BRAND in ~35 % of occurrences (7/20
sampled records). "iphone", "ipad", "macbook", "airpods", "imac", "ipod", "iwatch"
are Apple product-line names — they are B-MODEL tokens.

Correction rule added to DISAMBIGUATION:
  "iphone 13" (no apple present) → B-MODEL I-MODEL
  "apple iphone 13" → B-BRAND B-MODEL I-MODEL
  NEVER tag iphone/ipad/macbook/airpods as B-BRAND.

Also fixed: tokenizer `_DIGIT_HYPHEN_ALPHA_RE` now splits "P5-iPhone" →
["p5", "iphone"] so both tokens are individually labelable.
