"""
Uses an LLM to estimate market value when eBay data is sparse or unavailable.

Backend is resolved from environment keys, in priority order:
  1. GROQ_API_KEY      — Groq (OpenAI-compatible API, default backend)
  2. ANTHROPIC_API_KEY — Anthropic Claude (fallback)
"""
import json
import os
from dataclasses import dataclass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LLMPriceEstimate:
    estimated_value_usd: float
    confidence: float           # 0.0 - 1.0
    reasoning: str
    price_range_low_usd: float
    price_range_high_usd: float
    demand_notes: str
    error: str = ""


class LLMPriceEstimator:
    def __init__(self, config: dict, usd_to_ron: float = 4.55):
        self.config = config
        self.usd_to_ron = usd_to_ron
        self._backend: str | None = None
        self._model: str | None = None
        self._client = None

    def _resolve_backend(self):
        if self._client:
            return
        groq_key = os.environ.get("GROQ_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if groq_key:
            from openai import AsyncOpenAI
            self._backend = "groq"
            self._model = self.config.get(
                "groq_model", os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            )
            self._client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_key)
        elif anthropic_key:
            import anthropic
            self._backend = "anthropic"
            self._model = self.config.get("llm_model", DEFAULT_ANTHROPIC_MODEL)
            self._client = anthropic.AsyncAnthropic(api_key=anthropic_key)
        else:
            raise ValueError(
                "GROQ_API_KEY (or ANTHROPIC_API_KEY) not set. LLM pricing unavailable."
            )

    async def _complete(self, prompt: str) -> str:
        if self._backend == "groq":
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=400,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    async def estimate(
        self,
        title: str,
        description: str = "",
        condition: str = "unknown",
        platform: str = "",
        category: str = "",
        ner_entities: dict | None = None,
    ) -> LLMPriceEstimate:
        # Build a structured item description from NER entities when available
        if ner_entities:
            entity_lines = "\n".join(
                f"  {k.upper()}: {v}" for k, v in ner_entities.items() if v
            )
            item_block = f"Raw title: {title}\nExtracted entities:\n{entity_lines}"
        else:
            item_block = f"Item: {title}"

        prompt = f"""You are a professional reseller and market pricing expert.
Estimate the current fair market resale value in USD for the following second-hand item.

{item_block}
Category: {category or "General"}
Condition: {condition}
Platform: {platform}
Description snippet: {description[:300] if description else "N/A"}

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{{
  "estimated_value_usd": <float, your best estimate of resale value>,
  "price_range_low_usd": <float, conservative low end>,
  "price_range_high_usd": <float, optimistic high end>,
  "confidence": <float 0.0-1.0, how confident you are>,
  "reasoning": "<1-2 sentence reasoning>",
  "demand_notes": "<brief note on typical demand for this item>"
}}

Be realistic — use typical eBay/market sold prices, not retail.
If you have no idea or the item is too vague, set confidence to 0.1."""

        try:
            self._resolve_backend()
            raw = await self._complete(prompt)
            # Strip markdown code block if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return LLMPriceEstimate(
                estimated_value_usd=float(data.get("estimated_value_usd", 0)),
                confidence=float(data.get("confidence", 0.1)),
                reasoning=data.get("reasoning", ""),
                price_range_low_usd=float(data.get("price_range_low_usd", 0)),
                price_range_high_usd=float(data.get("price_range_high_usd", 0)),
                demand_notes=data.get("demand_notes", ""),
            )
        except Exception as e:
            return LLMPriceEstimate(
                estimated_value_usd=0.0,
                confidence=0.0,
                reasoning="",
                price_range_low_usd=0.0,
                price_range_high_usd=0.0,
                demand_notes="",
                error=str(e),
            )

    def to_ron(self, usd_amount: float) -> float:
        return round(usd_amount * self.usd_to_ron, 2)
