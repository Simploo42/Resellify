"""
Uses Claude to estimate market value when eBay data is sparse or unavailable.
"""
import json
import os
from dataclasses import dataclass

import anthropic


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
        self.model = config.get("llm_model", "claude-haiku-4-5-20251001")
        self.usd_to_ron = usd_to_ron
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if not self._client:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set. LLM pricing unavailable.")
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def estimate(
        self,
        title: str,
        description: str = "",
        condition: str = "unknown",
        platform: str = "",
        category: str = "",
    ) -> LLMPriceEstimate:
        client = self._get_client()

        prompt = f"""You are a professional reseller and market pricing expert.
Estimate the current fair market resale value in USD for the following second-hand item.

Item: {title}
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
            response = await client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
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
