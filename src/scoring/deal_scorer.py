"""
Deal scoring engine. Produces a 0-100 score for each listing based on:
  - Profit potential (40%)
  - Market demand (35%)
  - Pricing confidence (15%)
  - Risk factors (10%)
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreResult:
    total_score: float
    profit_score: float
    demand_score: float
    confidence_score: float
    risk_score: float           # lower = higher risk, used inverted in total
    asking_price_ron: float
    estimated_value_ron: float
    estimated_profit_ron: float
    profit_percent: float
    grade: str                  # S / A / B / C / D
    notes: list[str] = field(default_factory=list)

    @property
    def is_good_deal(self) -> bool:
        return self.total_score >= 55


CONDITION_RISK = {
    "nou": 0.0,
    "ca nou": 5.0,
    "folosit": 15.0,
    "deteriorat": 40.0,
    "new": 0.0,
    "like new": 5.0,
    "very good": 8.0,
    "good": 15.0,
    "acceptable": 30.0,
    "for parts": 60.0,
    "unknown": 20.0,
}


def grade_for(score: float) -> str:
    """Map a 0-100 deal score to a letter grade. Single source of truth
    shared by the scorer and the dashboard so they never diverge."""
    if score >= 85:
        return "S"
    if score >= 72:
        return "A"
    if score >= 58:
        return "B"
    if score >= 45:
        return "C"
    return "D"


# Backwards-compatible alias.
_grade = grade_for


class DealScorer:
    def __init__(self, config: dict):
        weights = config.get("weights", {})
        self.w_profit = weights.get("profit", 0.40)
        self.w_demand = weights.get("demand", 0.35)
        self.w_confidence = weights.get("confidence", 0.15)
        self.w_risk = weights.get("risk", 0.10)
        self.min_profit_pct = config.get("min_profit_percent", 15.0)
        self.min_profit_ron = config.get("min_profit_ron", 50.0)
        self.min_score = config.get("min_deal_score", 55.0)

    def score(
        self,
        asking_price_ron: float,
        estimated_value_ron: float,
        demand_score: float,           # 0-100, from EbaySoldPricer
        confidence: float,             # 0-1, from price estimate
        condition: str = "unknown",
        sold_count: int = 0,
        recent_sold_30d: int = 0,
        platform: str = "",
        extra_risk: float = 0.0,       # optional extra risk penalty
    ) -> ScoreResult:
        notes: list[str] = []

        # ── Profit Score ───────────────────────────────────────────────────────
        if estimated_value_ron <= 0 or asking_price_ron <= 0:
            profit_percent = 0.0
        else:
            profit_percent = (estimated_value_ron - asking_price_ron) / estimated_value_ron * 100

        estimated_profit_ron = estimated_value_ron - asking_price_ron

        if profit_percent >= 60:
            profit_score = 100.0
            notes.append(f"Exceptional margin: {profit_percent:.0f}% potential profit")
        elif profit_percent >= 40:
            profit_score = 85.0 + (profit_percent - 40) * 0.75
            notes.append(f"Great margin: {profit_percent:.0f}%")
        elif profit_percent >= 25:
            profit_score = 65.0 + (profit_percent - 25) * 1.33
            notes.append(f"Good margin: {profit_percent:.0f}%")
        elif profit_percent >= self.min_profit_pct:
            profit_score = 30.0 + (profit_percent - self.min_profit_pct) * (35.0 / max(1, 25 - self.min_profit_pct))
            notes.append(f"Acceptable margin: {profit_percent:.0f}%")
        elif profit_percent > 0:
            profit_score = max(0.0, profit_percent * 2.0)
            notes.append(f"Low margin: {profit_percent:.0f}%")
        else:
            profit_score = 0.0
            notes.append(f"No profit or overpriced (margin: {profit_percent:.0f}%)")

        # ── Demand Score ───────────────────────────────────────────────────────
        # demand_score may come from OLX market depth (sold_count = active listings)
        # or eBay sold data (recent_sold_30d = actual sales). Detect which.
        _market_label = (
            f"{recent_sold_30d} sold last 30d" if recent_sold_30d > 0
            else f"{sold_count} similar OLX listings" if sold_count > 0
            else "no market data"
        )
        if demand_score >= 70:
            notes.append(f"High market presence ({_market_label})")
        elif demand_score >= 40:
            notes.append(f"Moderate market presence ({_market_label})")
        elif demand_score >= 15:
            notes.append(f"Low market presence ({_market_label})")
        else:
            notes.append("Very thin market — verify demand before buying")

        # ── Confidence Score ───────────────────────────────────────────────────
        confidence_score = confidence * 100
        if confidence < 0.4:
            notes.append("Low price confidence — verify before buying")
        elif confidence < 0.7:
            notes.append("Moderate price confidence")

        # ── Risk Score ────────────────────────────────────────────────────────
        condition_risk = CONDITION_RISK.get(condition.lower(), 20.0)
        risk_score = min(100.0, condition_risk + extra_risk)

        if condition.lower() in ("deteriorat", "for parts"):
            notes.append("Poor condition — high resell risk")
        elif condition.lower() in ("nou", "ca nou", "new", "like new"):
            notes.append("Good condition — low risk")

        # Platform-specific risk adjustments
        if platform == "facebook":
            risk_score = min(100.0, risk_score + 5.0)  # slightly higher risk (no buyer protection)
            notes.append("Facebook: inspect before buying (no buyer protection)")

        # ── Total Score ────────────────────────────────────────────────────────
        risk_factor = (100.0 - risk_score)  # invert: high risk → low contribution
        total = (
            self.w_profit * profit_score
            + self.w_demand * demand_score
            + self.w_confidence * confidence_score
            + self.w_risk * risk_factor
        )
        total = max(0.0, min(100.0, total))

        # ── Liquidity gate ──────────────────────────────────────────────────
        # No real market signal (no comparable listings and no recorded sales)
        # means we cannot vouch for resale. Cap the score so a fat paper margin
        # on an illiquid/unknown item can never present as a strong deal.
        has_market_signal = sold_count > 0 or recent_sold_30d > 0 or demand_score > 0
        if not has_market_signal:
            total = min(total, 45.0)
            notes.append("No comparable market \u2014 demand unverified, score capped")

        return ScoreResult(
            total_score=round(total, 1),
            profit_score=round(profit_score, 1),
            demand_score=round(demand_score, 1),
            confidence_score=round(confidence_score, 1),
            risk_score=round(risk_score, 1),
            asking_price_ron=asking_price_ron,
            estimated_value_ron=estimated_value_ron,
            estimated_profit_ron=round(estimated_profit_ron, 2),
            profit_percent=round(profit_percent, 1),
            grade=_grade(total),
            notes=notes,
        )
