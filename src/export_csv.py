"""
Deals CSV export — flat, classifier-friendly rows.

One row per scored deal (listing joined with its deal score), with NER
entities flattened into columns and the latest price estimate attached.
Shared by the dashboard endpoint (/api/deals/csv) and the CLI
(python main.py --export-csv deals.csv).
"""
import csv
import io
import json

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import Listing, DealScore, PriceEstimate

# NER labels emitted as dedicated columns (see title_engine tokenizer schema)
NER_LABELS = [
    "brand", "model", "variant", "dimension", "color",
    "material", "condition", "identifier", "accessory", "quantity",
]

CSV_COLUMNS = [
    # Identity
    "id", "platform", "category", "matched_keyword",
    # Listing features
    "title", "condition", "location", "seller_name",
    "price_ron", "first_price_ron", "price_drop_ron", "price_drop_pct", "price_changes",
    # NER entities (flattened from listings.ner_entities)
    *[f"ner_{label}" for label in NER_LABELS],
    "ner_entities_json",
    # Valuation
    "estimated_value_ron", "estimated_profit_ron", "profit_percent",
    # Latest price estimate detail
    "estimate_method", "estimate_confidence", "sold_count",
    "avg_days_to_sell", "sell_through_rate",
    # Deal score
    "total_score", "profit_score", "demand_score", "confidence_score", "risk_score",
    "grade",
    # Meta
    "is_active", "posted_at", "scraped_at", "url",
]


def grade(score: float) -> str:
    if score >= 85:
        return "S"
    if score >= 72:
        return "A"
    if score >= 58:
        return "B"
    if score >= 45:
        return "C"
    return "D"


def _iso(dt) -> str:
    return dt.isoformat() if dt else ""


def deal_row(listing: Listing, score: DealScore, estimate: PriceEstimate | None) -> dict:
    entities = listing.ner_entities or {}
    row = {
        "id": listing.id,
        "platform": listing.platform,
        "category": listing.category,
        "matched_keyword": listing.matched_keyword,
        "title": listing.title,
        "condition": listing.condition,
        "location": listing.location,
        "seller_name": listing.seller_name,
        "price_ron": listing.price_ron,
        "first_price_ron": listing.first_price_ron,
        "price_drop_ron": listing.price_drop_ron,
        "price_drop_pct": listing.price_drop_pct,
        "price_changes": listing.price_changes,
        "ner_entities_json": json.dumps(entities, ensure_ascii=False) if entities else "",
        "estimated_value_ron": score.estimated_value_ron,
        "estimated_profit_ron": score.estimated_profit_ron,
        "profit_percent": score.profit_percent,
        "estimate_method": estimate.method if estimate else "",
        "estimate_confidence": estimate.confidence if estimate else "",
        "sold_count": estimate.sold_count if estimate else "",
        "avg_days_to_sell": estimate.avg_days_to_sell if estimate else "",
        "sell_through_rate": estimate.sell_through_rate if estimate else "",
        "total_score": score.total_score,
        "profit_score": score.profit_score,
        "demand_score": score.demand_score,
        "confidence_score": score.confidence_score,
        "risk_score": score.risk_score,
        "grade": grade(score.total_score),
        "is_active": int(bool(listing.is_active)),
        "posted_at": _iso(listing.posted_at),
        "scraped_at": _iso(listing.scraped_at),
        "url": listing.url,
    }
    for label in NER_LABELS:
        row[f"ner_{label}"] = entities.get(label, "")
    return row


async def fetch_deal_rows(
    db: AsyncSession,
    min_score: float = 0,
    platform: str = "",
    include_inactive: bool = False,
) -> list[dict]:
    q = (
        select(Listing, DealScore)
        .join(DealScore, Listing.id == DealScore.listing_id)
        .where(DealScore.total_score >= min_score)
        .order_by(desc(DealScore.total_score))
    )
    if not include_inactive:
        q = q.where(Listing.is_active == True)  # noqa: E712
    if platform:
        q = q.where(Listing.platform == platform)
    results = (await db.execute(q)).all()
    if not results:
        return []

    # Latest price estimate per listing, fetched in one query
    listing_ids = [listing.id for listing, _ in results]
    est_q = (
        select(PriceEstimate)
        .where(PriceEstimate.listing_id.in_(listing_ids))
        .order_by(PriceEstimate.created_at)
    )
    latest_estimates: dict[str, PriceEstimate] = {}
    for est in (await db.execute(est_q)).scalars():
        latest_estimates[est.listing_id] = est  # later rows overwrite earlier

    return [
        deal_row(listing, score, latest_estimates.get(listing.id))
        for listing, score in results
    ]


def rows_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
