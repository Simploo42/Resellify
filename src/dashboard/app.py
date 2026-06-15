import asyncio
import os
import yaml
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, desc, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Listing, PriceEstimate, DealScore, ScanLog, get_db, init_db
from ..agent import load_config, run_scan, start_agent, stop_agent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Resellify", description="Market resell opportunity finder")

_agent_task: Optional[asyncio.Task] = None
CONFIG_PATH = "config/settings.yaml"


@app.on_event("startup")
async def startup():
    await init_db()


def _grade_color(grade: str) -> str:
    return {"S": "#10b981", "A": "#22c55e", "B": "#eab308", "C": "#f97316", "D": "#ef4444"}.get(grade, "#6b7280")


def _score_color(score: float) -> str:
    if score >= 80:
        return "emerald"
    if score >= 65:
        return "green"
    if score >= 50:
        return "yellow"
    if score >= 35:
        return "orange"
    return "red"


templates.env.globals["grade_color"] = _grade_color
templates.env.globals["score_color"] = _score_color


# ── API: Agent control ─────────────────────────────────────────────────────

@app.post("/api/agent/start")
async def api_start_agent():
    global _agent_task
    if _agent_task and not _agent_task.done():
        return JSONResponse({"status": "already_running"})
    _agent_task = asyncio.create_task(start_agent(CONFIG_PATH))
    return JSONResponse({"status": "started"})


@app.post("/api/agent/stop")
async def api_stop_agent():
    global _agent_task
    stop_agent()
    if _agent_task:
        _agent_task.cancel()
    return JSONResponse({"status": "stopped"})


@app.post("/api/agent/scan-now")
async def api_scan_now():
    config = load_config(CONFIG_PATH)
    asyncio.create_task(run_scan(config))
    return JSONResponse({"status": "scan_started"})


@app.get("/api/agent/status")
async def api_agent_status():
    global _agent_task
    running = bool(_agent_task and not _agent_task.done())
    return JSONResponse({"running": running})


# ── Dashboard: Main ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
    platform: str = "",
    category: str = "",
    min_score: float = 0,
    min_profit: float = 0,
    sort: str = "score",
    page: int = 1,
):
    page_size = 20
    offset = (page - 1) * page_size

    # Build deal query joining listings + deal_scores
    q = (
        select(Listing, DealScore)
        .join(DealScore, Listing.id == DealScore.listing_id)
        .where(Listing.is_active == True)
        .where(DealScore.total_score >= min_score)
    )
    if platform:
        q = q.where(Listing.platform == platform)
    if category:
        q = q.where(Listing.category == category)
    if min_profit > 0:
        q = q.where(DealScore.profit_percent >= min_profit)

    sort_col = {
        "score": DealScore.total_score,
        "profit": DealScore.profit_percent,
        "profit_ron": DealScore.estimated_profit_ron,
        "newest": Listing.scraped_at,
    }.get(sort, DealScore.total_score)
    q = q.order_by(desc(sort_col))

    total_q = q
    count_result = await db.execute(select(func.count()).select_from(total_q.subquery()))
    total_count = count_result.scalar() or 0

    q = q.offset(offset).limit(page_size)
    results = await db.execute(q)
    rows = results.all()

    deals = []
    for listing, score in rows:
        deals.append({
            "listing": listing,
            "score": score,
            "grade": score.notes,
        })

    # Stats
    stats_q = await db.execute(
        select(
            func.count(DealScore.id),
            func.avg(DealScore.profit_percent),
            func.avg(DealScore.total_score),
        ).where(DealScore.total_score >= 55)
    )
    stats_row = stats_q.first()
    stats = {
        "total_deals": stats_row[0] or 0,
        "avg_profit_pct": round(stats_row[1] or 0, 1),
        "avg_score": round(stats_row[2] or 0, 1),
    }

    # Recent scan log
    scan_q = await db.execute(
        select(ScanLog).order_by(desc(ScanLog.started_at)).limit(5)
    )
    recent_scans = scan_q.scalars().all()

    # Categories for filter
    cat_q = await db.execute(
        select(Listing.category).distinct().where(Listing.category != "")
    )
    categories = [r[0] for r in cat_q.all()]

    agent_running = bool(_agent_task and not _agent_task.done())

    return templates.TemplateResponse("index.html", {
        "request": request,
        "deals": deals,
        "stats": stats,
        "recent_scans": recent_scans,
        "categories": categories,
        "agent_running": agent_running,
        "filters": {
            "platform": platform,
            "category": category,
            "min_score": min_score,
            "min_profit": min_profit,
            "sort": sort,
        },
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total_count,
            "total_pages": max(1, (total_count + page_size - 1) // page_size),
        },
    })


@app.get("/deal/{listing_id:path}", response_class=HTMLResponse)
async def deal_detail(request: Request, listing_id: str, db: AsyncSession = Depends(get_db)):
    listing_result = await db.execute(select(Listing).where(Listing.id == listing_id))
    listing = listing_result.scalar_one_or_none()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")

    score_result = await db.execute(select(DealScore).where(DealScore.listing_id == listing_id))
    deal_score = score_result.scalar_one_or_none()

    estimates_result = await db.execute(
        select(PriceEstimate).where(PriceEstimate.listing_id == listing_id)
    )
    estimates = estimates_result.scalars().all()

    return templates.TemplateResponse("deal_detail.html", {
        "request": request,
        "listing": listing,
        "deal_score": deal_score,
        "estimates": estimates,
    })


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    with open(CONFIG_PATH) as f:
        config_text = f.read()
    return templates.TemplateResponse("config.html", {
        "request": request,
        "config_text": config_text,
    })


@app.post("/config")
async def save_config(request: Request, config_text: str = Form(...)):
    try:
        yaml.safe_load(config_text)  # validate
    except yaml.YAMLError as e:
        return templates.TemplateResponse("config.html", {
            "request": request,
            "config_text": config_text,
            "error": f"Invalid YAML: {e}",
        })
    with open(CONFIG_PATH, "w") as f:
        f.write(config_text)
    return RedirectResponse(url="/config?saved=1", status_code=303)


@app.delete("/api/deal/{listing_id:path}")
async def delete_deal(listing_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(DealScore).where(DealScore.listing_id == listing_id))
    await db.execute(delete(PriceEstimate).where(PriceEstimate.listing_id == listing_id))
    await db.execute(delete(Listing).where(Listing.id == listing_id))
    await db.commit()
    return JSONResponse({"status": "deleted"})


@app.get("/api/deals/json")
async def deals_json(
    db: AsyncSession = Depends(get_db),
    min_score: float = 55,
    limit: int = 50,
):
    q = (
        select(Listing, DealScore)
        .join(DealScore, Listing.id == DealScore.listing_id)
        .where(Listing.is_active == True)
        .where(DealScore.total_score >= min_score)
        .order_by(desc(DealScore.total_score))
        .limit(limit)
    )
    results = await db.execute(q)
    deals = []
    for listing, score in results.all():
        deals.append({
            "id": listing.id,
            "title": listing.title,
            "platform": listing.platform,
            "price_ron": listing.price_ron,
            "estimated_value_ron": score.estimated_value_ron,
            "profit_ron": score.estimated_profit_ron,
            "profit_pct": score.profit_percent,
            "score": score.total_score,
            "grade": _grade(score.total_score),
            "url": listing.url,
            "scraped_at": listing.scraped_at.isoformat() if listing.scraped_at else None,
        })
    return JSONResponse(deals)


def _grade(score: float) -> str:
    if score >= 85:
        return "S"
    if score >= 72:
        return "A"
    if score >= 58:
        return "B"
    if score >= 45:
        return "C"
    return "D"
