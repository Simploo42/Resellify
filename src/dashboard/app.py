import asyncio
import collections
import json
import logging
import os
import tempfile
import threading
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, desc, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Listing, PriceEstimate, DealScore, ScanLog, get_db, init_db
from ..agent import load_config, run_scan, start_agent, stop_agent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Resellify", description="Market resell opportunity finder")

# ── Live log capture ───────────────────────────────────────────────────────

_log_buffer: collections.deque = collections.deque(maxlen=500)
_log_subscribers: set[asyncio.Queue] = set()


class _LiveLogHandler(logging.Handler):
    """Captures log records into the in-memory buffer and fans out to SSE subscribers."""

    LEVEL_COLOR = {
        "DEBUG": "gray",
        "INFO": "slate",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }

    _SUPPRESS = frozenset([
        "pipe closed by peer or os.write(pipe, data) raised exception",
    ])

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if any(s in msg for s in self._SUPPRESS):
            return
        entry = {
            "ts": datetime.utcnow().strftime("%H:%M:%S"),
            "level": record.levelname,
            "color": self.LEVEL_COLOR.get(record.levelname, "slate"),
            "msg": self.format(record),
        }
        _log_buffer.append(entry)
        dead: set[asyncio.Queue] = set()
        for q in list(_log_subscribers):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                dead.add(q)
        _log_subscribers.difference_update(dead)


def _install_log_handler() -> None:
    handler = _LiveLogHandler()
    handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
    handler.setLevel(logging.DEBUG)
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error", "src"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)


_install_log_handler()

_agent_task: Optional[asyncio.Task] = None
CONFIG_PATH = "config/settings.yaml"

# ── NER Spot-Check state ───────────────────────────────────────────────────

_QUEUE_PATH = Path("data/title_engine/spot_check_queue.jsonl")
_TRAIN_PATH = Path("data/title_engine/train_v0.jsonl")
_REJECTED_PATH = Path("data/title_engine/rejected_manual.jsonl")
_REVIEWED_IDS_PATH = Path("data/title_engine/.reviewed_ids")

# In-memory state (loaded lazily)
_ner_queue: Optional[list] = None          # list of dicts (unreviewed records)
_ner_reviewed_ids: Optional[set] = None    # IDs already reviewed in this or prior sessions

# Retrain state
_retrain_running = False
_retrain_log_path: Optional[str] = None


def _load_ner_state():
    """Load queue and reviewed IDs into memory if not already loaded."""
    global _ner_queue, _ner_reviewed_ids

    if _ner_reviewed_ids is None:
        if _REVIEWED_IDS_PATH.exists():
            _ner_reviewed_ids = set(_REVIEWED_IDS_PATH.read_text().splitlines())
        else:
            _ner_reviewed_ids = set()

    if _ner_queue is None:
        if _QUEUE_PATH.exists():
            all_records = [
                json.loads(line)
                for line in _QUEUE_PATH.read_text().splitlines()
                if line.strip()
            ]
            # Filter out already-reviewed records
            _ner_queue = [r for r in all_records if r["id"] not in _ner_reviewed_ids]
        else:
            _ner_queue = []


def _persist_reviewed_id(record_id: str):
    """Append an ID to the reviewed_ids file and update in-memory set."""
    global _ner_reviewed_ids
    _ner_reviewed_ids.add(record_id)
    _REVIEWED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REVIEWED_IDS_PATH, "a") as f:
        f.write(record_id + "\n")


def _rewrite_queue():
    """Rewrite the queue file from in-memory state."""
    _QUEUE_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _ner_queue) + ("\n" if _ner_queue else "")
    )


def _total_original_size() -> int:
    """Total number of records originally in the queue (reviewed + remaining)."""
    reviewed = len(_ner_reviewed_ids) if _ner_reviewed_ids is not None else 0
    remaining = len(_ner_queue) if _ner_queue is not None else 0
    return reviewed + remaining


def _next_unreviewed() -> Optional[dict]:
    """Return the first record in the queue."""
    _load_ner_state()
    return _ner_queue[0] if _ner_queue else None


class AcceptBody(BaseModel):
    id: str
    tags: list[str]


class RejectBody(BaseModel):
    id: str


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    await init_db()
    # Re-attach after uvicorn has set up its own logging
    _install_log_handler()


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



@app.delete("/api/deals/clear-all")
async def clear_all_deals(db: AsyncSession = Depends(get_db)):
    """Wipe all deals, scores, price estimates, and listings from the DB."""
    await db.execute(delete(DealScore))
    await db.execute(delete(PriceEstimate))
    await db.execute(delete(ScanLog))
    await db.execute(delete(Listing))
    await db.commit()
    return JSONResponse({"status": "cleared"})

# ── API: Live logs ─────────────────────────────────────────────────────────

@app.get("/api/logs/recent")
async def api_logs_recent():
    return JSONResponse(list(_log_buffer)[-200:])


@app.get("/api/logs/stream")
async def api_logs_stream(request: Request):
    from fastapi.responses import StreamingResponse
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _log_subscribers.add(q)

    async def event_gen():
        try:
            # Send backlog on connect
            for entry in list(_log_buffer)[-50:]:
                yield f"data: {json.dumps(entry)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # keep-alive
        finally:
            _log_subscribers.discard(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── API: Live deals stream ──────────────────────────────────────────────────

@app.get("/api/deals/stream")
async def api_deals_stream(request: Request):
    from fastapi.responses import StreamingResponse
    from ..events import _deal_buffer, _deal_subscribers
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _deal_subscribers.add(q)

    async def event_gen():
        try:
            for deal in list(_deal_buffer):
                yield f"data: {json.dumps(deal)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    deal = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(deal)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _deal_subscribers.discard(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@app.get("/api/scan/status")
async def api_scan_status():
    from ..agent import _scan_lock
    return JSONResponse({"scanning": _scan_lock.locked()})


@app.get("/api/pipeline/stream")
async def api_pipeline_stream(request: Request):
    from fastapi.responses import StreamingResponse
    from ..events import _pipeline_buffer, _pipeline_subscribers, _scan_state
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _pipeline_subscribers.add(q)

    async def event_gen():
        try:
            # Send sticky scan state first — survives buffer eviction
            if _scan_state:
                yield f"data: {json.dumps(_scan_state)}\n\n"
            # Replay buffer, skipping scan_start (already sent above)
            for ev in list(_pipeline_buffer):
                if ev.get("type") != "scan_start":
                    yield f"data: {json.dumps(ev)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _pipeline_subscribers.discard(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

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

    return templates.TemplateResponse(request, "index.html", {
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

    return templates.TemplateResponse(request, "deal_detail.html", {
        "listing": listing,
        "deal_score": deal_score,
        "estimates": estimates,
    })


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    with open(CONFIG_PATH) as f:
        config_text = f.read()
    return templates.TemplateResponse(request, "config.html", {
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


# ── NER Spot-Check routes ──────────────────────────────────────────────────

@app.get("/spot-check", response_class=HTMLResponse)
async def spot_check_page(request: Request, skip: str = ""):
    global _ner_queue
    _load_ner_state()

    # Skip: rotate the skipped record to the end of the queue (no reviewed count increment)
    if skip and _ner_queue and _ner_queue[0]["id"] == skip:
        _ner_queue.append(_ner_queue.pop(0))

    record = _next_unreviewed()
    total = _total_original_size()
    reviewed = len(_ner_reviewed_ids)
    remaining = len(_ner_queue)
    progress = {
        "reviewed": reviewed,
        "total": total,
        "remaining": remaining,
    }
    return templates.TemplateResponse(request, "spot_check.html", {
        "record": record,
        "progress": progress,
    })


@app.post("/spot-check/accept")
async def spot_check_accept(body: AcceptBody):
    global _ner_queue
    _load_ner_state()

    # Find and remove from queue
    record = next((r for r in _ner_queue if r["id"] == body.id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found in queue")

    # Apply updated tags
    record = dict(record)
    record["tags"] = body.tags
    record["needs_review"] = False

    # Append to train file
    _TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TRAIN_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Remove from in-memory queue and persist
    _ner_queue = [r for r in _ner_queue if r["id"] != body.id]
    _persist_reviewed_id(body.id)
    _rewrite_queue()

    next_record = _next_unreviewed()
    return JSONResponse({"ok": True, "next_id": next_record["id"] if next_record else None})


@app.post("/spot-check/reject")
async def spot_check_reject(body: RejectBody):
    global _ner_queue
    _load_ner_state()

    record = next((r for r in _ner_queue if r["id"] == body.id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found in queue")

    # Append to rejected file
    _REJECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REJECTED_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Remove from in-memory queue and persist
    _ner_queue = [r for r in _ner_queue if r["id"] != body.id]
    _persist_reviewed_id(body.id)
    _rewrite_queue()

    next_record = _next_unreviewed()
    return JSONResponse({"ok": True, "next_id": next_record["id"] if next_record else None})


# ── NER Retrain routes ─────────────────────────────────────────────────────

def _run_retrain_thread(log_path: str):
    global _retrain_running
    try:
        import sys
        from src.title_engine.trainer import train

        with open(log_path, "w") as log_f:
            # Redirect stdout/stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = log_f
            sys.stderr = log_f
            try:
                train(
                    Path("data/title_engine/train_v0.jsonl"),
                    Path("models/title_ner"),
                )
            except Exception as exc:
                log_f.write(f"\n[Retrain ERROR] {exc}\n")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    finally:
        _retrain_running = False


@app.post("/api/ner/retrain")
async def api_ner_retrain():
    global _retrain_running, _retrain_log_path

    if _retrain_running:
        return JSONResponse({"ok": False, "status": "already_running"})

    # Create a persistent temp file for logging
    fd, log_path = tempfile.mkstemp(prefix="ner_retrain_", suffix=".log")
    os.close(fd)
    _retrain_log_path = log_path
    _retrain_running = True

    t = threading.Thread(target=_run_retrain_thread, args=(log_path,), daemon=True)
    t.start()

    return JSONResponse({"ok": True, "status": "started"})


@app.get("/api/ner/retrain/status")
async def api_ner_retrain_status():
    global _retrain_running, _retrain_log_path

    log_lines = ""
    if _retrain_log_path and os.path.exists(_retrain_log_path):
        with open(_retrain_log_path) as f:
            lines = f.read().splitlines()
        log_lines = "\n".join(lines[-20:])

    return JSONResponse({"running": _retrain_running, "log": log_lines})
