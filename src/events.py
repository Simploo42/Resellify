"""
Shared in-process event bus for real-time deal broadcasting.
Imported by both agent.py and dashboard/app.py — no circular dependency.
"""
import asyncio
from collections import deque
from typing import Any

_deal_buffer: deque = deque(maxlen=100)
_deal_subscribers: set[asyncio.Queue] = set()


def broadcast_deal(deal: dict[str, Any]) -> None:
    """Push a new deal to all live SSE subscribers. Safe to call from any coroutine."""
    _deal_buffer.append(deal)
    dead: set[asyncio.Queue] = set()
    for q in list(_deal_subscribers):
        try:
            q.put_nowait(deal)
        except asyncio.QueueFull:
            dead.add(q)
    _deal_subscribers.difference_update(dead)

_pipeline_buffer: deque = deque(maxlen=500)
_pipeline_subscribers: set[asyncio.Queue] = set()
_scan_state: dict[str, Any] = {}   # sticky: last scan_start or scan_done; survives buffer eviction


def broadcast_pipeline(event: dict[str, Any]) -> None:
    """Push a pipeline status event to all live SSE subscribers."""
    global _scan_state
    t = event.get("type")
    if t == "scan_start":
        _scan_state = event
    elif t == "scan_done":
        _scan_state = {**event, "scanning": False}
    elif t == "listing":
        # Update progress counters; keep type=scan_start so frontend knows state
        _scan_state = {"type": "scan_start", "total": event["total"], "idx": event["idx"]}
    # listing events are live-only (too numerous); everything else including
    # scraping placeholder events are buffered for late-connecting clients
    if t != "listing":
        _pipeline_buffer.append(event)
    dead: set[asyncio.Queue] = set()
    for q in list(_pipeline_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    _pipeline_subscribers.difference_update(dead)
