from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.database import (
    awb_exists,
    create_card,
    create_zone,
    delete_card,
    delete_zone,
    get_cards,
    get_zones,
    toggle_called,
)
from src.ocr_processor import process_image

router = APIRouter(prefix="/api")


# ── Zones ─────────────────────────────────────────────────────────────────────

class ZoneIn(BaseModel):
    name: str
    keyword: str


@router.get("/zones")
async def list_zones():
    return await get_zones()


@router.post("/zones", status_code=201)
async def add_zone(body: ZoneIn):
    if not body.name.strip() or not body.keyword.strip():
        raise HTTPException(400, "name and keyword are required")
    return await create_zone(body.name.strip(), body.keyword.strip())


@router.delete("/zones/{zone_id}")
async def remove_zone(zone_id: int):
    await delete_zone(zone_id)
    return {"ok": True}


# ── Cards ─────────────────────────────────────────────────────────────────────

class CardIn(BaseModel):
    awb: str
    name: str
    address: str
    phone: str


class CardsAddIn(BaseModel):
    cards: list[CardIn]


@router.get("/cards")
async def list_cards():
    zones = await get_zones()
    cards = await get_cards()
    for card in cards:
        card["called"] = bool(card["called"])
        card["zones"] = [
            z["name"]
            for z in zones
            if z["keyword"].lower() in card["address"].lower()
        ]
    return cards


@router.post("/cards", status_code=201)
async def add_cards(body: CardsAddIn):
    added, skipped = [], []
    for c in body.cards:
        result = await create_card(c.awb, c.name, c.address, c.phone)
        if result:
            added.append(result)
        else:
            skipped.append(c.awb)
    return {"added": len(added), "skipped": skipped, "cards": added}


@router.patch("/cards/{card_id}/called")
async def update_called(card_id: int):
    status = await toggle_called(card_id)
    if status is None:
        raise HTTPException(404, "Card not found")
    return {"called": status}


@router.delete("/cards/{card_id}")
async def remove_card(card_id: int):
    await delete_card(card_id)
    return {"ok": True}


# ── OCR Scan ──────────────────────────────────────────────────────────────────

@router.post("/scan")
async def scan(file: UploadFile = File(...)):
    data = await file.read()
    try:
        cards = process_image(data)
    except Exception as exc:
        raise HTTPException(500, f"OCR failed: {exc}") from exc

    for card in cards:
        card["exists"] = await awb_exists(card["awb"])

    return {"cards": cards, "total": len(cards)}
