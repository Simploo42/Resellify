import os
import aiosqlite

DB_PATH = os.getenv("DB_PATH", "courier.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    keyword TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    awb        TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL,
    address    TEXT NOT NULL,
    phone      TEXT NOT NULL,
    called     INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


# ── Zones ────────────────────────────────────────────────────────────────────

async def get_zones() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM zones ORDER BY name") as cur:
            return [dict(r) for r in await cur.fetchall()]


async def create_zone(name: str, keyword: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO zones (name, keyword) VALUES (?, ?)", (name, keyword)
        )
        await db.commit()
        return {"id": cur.lastrowid, "name": name, "keyword": keyword}


async def delete_zone(zone_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
        await db.commit()


# ── Cards ─────────────────────────────────────────────────────────────────────

async def get_cards() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM cards ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def awb_exists(awb: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM cards WHERE awb = ?", (awb,)) as cur:
            return await cur.fetchone() is not None


async def create_card(awb: str, name: str, address: str, phone: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            cur = await db.execute(
                "INSERT INTO cards (awb, name, address, phone) VALUES (?, ?, ?, ?)",
                (awb, name, address, phone),
            )
            await db.commit()
            return {
                "id": cur.lastrowid,
                "awb": awb,
                "name": name,
                "address": address,
                "phone": phone,
                "called": False,
            }
        except aiosqlite.IntegrityError:
            return None


async def toggle_called(card_id: int) -> bool | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT called FROM cards WHERE id = ?", (card_id,)
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            new_val = 0 if row[0] else 1
        await db.execute(
            "UPDATE cards SET called = ? WHERE id = ?", (new_val, card_id)
        )
        await db.commit()
        return bool(new_val)


async def delete_card(card_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        await db.commit()
