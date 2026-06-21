from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, JSON, Index,
    create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./resellify.db"
SYNC_DATABASE_URL = "sqlite:///./resellify.db"


class Base(DeclarativeBase):
    pass


class Listing(Base):
    __tablename__ = "listings"

    id = Column(String, primary_key=True)           # platform:listing_id
    platform = Column(String, nullable=False)        # olx / facebook / ebay
    external_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    price = Column(Float, nullable=False)
    currency = Column(String, default="RON")
    price_ron = Column(Float, nullable=True)
    condition = Column(String, default="unknown")    # nou / ca nou / folosit / deteriorat
    location = Column(String, default="")
    url = Column(String, nullable=False)
    images = Column(JSON, default=list)
    seller_name = Column(String, default="")
    seller_url = Column(String, default="")
    category = Column(String, default="")
    matched_keyword = Column(String, default="")
    posted_at = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    # Price-change tracking
    first_price_ron = Column(Float, nullable=True)   # price when first seen
    price_drop_ron = Column(Float, default=0.0)      # total drop since first seen
    price_drop_pct = Column(Float, default=0.0)      # % drop since first seen
    price_changes = Column(Integer, default=0)       # number of observed changes
    # NER entities extracted from the title
    ner_entities = Column(JSON, default=dict)        # {"brand": "Samsung", "model": "Galaxy S24", ...}

    __table_args__ = (
        Index("ix_listings_platform", "platform"),
        Index("ix_listings_scraped_at", "scraped_at"),
        Index("ix_listings_is_active", "is_active"),
    )


class PriceChange(Base):
    __tablename__ = "price_changes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    listing_id = Column(String, nullable=False)
    old_price_ron = Column(Float, nullable=False)
    new_price_ron = Column(Float, nullable=False)
    delta_ron = Column(Float, nullable=False)        # new - old (negative = drop)
    delta_pct = Column(Float, nullable=False)
    changed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_price_changes_listing_id", "listing_id"),
        Index("ix_price_changes_changed_at", "changed_at"),
    )


class PriceEstimate(Base):
    __tablename__ = "price_estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    listing_id = Column(String, nullable=False)
    method = Column(String, nullable=False)          # ebay_sold / llm / combined
    estimated_value_ron = Column(Float, nullable=False)
    confidence = Column(Float, default=0.0)          # 0.0 - 1.0
    sold_count = Column(Integer, default=0)
    avg_price_ron = Column(Float, nullable=True)
    min_price_ron = Column(Float, nullable=True)
    max_price_ron = Column(Float, nullable=True)
    avg_days_to_sell = Column(Float, nullable=True)
    sell_through_rate = Column(Float, nullable=True)  # sold / listed ratio
    llm_reasoning = Column(Text, default="")
    raw_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_price_estimates_listing_id", "listing_id"),
    )


class DealScore(Base):
    __tablename__ = "deal_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    listing_id = Column(String, nullable=False, unique=True)
    total_score = Column(Float, nullable=False)      # 0 - 100
    profit_score = Column(Float, default=0.0)
    demand_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)          # lower = higher risk
    asking_price_ron = Column(Float, nullable=False)
    estimated_value_ron = Column(Float, nullable=False)
    estimated_profit_ron = Column(Float, nullable=False)
    profit_percent = Column(Float, nullable=False)
    is_notified = Column(Boolean, default=False)
    notes = Column(JSON, default=list)               # human-readable scoring notes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_deal_scores_total_score", "total_score"),
        Index("ix_deal_scores_created_at", "created_at"),
        Index("ix_deal_scores_profit_percent", "profit_percent"),
        Index("ix_deal_scores_estimated_profit_ron", "estimated_profit_ron"),
    )


class ScanLog(Base):
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String, nullable=False)
    keyword = Column(String, nullable=False)
    category = Column(String, default="")
    listings_found = Column(Integer, default=0)
    new_listings = Column(Integer, default=0)
    deals_scored = Column(Integer, default=0)
    errors = Column(JSON, default=list)
    duration_seconds = Column(Float, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_scan_logs_started_at", "started_at"),
        Index("ix_scan_logs_platform", "platform"),
    )


async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    poolclass=NullPool,          # each session gets its own connection — no shared state
    connect_args={"timeout": 30},
)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@event.listens_for(async_engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record):
    """Applied to every new SQLite connection — guarantees WAL + generous busy timeout."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")   # 30 s — outlasts any realistic write
    cur.execute("PRAGMA synchronous=NORMAL")   # safe with WAL, faster than FULL
    cur.close()


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # WAL mode: concurrent reads, serialised writes, no busy-lock on SELECT
        await conn.execute(__import__("sqlalchemy").text("PRAGMA journal_mode=WAL"))
        await conn.execute(__import__("sqlalchemy").text("PRAGMA busy_timeout=10000"))
        # Add ner_entities column to existing databases that predate it
        try:
            await conn.execute(
                __import__("sqlalchemy").text(
                    "ALTER TABLE listings ADD COLUMN ner_entities JSON DEFAULT '{}'"
                )
            )
        except Exception:
            pass  # column already exists

        # Ensure performance indexes exist on databases created before they were
        # added (create_all does not retrofit indexes onto existing tables).
        _index_ddl = (
            "CREATE INDEX IF NOT EXISTS ix_deal_scores_profit_percent "
            "ON deal_scores (profit_percent)",
            "CREATE INDEX IF NOT EXISTS ix_deal_scores_estimated_profit_ron "
            "ON deal_scores (estimated_profit_ron)",
            "CREATE INDEX IF NOT EXISTS ix_listings_category ON listings (category)",
        )
        _text = __import__("sqlalchemy").text
        for _ddl in _index_ddl:
            try:
                await conn.execute(_text(_ddl))
            except Exception:
                pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
