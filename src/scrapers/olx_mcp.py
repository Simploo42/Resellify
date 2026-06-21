"""
OLX scraper via the olx-mcp MCP server (https://github.com/l-margiela/olx-mcp).
Spawns the Node.js MCP server as a subprocess and communicates via JSON-RPC/stdio.

Prerequisites:
  npm install -g olx-mcp
  OR ensure npx is available (npx olx-mcp will auto-install on first run)
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Optional, Any

from .base import BaseScraper, RawListing

logger = logging.getLogger(__name__)


def _parse_price(text: str) -> tuple[float, str]:
    text = str(text).strip().replace(" ", "").replace(",", ".")
    currency = "RON"
    if "€" in text or "EUR" in text:
        currency = "EUR"
    elif "$" in text or "USD" in text:
        currency = "USD"
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return float(digits), currency
    except ValueError:
        return 0.0, currency


class MCPStdioClient:
    """Minimal async client for an MCP server running over stdio."""

    def __init__(self, command: list[str]):
        self.command = command
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._msg_id = 0
        self._lock = asyncio.Lock()

    async def start(self):
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self._initialize()

    async def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send(self, msg: dict) -> None:
        data = json.dumps(msg) + "\n"
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def _recv(self) -> dict:
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=30)
            if not line:
                raise RuntimeError("MCP server closed stdout")
            text = line.decode().strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue  # skip non-JSON lines (e.g., log output)

    async def _initialize(self):
        msg_id = await self._next_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "resellify", "version": "1.0.0"},
            },
        })
        # Read until we get the response for our init request
        while True:
            resp = await self._recv()
            if resp.get("id") == msg_id:
                break

        # Send initialized notification
        await self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

    async def call_tool(self, name: str, arguments: dict) -> Any:
        async with self._lock:
            msg_id = await self._next_id()
            await self._send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })
            # Read responses until we get ours (skip notifications)
            while True:
                resp = await self._recv()
                if resp.get("id") == msg_id:
                    if "error" in resp:
                        raise RuntimeError(f"MCP tool error: {resp['error']}")
                    result = resp.get("result", {})
                    # MCP returns content as list of text/json blocks
                    content = result.get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            try:
                                return json.loads(block["text"])
                            except json.JSONDecodeError:
                                return block["text"]
                    return result

    async def close(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                await self._proc.wait()
            except Exception:
                pass
            self._proc = None


class OLXMCPScraper(BaseScraper):
    """
    Wraps the olx-mcp MCP server to search OLX Romania listings.

    Install the server first:
      npm install -g olx-mcp
    Or leave as-is to use npx (downloads on first run).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.domain = "olx.ro"  # OLX Romania (server enum: olx.pt/pl/bg/ro/ua)
        self.location = config.get("location", "")
        self.max_pages = config.get("max_pages_per_keyword", 3)
        # Optional path to a locally built server (dist/index.js); else use npx
        self.mcp_server_path = config.get("mcp_server_path", "")
        self._client: Optional[MCPStdioClient] = None

    def _build_command(self) -> list[str]:
        if self.mcp_server_path:
            return ["node", self.mcp_server_path]
        return ["npx", "--yes", "olx-mcp@latest"]

    async def _get_client(self) -> MCPStdioClient:
        if not self._client:
            self._client = MCPStdioClient(self._build_command())
            await self._client.start()
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        client = await self._get_client()
        price_range = category_config.get("price_range", {})
        min_price = price_range.get("min_ron", None)
        max_price = price_range.get("max_ron", None)
        listings: list[RawListing] = []

        for page in range(1, self.max_pages + 1):
            args: dict = {
                "domain": self.domain,
                "query": keyword,
                "page": page,
                "limit": 40,
                "sortBy": "date",
            }
            if min_price:
                args["minPrice"] = int(min_price)
            if max_price:
                args["maxPrice"] = int(max_price)
            if self.location:
                args["location"] = self.location

            try:
                result = await client.call_tool("searchListings", args)
                page_listings = self._parse_results(result, keyword, category_config)
                if not page_listings:
                    break
                listings.extend(page_listings)
                if page < self.max_pages:
                    await self._random_delay()
            except Exception as e:
                logger.info(f"[OLX-MCP] Error searching '{keyword}' page {page}: {e}")
                break

        return listings

    def _parse_results(self, result: Any, keyword: str, category_config: dict) -> list[RawListing]:
        if not isinstance(result, dict):
            return []

        items = result.get("listings", result.get("results", result.get("items", [])))
        if not isinstance(items, list):
            return []

        listings: list[RawListing] = []
        cat_name = category_config.get("name", "")

        for item in items:
            try:
                listing = self._parse_item(item, keyword, cat_name)
                if listing:
                    listings.append(listing)
            except Exception as e:
                logger.info(f"[OLX-MCP] Error parsing item: {e}")

        return listings

    def _parse_item(self, item: dict, keyword: str, category: str) -> Optional[RawListing]:
        # olx-mcp Listing schema:
        #   { id, title, price?, location?, category?, imageUrl?, url,
        #     publishedAt?, description?, seller?{name,phone,verified,memberSince} }
        external_id = str(item.get("id") or "")
        if not external_id:
            return None

        title = item.get("title") or ""
        if not title:
            return None

        url = item.get("url") or ""
        if not url:
            return None

        # Price comes back as a string like "1 200 lei" / "1.200 €" (may be absent)
        raw_price = item.get("price")
        if raw_price is None:
            price, currency = 0.0, "RON"
        elif isinstance(raw_price, (int, float)):
            price, currency = float(raw_price), "RON"
        else:
            price, currency = _parse_price(str(raw_price))

        if price <= 0:
            return None

        location = item.get("location") or ""

        # Search results carry no condition field; detail enrichment may add it later
        condition = str(item.get("condition") or "unknown").lower()

        # Single imageUrl string on search results; detail returns an images array
        images: list[str] = []
        if item.get("imageUrl"):
            images.append(item["imageUrl"])
        for img in (item.get("images") or []):
            if isinstance(img, str) and img not in images:
                images.append(img)

        seller = item.get("seller") or {}
        seller_name = seller.get("name", "") if isinstance(seller, dict) else str(seller)

        posted_at: Optional[datetime] = None
        date_raw = item.get("publishedAt")
        if date_raw:
            try:
                posted_at = datetime.fromisoformat(str(date_raw).replace("Z", "+00:00"))
            except Exception:
                pass

        return RawListing(
            external_id=external_id,
            platform="olx",
            title=title,
            price=price,
            currency=currency,
            url=url,
            location=str(location),
            condition=condition,
            images=images,
            seller_name=seller_name,
            category=category,
            matched_keyword=keyword,
            posted_at=posted_at,
        )

    async def get_listing_detail(self, url: str) -> dict:
        client = await self._get_client()
        # OLX URLs encode the id as ID<alphanumeric>.html
        id_match = re.search(r"ID([A-Za-z0-9]+)\.html", url)
        if not id_match:
            return {}
        listing_id = id_match.group(1)
        try:
            result = await client.call_tool("getListingDetails", {
                "domain": self.domain,
                "listingId": listing_id,
                "includeImages": True,
                "includeSellerInfo": True,
            })
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.info(f"[OLX-MCP] Error getting details for {url}: {e}")
        return {}
