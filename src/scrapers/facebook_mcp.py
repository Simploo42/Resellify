"""
Facebook Marketplace scraper via facebook-marketplace-mcp MCP server.
https://github.com/jdcodes1/facebook-marketplace-mcp

Uses Facebook's GraphQL API with Chrome session cookies — no browser automation.

Prerequisites:
  git clone https://github.com/jdcodes1/facebook-marketplace-mcp
  cd facebook-marketplace-mcp && npm install && npm run build

Note: Cookie extraction is macOS-only (Chrome Keychain).
      On Linux, use the Playwright-based scraper in facebook.py instead.

Set in config:
  markets.facebook.mcp_server_path: "/path/to/facebook-marketplace-mcp/dist/index.js"
  markets.facebook.chrome_profile: "Default"
"""
import asyncio
import json
import re
from datetime import datetime
from typing import Optional, Any

from .base import BaseScraper, RawListing

# Romanian city coordinates (lat, lon)
ROMANIA_CITY_COORDS: dict[str, tuple[float, float]] = {
    "bucharest": (44.4268, 26.1025),
    "bucuresti": (44.4268, 26.1025),
    "cluj-napoca": (46.7712, 23.6236),
    "timisoara": (45.7489, 21.2087),
    "iasi": (47.1585, 27.6014),
    "constanta": (44.1598, 28.6348),
    "brasov": (45.6427, 25.5887),
    "galati": (45.4353, 28.0080),
    "ploiesti": (44.9436, 26.0140),
    "craiova": (44.3302, 23.7949),
}


class MCPStdioClient:
    """Async MCP client over stdio — minimal JSON-RPC 2.0 implementation."""

    def __init__(self, command: list[str], env: dict[str, str] | None = None):
        self.command = command
        self.env = env or {}
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._msg_id = 0
        self._lock = asyncio.Lock()

    async def start(self):
        import os
        full_env = {**os.environ, **self.env}
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
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
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=45)
            if not line:
                raise RuntimeError("MCP server closed stdout")
            text = line.decode().strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue

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
        while True:
            resp = await self._recv()
            if resp.get("id") == msg_id:
                break
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def call_tool(self, name: str, arguments: dict) -> Any:
        async with self._lock:
            msg_id = await self._next_id()
            await self._send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })
            while True:
                resp = await self._recv()
                if resp.get("id") == msg_id:
                    if "error" in resp:
                        raise RuntimeError(f"MCP error: {resp['error']}")
                    result = resp.get("result", {})
                    for block in result.get("content", []):
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
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None


class FacebookMCPScraper(BaseScraper):
    """
    Facebook Marketplace via GraphQL MCP server.
    Falls back gracefully if the MCP server is not available.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.mcp_server_path = config.get("mcp_server_path", "")
        self.chrome_profile = config.get("chrome_profile", "Default")
        self.radius_km = config.get("radius_km", 50)
        self.max_results = config.get("max_results_per_keyword", 40)
        # Resolve city coordinates from location string
        location_raw = config.get("location", "bucuresti").lower()
        city_key = next(
            (k for k in ROMANIA_CITY_COORDS if k in location_raw),
            "bucuresti"
        )
        self.lat, self.lon = ROMANIA_CITY_COORDS[city_key]
        self._client: Optional[MCPStdioClient] = None

    def _build_command(self) -> list[str]:
        if self.mcp_server_path:
            return ["node", self.mcp_server_path]
        # Try to find it globally
        return ["npx", "--yes", "facebook-marketplace-mcp"]

    async def _get_client(self) -> MCPStdioClient:
        if not self._client:
            cmd = self._build_command()
            self._client = MCPStdioClient(
                command=cmd,
                env={"CHROME_PROFILE": self.chrome_profile},
            )
            await self._client.start()
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        try:
            client = await self._get_client()
        except Exception as e:
            print(f"[FB-MCP] Cannot start MCP server: {e}")
            return []

        price_range = category_config.get("price_range", {})
        args: dict = {
            "query": keyword,
            "latitude": self.lat,
            "longitude": self.lon,
            "radius_km": self.radius_km,
            "limit": self.max_results,
        }
        min_p = price_range.get("min_ron")
        max_p = price_range.get("max_ron")
        if min_p:
            args["min_price"] = int(min_p)
        if max_p:
            args["max_price"] = int(max_p)

        try:
            # Rate limit: server allows 3 req/min
            await self._random_delay(20)
            result = await client.call_tool("search_listings", args)
            return self._parse_results(result, keyword, category_config.get("name", ""))
        except Exception as e:
            print(f"[FB-MCP] Search error for '{keyword}': {e}")
            return []

    def _parse_results(self, result: Any, keyword: str, category: str) -> list[RawListing]:
        if not result:
            return []

        items: list[dict] = []
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict):
            items = (
                result.get("listings")
                or result.get("results")
                or result.get("edges")
                or result.get("nodes")
                or []
            )

        listings: list[RawListing] = []
        for item in items:
            try:
                parsed = self._parse_item(item, keyword, category)
                if parsed:
                    listings.append(parsed)
            except Exception as e:
                print(f"[FB-MCP] Parse error: {e}")

        return listings

    def _parse_item(self, item: dict, keyword: str, category: str) -> Optional[RawListing]:
        # Handle nested GraphQL node structure
        node = item.get("node", item)

        external_id = str(
            node.get("id")
            or node.get("listing_id")
            or node.get("marketplace_listing_id")
            or ""
        )
        if not external_id:
            return None

        title = (
            node.get("marketplace_listing_title")
            or node.get("title")
            or node.get("name")
            or ""
        )
        if not title:
            return None

        # Price
        price_obj = node.get("listing_price") or node.get("price") or {}
        if isinstance(price_obj, dict):
            price_amount = price_obj.get("amount") or price_obj.get("amount_with_offset_in_currency", 0)
            currency = price_obj.get("currency", "RON")
            try:
                price = float(price_amount) / 100 if price_amount > 1000 else float(price_amount)
            except (TypeError, ValueError):
                price = 0.0
        elif isinstance(price_obj, (int, float)):
            price = float(price_obj)
            currency = "RON"
        else:
            price_str = str(price_obj)
            price_match = re.search(r"[\d,]+\.?\d*", price_str.replace(",", ""))
            price = float(price_match.group().replace(",", "")) if price_match else 0.0
            currency = "RON"

        if price <= 0:
            return None

        url = node.get("url") or f"https://www.facebook.com/marketplace/item/{external_id}/"

        # Location
        location_obj = (
            node.get("location")
            or node.get("delivery_location")
            or node.get("location_text")
            or {}
        )
        if isinstance(location_obj, dict):
            location = (
                location_obj.get("city")
                or location_obj.get("reverse_geocode", {}).get("city", "")
                or location_obj.get("name", "")
            )
        else:
            location = str(location_obj)

        # Condition
        condition_raw = node.get("condition") or node.get("item_condition") or "unknown"
        condition = str(condition_raw).lower().replace("_", " ")

        # Images
        images: list[str] = []
        primary_img = node.get("primary_listing_photo") or node.get("cover_photo") or {}
        if isinstance(primary_img, dict):
            img_url = (
                primary_img.get("image", {}).get("uri")
                or primary_img.get("uri")
                or primary_img.get("url")
                or ""
            )
            if img_url:
                images.append(img_url)
        all_photos = node.get("listing_photos") or node.get("photos") or []
        for p in (all_photos if isinstance(all_photos, list) else []):
            uri = (p.get("image", {}).get("uri") or p.get("uri") or "") if isinstance(p, dict) else ""
            if uri and uri not in images:
                images.append(uri)

        # Seller
        seller_obj = node.get("seller") or node.get("marketplace_listing_seller") or {}
        seller_name = ""
        seller_url = ""
        if isinstance(seller_obj, dict):
            seller_name = seller_obj.get("name") or seller_obj.get("display_name") or ""
            seller_id = seller_obj.get("id") or ""
            if seller_id:
                seller_url = f"https://www.facebook.com/{seller_id}"

        # Date
        posted_at: Optional[datetime] = None
        date_raw = node.get("creation_time") or node.get("created_at") or node.get("publish_time") or 0
        if date_raw:
            try:
                posted_at = datetime.utcfromtimestamp(int(date_raw))
            except Exception:
                pass

        return RawListing(
            external_id=external_id,
            platform="facebook",
            title=str(title),
            price=price,
            currency=currency,
            url=str(url),
            location=str(location),
            condition=condition,
            images=images,
            seller_name=str(seller_name),
            seller_url=str(seller_url),
            category=category,
            matched_keyword=keyword,
            posted_at=posted_at,
        )

    async def get_listing_detail(self, url: str) -> dict:
        id_match = re.search(r"/item/(\d+)", url)
        if not id_match:
            return {}
        try:
            client = await self._get_client()
            await self._random_delay(20)  # respect rate limit
            result = await client.call_tool("get_listing", {"listing_id": id_match.group(1)})
            return result if isinstance(result, dict) else {}
        except Exception as e:
            print(f"[FB-MCP] get_listing error: {e}")
            return {}
