"""
Facebook Marketplace scraper via httpx + saved session cookies.
No Playwright or Node.js required — works on any platform (including Android).

Setup:
  1. Log into Facebook in your browser.
  2. Export cookies with the "Cookie-Editor" extension → JSON.
  3. Save the file as facebook_cookies.json in the project root
     (or set FACEBOOK_COOKIES_FILE env var to its path).

How it works:
  1. GET https://www.facebook.com/marketplace/ → extract dtsg + jazoest tokens.
  2. POST to /api/graphql/ with the search query → parse listing edges.

Important: pass cookies as a raw Cookie header — httpx's cookie dict
re-encodes percent-encoded values (e.g. `xs`) and breaks authentication.
"""
import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

import httpx

from .base import BaseScraper, RawListing

logger = logging.getLogger(__name__)

# Romanian city coordinates (lat, lon)
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "bucharest":   (44.4268, 26.1025),
    "bucuresti":   (44.4268, 26.1025),
    "cluj-napoca": (46.7712, 23.6236),
    "timisoara":   (45.7489, 21.2087),
    "iasi":        (47.1585, 27.6014),
    "constanta":   (44.1598, 28.6348),
    "brasov":      (45.6427, 25.5887),
    "galati":      (45.4353, 28.0080),
    "ploiesti":    (44.9436, 26.0140),
    "craiova":     (44.3302, 23.7949),
}

_BASE = "https://www.facebook.com"

# Baseline headers — must include sec-fetch fields so FB returns real content
_HTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; SM-S908B) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Mobile Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_GQL_HEADERS = {
    "User-Agent": _HTML_HEADERS["User-Agent"],
    "Accept": "*/*",
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": _BASE,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-FB-Friendly-Name": "CometMarketplaceSearchContentContainerQuery",
}

# GraphQL persisted query ID for marketplace search
_DOC_ID = "7111939778879383"


def _parse_formatted_price(text: str) -> tuple[float, str]:
    """
    Parse a formatted FB price string into (amount_float, currency_str).
    Handles Romanian/European formats: '800 RON', '1.200 RON', '1.299,99 EUR'.
    """
    t = text.upper()
    if "EUR" in t or "\u20ac" in text:
        currency = "EUR"
    elif "USD" in t or "$" in text:
        currency = "USD"
    else:
        currency = "RON"
    # Extract numeric part only
    num = re.sub(r"[^\d.,]", "", text)
    if "," in num and "." in num:
        # e.g. '1.299,99' \u2192 dot=thousands, comma=decimal
        num = num.replace(".", "").replace(",", ".")
    elif "." in num and len(num.split(".")[-1]) == 3:
        # e.g. '1.200' \u2192 dot is thousands separator
        num = num.replace(".", "")
    elif "," in num:
        num = num.replace(",", ".")
    try:
        return float(num), currency
    except ValueError:
        return 0.0, currency


class FacebookMarketplaceScraper(BaseScraper):
    """
    HTTP-based Facebook Marketplace scraper using saved session cookies.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cookies_file = os.environ.get("FACEBOOK_COOKIES_FILE", "./facebook_cookies.json")
        self.radius_km = config.get("radius_km", 50)
        self.max_results = config.get("max_results_per_keyword", 40)
        location_raw = config.get("location", "bucuresti").lower()
        city_key = next((k for k in _CITY_COORDS if k in location_raw), "bucuresti")
        self.lat, self.lon = _CITY_COORDS[city_key]
        self._client: Optional[httpx.AsyncClient] = None
        self._cookie_header: Optional[str] = None  # raw Cookie header string
        self._dtsg: Optional[str] = None
        self._jazoest: Optional[str] = None

    # ── Cookie loading ─────────────────────────────────────────────────────────

    def _build_cookie_header(self) -> Optional[str]:
        """
        Load cookies from JSON file and build a raw Cookie header string.
        We send cookies as a raw header (not via httpx's cookie jar) to avoid
        double-encoding of percent-encoded values like `xs`.
        """
        if not os.path.exists(self.cookies_file):
            logger.info(f"[Facebook] Cookie file not found: {self.cookies_file}\n"
            "  Export your Facebook session cookies with the 'Cookie-Editor' "
            "browser extension (JSON format) and save them there.")
            return None
        with open(self.cookies_file) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return "; ".join(f"{c['name']}={c['value']}" for c in raw if "name" in c and "value" in c)
        if isinstance(raw, dict):
            return "; ".join(f"{k}={v}" for k, v in raw.items())
        return None

    # ── HTTP client ────────────────────────────────────────────────────────────

    async def _ensure_client(self) -> Optional[httpx.AsyncClient]:
        if self._client:
            return self._client
        cookie_hdr = self._build_cookie_header()
        if not cookie_hdr:
            return None
        self._cookie_header = cookie_hdr
        # Do NOT pass cookies to the client — they go as a raw header per request
        self._client = httpx.AsyncClient(follow_redirects=True, timeout=30.0)
        return self._client

    def _base_headers(self, extra: dict | None = None) -> dict:
        h = {**_HTML_HEADERS, "Cookie": self._cookie_header or ""}
        if extra:
            h.update(extra)
        return h

    # ── Session token extraction ───────────────────────────────────────────────

    async def _get_tokens(self) -> bool:
        """GET the marketplace page, extract dtsg + jazoest CSRF tokens."""
        if self._dtsg and self._jazoest:
            return True
        client = await self._ensure_client()
        if not client:
            return False
        try:
            resp = await client.get(f"{_BASE}/marketplace/", headers=self._base_headers())
            html = resp.text

            # dtsg: wlct.init({"dtsg":"TOKEN:23:...", ...})
            dtsg_m = re.search(r'"dtsg":"([^"]+)"', html)
            # jazoest (sprinkle value) — the secondary CSRF parameter
            jazoest_m = re.search(r'"sprinkleValue":"([^"]+)"', html)

            if dtsg_m:
                self._dtsg = dtsg_m.group(1)
            if jazoest_m:
                self._jazoest = jazoest_m.group(1)

            if not self._dtsg:
                logger.info("[Facebook] Could not extract dtsg token — cookies may be expired.")
                return False
            return True
        except Exception as e:
            logger.info(f"[Facebook] Token fetch error: {e}")
            return False

    # ── Main search ────────────────────────────────────────────────────────────

    async def search(self, keyword: str, category_config: dict) -> list[RawListing]:
        client = await self._ensure_client()
        if not client:
            return []

        if not await self._get_tokens():
            return []

        price_range = category_config.get("price_range", {})
        category = category_config.get("name", "")

        browse_params: dict = {
            "commerce_enable_local_pickup": True,
            "commerce_search_and_rp_available": True,
            "filter_location_latitude": self.lat,
            "filter_location_longitude": self.lon,
            "filter_radius_km": self.radius_km,
        }
        # Price bounds use FB-internal units — filter by parsed RON price locally instead

        variables = {
            "buyLocation": {"latitude": self.lat, "longitude": self.lon},
            "count": 24,  # always fetch 24; ~25% may lack listing data, so cap locally
            "cursor": None,
            "params": {
                "bqf": {"callsite": "COMMERCE_MKTPLACE_WWW", "query": keyword},
                "browse_request_params": browse_params,
                "custom_request_params": {"surface": "SEARCH"},
            },
            "savedSearchID": None,
            "savedSearchQuery": keyword,
            "scale": 1,
            "shouldIncludePopularSearches": False,
            "topicPageParams": {"location_id": "108378372518985", "url": None},
        }

        post_data = {
            "fb_dtsg": self._dtsg,
            "jazoest": self._jazoest or "",
            "variables": json.dumps(variables),
            "doc_id": _DOC_ID,
            "__a": "1",
        }

        gql_headers = self._base_headers({
            **_GQL_HEADERS,
            "Referer": f"{_BASE}/marketplace/search?query={keyword.replace(' ', '%20')}",
        })

        try:
            await self._random_delay(1.2)
            resp = await client.post(
                f"{_BASE}/api/graphql/",
                data=post_data,
                headers=gql_headers,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.info(f"[Facebook] GraphQL request failed for '{keyword}': {e}")
            return []

        return self._parse_response(resp.text, keyword, category, price_range)

    # ── Response parser ────────────────────────────────────────────────────────

    def _parse_response(
        self, text: str, keyword: str, category: str, price_range: dict
    ) -> list[RawListing]:
        # Facebook sometimes prepends "for (;;);" to prevent JSON hijacking
        body = text[9:] if text.startswith("for (;;);") else text
        try:
            data = json.loads(body)
        except Exception:
            logger.info(f"[Facebook] Could not parse GraphQL response ({len(text)} bytes)")
            return []

        try:
            edges = data["data"]["marketplace_search"]["feed_units"]["edges"]
        except (KeyError, TypeError):
            edges = []

        if not edges:
            logger.info(f"[Facebook] No edges in response (keys: {list(data.get('data', {}).keys())})")
            return []

        min_p = price_range.get("min_ron", 0)
        max_p = price_range.get("max_ron", 999_999)
        listings = []
        for edge in edges[:self.max_results]:
            try:
                # Listing data lives under edge.node.listing
                listing_node = edge.get("node", {}).get("listing")
                if not listing_node:
                    continue
                parsed = self._parse_listing(listing_node, keyword, category)
                if parsed and min_p <= parsed.price <= max_p:
                    listings.append(parsed)
            except Exception as e:
                logger.info(f"[Facebook] Parse error: {e}")
        return listings

    # ── Listing node parser ────────────────────────────────────────────────────

    def _parse_listing(self, node: dict, keyword: str, category: str) -> Optional[RawListing]:
        external_id = str(node.get("id", ""))
        if not external_id:
            return None

        title = str(node.get("marketplace_listing_title") or node.get("custom_title") or "")
        if not title:
            return None

        # Price — prefer formatted_amount string ("800 RON") over the raw int
        price_obj = node.get("listing_price") or {}
        price, currency = 0.0, "RON"
        if isinstance(price_obj, dict):
            formatted = price_obj.get("formatted_amount") or price_obj.get("amount") or ""
            if formatted:
                price, currency = _parse_formatted_price(str(formatted))
            else:
                # Fallback: amount_with_offset_in_currency is FB-internal, not usable directly
                pass
        elif isinstance(price_obj, (int, float)):
            price = float(price_obj)

        if price <= 0:
            return None

        url = str(node.get("url") or f"{_BASE}/marketplace/item/{external_id}/")

        # Location
        loc = node.get("location") or {}
        location = ""
        if isinstance(loc, dict):
            rg = loc.get("reverse_geocode") or {}
            location = rg.get("city", "") or loc.get("city", "") or loc.get("name", "")

        # Condition
        condition = str(node.get("condition") or "unknown").lower().replace("_", " ")

        # Images
        images: list[str] = []
        primary = node.get("primary_listing_photo") or {}
        if isinstance(primary, dict):
            uri = (
                (primary.get("image") or {}).get("uri")
                or primary.get("uri")
                or ""
            )
            if uri:
                images.append(uri)

        # Seller
        seller = node.get("marketplace_listing_seller") or {}
        seller_name = ""
        seller_url = ""
        if isinstance(seller, dict):
            seller_name = str(seller.get("name") or "")
            sid = seller.get("id", "")
            if sid:
                seller_url = f"{_BASE}/{sid}"

        # Date
        posted_at: Optional[datetime] = None
        for key in ("creation_time", "created_at", "publish_time"):
            raw_date = node.get(key)
            if raw_date:
                try:
                    posted_at = datetime.utcfromtimestamp(int(raw_date))
                    break
                except Exception:
                    pass

        return RawListing(
            external_id=external_id,
            platform="facebook",
            title=title,
            price=price,
            currency=currency,
            url=url,
            location=str(location),
            condition=condition,
            images=images,
            seller_name=seller_name,
            seller_url=seller_url,
            category=category,
            matched_keyword=keyword,
            posted_at=posted_at,
        )

    async def get_listing_detail(self, url: str) -> dict:
        return {}

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
