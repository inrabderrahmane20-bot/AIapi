# AI.py - full updated file with tagline support and improved map fallbacks
import os
import re
import json
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import requests
import wikipediaapi
import diskcache
from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dataclasses import dataclass
from urllib.parse import quote_plus

# -------------------- CONFIG --------------------
@dataclass
class Config:
    UNSPLASH_ACCESS_KEY: str = ""  # Optional
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    MAX_IMAGE_WORKERS: int = int(os.getenv("MAX_IMAGE_WORKERS", "8"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    MAPBOX_ACCESS_TOKEN: str = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    MAP_TILE_PROVIDER: str = os.getenv("MAP_TILE_PROVIDER", "openstreetmap")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./diskcache")
    PRELOAD_TOP_CITIES: int = int(os.getenv("PRELOAD_TOP_CITIES", "12"))
    LOCAL_CACHE_FILE: str = os.getenv("LOCAL_CACHE_FILE", "./diskcache/cities_data.json")
    MAX_WIKIMEDIA_FILES_TO_SCAN: int = 50
    MAX_IMAGES_PER_REQUEST: int = 6
    WIKIMEDIA_RETRY_ATTEMPTS: int = 3

config = Config()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- DISKCACHE --------------------
cache = diskcache.Cache(config.CACHE_DIR)

# -------------------- LOCAL JSON CACHE --------------------
LOCAL_CITY_CACHE: Dict[str, dict] = {}
if os.path.exists(config.LOCAL_CACHE_FILE):
    try:
        with open(config.LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
            LOCAL_CITY_CACHE = json.load(f)
        logger.info(f"Loaded {len(LOCAL_CITY_CACHE)} cities from local cache")
    except Exception as e:
        logger.warning(f"Failed to load local JSON cache: {e}")

def save_local_cache():
    try:
        os.makedirs(os.path.dirname(config.LOCAL_CACHE_FILE), exist_ok=True)
        with open(config.LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(LOCAL_CITY_CACHE, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(LOCAL_CITY_CACHE)} cities to local cache")
    except Exception as e:
        logger.error(f"Failed to save local cache: {e}")

# -------------------- CACHE CLEARING FUNCTIONS --------------------
def clear_city_cache(city_names=None):
    """Clear cache for specific cities or all cities"""
    global LOCAL_CITY_CACHE

    if city_names is None:
        # Clear all cities
        cities_to_clear = list(LOCAL_CITY_CACHE.keys())
    else:
        cities_to_clear = city_names

    for city_name in cities_to_clear:
        if city_name in LOCAL_CITY_CACHE:
            del LOCAL_CITY_CACHE[city_name]
            logger.info(f"Cleared cache for {city_name}")

    # Also clear disk cache for these cities
    try:
        if city_names is None:
            cache.clear()
            logger.info("Cleared all disk cache")
        else:
            for city_name in city_names:
                # We need to clear various cache keys that might be associated with this city
                cache_keys_to_clear = []
                for key in cache:
                    if isinstance(key, str) and city_name.lower() in key.lower():
                        cache_keys_to_clear.append(key)

                for key in cache_keys_to_clear:
                    try:
                        cache.delete(key)
                    except:
                        pass
                logger.info(f"Cleared disk cache for {city_name}")
    except Exception as e:
        logger.warning(f"Error clearing disk cache: {e}")

    save_local_cache()

# -------------------- CACHE DECORATORS --------------------
def disk_cached(ttl=config.CACHE_TTL):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # build stable cache key, skip self for methods
            args_for_key = args[1:] if len(args) > 0 and hasattr(args[0], "__class__") else args
            try:
                key = f"{fn.__name__}|{json.dumps({'args': args_for_key,'kwargs': kwargs},default=str,sort_keys=True)}"
            except Exception:
                key = f"{fn.__name__}|{str(args_for_key)}|{str(kwargs)}"
            now = time.time()
            try:
                cached = cache.get(key)
            except Exception:
                cached = None
            if cached and (now - cached.get("ts",0) < ttl):
                return cached["val"]
            val = fn(*args, **kwargs)
            try:
                cache.set(key, {"ts": time.time(), "val": val}, expire=ttl)
            except Exception as e:
                logger.warning(f"Failed to set cache for key {key}: {e}")
            return val
        return wrapper
    return decorator

# -------------------- SIMPLE GET WITH SERIALIZABLE CACHE --------------------
def _make_cache_key(url: str, params: dict = None, headers: dict = None) -> str:
    p = json.dumps(params or {}, sort_keys=True, default=str)
    h = json.dumps(headers or {}, sort_keys=True, default=str)
    return f"GET {url} {p} {h}"

def cached_get(url: str, params: dict = None, headers: dict = None, ttl: int = config.CACHE_TTL):
    """
    Performs a GET and caches serializable data. Returns a lightweight response-like object
    with .status_code and .json() to preserve compatibility with existing callers.
    """
    key = _make_cache_key(url, params, headers)
    now = time.time()
    try:
        cached = cache.get(key)
    except Exception:
        cached = None
    if cached and (now - cached.get("ts",0) < ttl):
        class SimpleResp:
            def __init__(self, code, data):
                self.status_code = code
                self._data = data
            def json(self):
                return self._data
            @property
            def text(self):
                try:
                    return json.dumps(self._data)
                except Exception:
                    return str(self._data)
        return SimpleResp(cached["status_code"], cached["data"])

    try:
        hdrs = dict(headers or {})
        hdrs.setdefault("User-Agent", "CityExplorer/1.0")
        resp = requests.get(url, params=params or {}, headers=hdrs or {}, timeout=config.REQUEST_TIMEOUT)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        store = {"ts": now, "status_code": resp.status_code, "data": data}
        try:
            cache.set(key, store, expire=ttl)
        except Exception as e:
            logger.debug(f"Failed to write GET cache for {key}: {e}")

        class SimpleResp:
            def __init__(self, code, data):
                self.status_code = code
                self._data = data
            def json(self):
                return self._data
            @property
            def text(self):
                try:
                    return json.dumps(self._data)
                except Exception:
                    return str(self._data)
        return SimpleResp(resp.status_code, data)
    except Exception as e:
        logger.error(f"cached_get error for {url}: {e}")
        class DummyResponse:
            status_code = 500
            def json(self): return {"error": "Request failed", "exception": str(e)}
            @property
            def text(self): return str(e)
        return DummyResponse()

# -------------------- MAP PROVIDER --------------------
class MapProvider:
    def __init__(self):
        self.tile_providers = {
            "openstreetmap": {
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attribution": "&copy; OpenStreetMap contributors"
            },
            "carto": {
                "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                "attribution": "&copy; OpenStreetMap & CARTO"
            }
        }
        if config.MAPBOX_ACCESS_TOKEN:
            self.tile_providers["mapbox"] = {
                "url": f"https://api.mapbox.com/styles/v1/mapbox/light-v10/tiles/{{z}}/{{x}}/{{y}}?access_token={config.MAPBOX_ACCESS_TOKEN}",
                "attribution": "&copy; Mapbox & OpenStreetMap"
            }

    def get_map_config(self, city_name: str, coordinates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        # Safely get tile provider config with fallback
        default_provider = "openstreetmap"
        provider_key = config.MAP_TILE_PROVIDER if hasattr(config, 'MAP_TILE_PROVIDER') else default_provider

        provider_config = self.tile_providers.get(provider_key, self.tile_providers[default_provider])

        # Always return basic map configuration
        map_config = {
            "tile_provider": provider_key,
            "tile_url": provider_config["url"],
            "attribution": provider_config["attribution"],
            "zoom": 12,
            "min_zoom": 2,
            "max_zoom": 18,
            "center": {"lat": 0, "lon": 0},  # Default center
            "marker": None  # No marker by default
        }

        # Add coordinates and marker if valid coordinates are provided
        if coordinates and self._validate_coordinates(coordinates):
            map_config.update({
                "center": coordinates,
                "marker": {
                    "coordinates": coordinates,
                    "popup_content": f"<strong>{city_name}</strong>",
                    "color": "#3388ff"
                }
            })

        return map_config

    def generate_static_map_url(self, coordinates: Optional[Dict[str, float]], width: int = 600, height: int = 400) -> Optional[str]:
        """Return a static map URL using Mapbox if configured else use OpenStreetMap static service"""
        if not coordinates or not self._validate_coordinates(coordinates):
            # No coordinates - fallback to a generic world map placeholder if present
            placeholder = "/static/default_world_map.jpg"
            if os.path.exists(placeholder.lstrip("/")):
                return placeholder
            return None

        lat, lon = coordinates["lat"], coordinates["lon"]

        # Mapbox static if token present
        if hasattr(config, 'MAPBOX_ACCESS_TOKEN') and config.MAPBOX_ACCESS_TOKEN:
            try:
                return f"https://api.mapbox.com/styles/v1/mapbox/light-v10/static/pin-l+3388ff({lon},{lat})/{lon},{lat},12/{width}x{height}?access_token={config.MAPBOX_ACCESS_TOKEN}"
            except Exception:
                pass

        # OpenStreetMap static via staticmap.openstreetmap.de (no API key)
        try:
            # staticmap.openstreetmap.de expects width x height up to reasonable limits (e.g., 1024)
            zoom = 12
            size = f"{width}x{height}"
            return f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={size}&markers={lat},{lon},red-pushpin"
        except Exception:
            pass

        # Last resort - placeholder if present
        placeholder = "/static/default_world_map.jpg"
        if os.path.exists(placeholder.lstrip("/")):
            return placeholder

        return None

    def _validate_coordinates(self, coordinates: Dict[str, float]) -> bool:
        """Validate that coordinates contain required keys and valid values"""
        try:
            lat = coordinates.get("lat")
            lon = coordinates.get("lon")

            # Check if values exist and are numeric
            if lat is None or lon is None:
                return False

            # Convert to float if they're strings
            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                return False

            # Check valid ranges
            return (-90 <= lat <= 90 and -180 <= lon <= 180)

        except (TypeError, ValueError, AttributeError):
            return False

# -------------------- CITY DATA PROVIDER --------------------
class CityDataProvider:
    def __init__(self, user_agent: str = "CityExplorer/1.0"):
        browser_like_ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 CityExplorer/1.0'
        self.user_agent = user_agent or browser_like_ua
        self.geolocator = Nominatim(user_agent=self.user_agent, timeout=max(10, config.REQUEST_TIMEOUT))
        self.wiki = wikipediaapi.Wikipedia(language='en', user_agent=self.user_agent)
        self.map_provider = MapProvider()

    def _get_country_for_city(self, city_name: str) -> str:
        """Get country name for a city to help with Wikipedia search"""
        for city in WORLD_CITIES:
            if city["name"].lower() == city_name.lower():
                return city["country"]
        return ""

    # ---------- COORDINATES ----------
    @disk_cached(ttl=86400)
    def get_coordinates(self, city_name: str) -> Optional[Tuple[float,float,dict]]:
        """Try geolocator first, then Wikipedia coordinate fallback, with small retries"""
        # Try Nominatim / geopy
        try:
            location = self.geolocator.geocode(city_name, exactly_one=True, addressdetails=True, language="en")
            if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
                logger.info(f"Found coordinates for {city_name}: {location.latitude}, {location.longitude} (nominatim)")
                return (location.latitude, location.longitude, getattr(location,'raw',{}))

            # Try with country hint
            country_hint = self._get_country_for_city(city_name)
            if country_hint:
                try_name = f"{city_name}, {country_hint}"
                location = self.geolocator.geocode(try_name, exactly_one=True, addressdetails=True, language="en")
                if location and hasattr(location, 'latitude') and hasattr(location, 'longitude'):
                    logger.info(f"Found coordinates for {city_name} using country hint: {location.latitude}, {location.longitude}")
                    return (location.latitude, location.longitude, getattr(location,'raw',{}))
        except Exception as e:
            logger.warning(f"Geo error (nominatim) {city_name}: {e}")

        # Fallback: try to extract coordinates from Wikipedia via MediaWiki API
        try:
            wiki_coords = self.get_coordinates_from_wikipedia(city_name)
            if wiki_coords:
                logger.info(f"Found coordinates for {city_name} via Wikipedia: {wiki_coords}")
                return (wiki_coords["lat"], wiki_coords["lon"], {"source": "wikipedia"})
        except Exception as e:
            logger.debug(f"Wikipedia coordinate fallback failed for {city_name}: {e}")

        return None

    @disk_cached(ttl=86400)
    def get_coordinates_from_wikipedia(self, city_name: str) -> Optional[Dict[str, float]]:
        """Use the MediaWiki API to fetch coordinate metadata for the city's page (if available)."""
        api_url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": self.user_agent}
        # Try exact title and some variations to find a page with coordinates
        variations = [city_name, f"{city_name} (city)", f"{city_name}, {self._get_country_for_city(city_name)}", city_name.split(',')[0].strip()]
        for var in variations:
            if not var:
                continue
            params = {
                "action": "query",
                "titles": var,
                "prop": "coordinates",
                "format": "json",
                "colimit": 1
            }
            resp = cached_get(api_url, params=params, headers=headers, ttl=3600)
            if getattr(resp, "status_code", 0) == 200:
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    coords = page.get("coordinates")
                    if coords and isinstance(coords, list) and len(coords) > 0:
                        c = coords[0]
                        lat = c.get("lat")
                        lon = c.get("lon")
                        if lat is not None and lon is not None:
                            try:
                                return {"lat": float(lat), "lon": float(lon)}
                            except Exception:
                                continue
        return None

    # ---------- WIKIPEDIA PAGE ----------
    @disk_cached(ttl=3600)
    def fetch_wikipedia_page(self, city_name: str):
        try:
            # Special handling for problematic cities with exact Wikipedia titles
            wiki_exact_titles = {
                "London": "London",
                "Rome": "Rome",
                "Paris": "Paris",
                "New York": "New York City",
                "Washington DC": "Washington, D.C.",
                "Barcelona": "Barcelona",
                "Amsterdam": "Amsterdam",
                "Berlin": "Berlin",
                "Prague": "Prague",
                "Vienna": "Vienna",
                "Budapest": "Budapest",
                "Lisbon": "Lisbon"
            }

            exact_title = wiki_exact_titles.get(city_name, city_name)
            page = self.wiki.page(exact_title)

            if page.exists() and page.ns == 0 and hasattr(page, 'text') and page.text:
                sections = self.extract_sections(page)
                result = {
                    "title": page.title,
                    "summary": (page.summary or "")[:2000],
                    "fullurl": getattr(page, "fullurl", ""),
                    "sections": sections
                }
                logger.info(f"Found Wikipedia page for {city_name}: {page.title}")
                return result, page.title

            # If exact title fails, try common variations
            variations = [
                f"{city_name} (city)",
                f"{city_name}, {self._get_country_for_city(city_name)}",
                city_name.split(',')[0].strip(),
                f"{city_name} City"
            ]

            for variation in variations:
                if not variation or variation == city_name:
                    continue

                page = self.wiki.page(variation)
                if page.exists() and page.ns == 0 and hasattr(page, 'text') and page.text:
                    sections = self.extract_sections(page)
                    result = {
                        "title": page.title,
                        "summary": (page.summary or "")[:2000],
                        "fullurl": getattr(page, "fullurl", ""),
                        "sections": sections
                    }
                    logger.info(f"Found Wikipedia page for {city_name} via variation '{variation}': {page.title}")
                    return result, page.title

            logger.warning(f"No valid Wikipedia page found for {city_name}")
            return None, None

        except Exception as e:
            logger.warning(f"Wikipedia fetch error {city_name}: {e}")
            return None, None

    def extract_sections(self, page):
        sections = {}
        def recurse(section, depth=0):
            if depth > 1:
                return
            title = section.title.strip()
            text = (section.text or "").strip()
            if title and text and depth <= 1:
                cleaned = re.sub(r'\[\d+\]', '', text)
                if title in sections:
                    sections[title] += f"\n\n{cleaned}"
                else:
                    sections[title] = cleaned
            for subsection in getattr(section, 'sections', []):
                recurse(subsection, depth + 1)

        for section in getattr(page, 'sections', []):
            recurse(section)
        return sections

    # ---------- ENHANCED WIKIMEDIA IMAGES FETCHING ----------
    @disk_cached(ttl=86400)
    def get_wikimedia_images(self, page_title: str, limit: int = 6) -> List[Dict[str, Any]]:
        """
        Enhanced image fetching with better error handling and more comprehensive search
        """
        api_url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": self.user_agent}
        images = []

        try:
            # Strategy 1: Get lead image with higher quality
            params = {
                "action": "query",
                "titles": page_title,
                "prop": "pageimages",
                "pithumbsize": 1000,  # Higher quality
                "format": "json",
                "piprop": "thumbnail|name"
            }

            resp = cached_get(api_url, params=params, headers=headers, ttl=3600)
            if getattr(resp, "status_code", 0) == 200:
                data = resp.json()
                for page in data.get("query", {}).get("pages", {}).values():
                    thumb = page.get("thumbnail")
                    if thumb and thumb.get("source"):
                        url = thumb["source"]
                        if url.startswith("//"):
                            url = "https:" + url

                        images.append({
                            "url": url,
                            "title": f"Main image of {page_title}",
                            "description": f"Featured image of {page_title}",
                            "source": "wikipedia",
                            "width": thumb.get("width", 800),
                            "height": thumb.get("height", 600)
                        })
                        logger.info(f"Found lead image for {page_title}: {url}")
                        break

            # Strategy 2: Get more images from page with better filtering
            if len(images) < limit:
                params = {
                    "action": "query",
                    "titles": page_title,
                    "prop": "images",
                    "format": "json",
                    "imlimit": 20  # Get more images to filter from
                }

                resp = cached_get(api_url, params=params, headers=headers, ttl=3600)
                if getattr(resp, "status_code", 0) == 200:
                    data = resp.json()
                    file_titles = []

                    for page in data.get("query", {}).get("pages", {}).values():
                        for img in page.get("images", []):
                            title = img.get("title", "")
                            if title.startswith("File:"):
                                # Better filtering - exclude SVGs, audio, video, but include more image types
                                lower_title = title.lower()
                                excluded_extensions = ['.svg', '.ogg', '.webm', '.tif', '.tiff', '.xcf']
                                if not any(ext in lower_title for ext in excluded_extensions):
                                    file_titles.append(title)

                    # Get image info in smaller batches with better error handling
                    if file_titles:
                        batch_size = 8
                        for i in range(0, min(len(file_titles), 24), batch_size):  # Limit to 24 files max
                            batch = file_titles[i:i + batch_size]
                            titles_param = "|".join(batch)
                            params = {
                                "action": "query",
                                "titles": titles_param,
                                "prop": "imageinfo",
                                "iiprop": "url|size|mime",
                                "iiurlwidth": 800,
                                "format": "json"
                            }

                            resp = cached_get(api_url, params=params, headers=headers, ttl=3600)
                            if getattr(resp, "status_code", 0) == 200:
                                data = resp.json()
                                for page in data.get("query", {}).get("pages", {}).values():
                                    iinfo = page.get("imageinfo")
                                    if iinfo:
                                        info = iinfo[0]
                                        url = info.get("thumburl") or info.get("url")
                                        mime = info.get("mime", "")

                                        # Only include if we have a valid URL and it's an image
                                        if (url and not url.startswith("//") and
                                            mime and mime.startswith('image/')):

                                            images.append({
                                                "url": url,
                                                "title": page.get("title", "").replace("File:", ""),
                                                "description": f"Image from {page_title}",
                                                "source": "wikipedia",
                                                "width": info.get("width"),
                                                "height": info.get("height")
                                            })
                                            if len(images) >= limit:
                                                break
                            if len(images) >= limit:
                                break

            logger.info(f"Found {len(images)} images for {page_title}")
            return images[:limit]

        except Exception as e:
            logger.error(f"Enhanced image fetch error for {page_title}: {e}")
            return images[:limit]

    # ---------- SHORT CITY DESCRIPTION / TAGLINE ----------
    @disk_cached(ttl=86400)
    def get_city_tagline(self, city_name: str) -> Dict[str, str]:
        """Return a short, tagline-style description for a city (e.g. 'Paris → City of Light')."""
        city_name_clean = city_name.strip()
        if not city_name_clean:
            return {"city": city_name_clean, "tagline": "A beautiful city worth exploring", "source": "default"}

        # Normalize title casing for known taglines lookup
        title_key = city_name_clean.title()

        # 1️⃣ Hardcoded famous taglines
        known_taglines = {
            "Paris": "The City of Light",
            "New York": "The Big Apple",
            "Tokyo": "The Eastern Capital",
            "London": "The Old Smoke",
            "Rome": "The Eternal City",
            "Venice": "The Floating City",
            "Los Angeles": "The City of Angels",
            "San Francisco": "The Golden City",
            "Berlin": "The City of Freedom",
            "Dubai": "The City of Gold",
            "Barcelona": "The City of Gaudí",
            "Istanbul": "Where East Meets West",
            "Marrakesh": "The Red City",
            "Cairo": "The City of a Thousand Minarets",
            "Lisbon": "The City of Seven Hills",
            "Amsterdam": "The Venice of the North"
        }

        if title_key in known_taglines:
            tagline = known_taglines[title_key]
            return {"city": city_name_clean, "tagline": tagline, "source": "known"}

        # 2️⃣ If not known, try to get a short one-line summary from Wikipedia (try to keep it very short)
        try:
            page_data, _ = self.fetch_wikipedia_page(city_name_clean)
            if page_data and page_data.get("summary"):
                # Remove parenthesis and references, then keep first 10-12 words
                summary = re.sub(r'\(.*?\)', '', page_data["summary"])
                summary = re.sub(r'\[\d+\]', '', summary)
                words = summary.strip().split()
                if not words:
                    raise ValueError("Empty summary")
                tagline = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
                # Trim trailing punctuation
                tagline = tagline.strip().rstrip('.,;:')
                return {"city": city_name_clean, "tagline": tagline, "source": "wikipedia"}
        except Exception as e:
            logger.debug(f"Tagline fetch failed for {city_name_clean}: {e}")

        # 3️⃣ Default fallback
        return {"city": city_name_clean, "tagline": "A beautiful city worth exploring", "source": "default"}

    # ---------- CITY PREVIEW ----------
    def get_city_preview(self, city_name: str) -> Dict:
        # Respect local cache
        cached = LOCAL_CITY_CACHE.get(city_name)
        if cached:
            # Ensure tagline available
            if "tagline" not in cached:
                try:
                    tagline_obj = self.get_city_tagline(city_name)
                    cached["tagline"] = tagline_obj.get("tagline")
                    cached["tagline_source"] = tagline_obj.get("source")
                except Exception:
                    pass
            return cached

        preview = {
            "id": self._generate_city_id(city_name),
            "name": city_name,
            "display_name": city_name,
            "summary": "Information available",
            "has_details": False,
            "image": {"url": "/static/default_city.jpg", "source": "placeholder"},
            "tagline": None,
            "tagline_source": None
        }

        coords = self.get_coordinates(city_name)
        if coords:
            preview["coordinates"] = {"lat": coords[0], "lon": coords[1]}
            preview["static_map"] = self.map_provider.generate_static_map_url(
                {"lat": coords[0], "lon": coords[1]}, width=400, height=200
            )
        else:
            # Try to still produce a static_map using placeholder behavior
            preview["static_map"] = self.map_provider.generate_static_map_url(None, width=400, height=200)

        page_data, title = self.fetch_wikipedia_page(city_name)
        if page_data:
            preview["display_name"] = page_data.get("title", city_name)
            summary = page_data.get("summary", "")
            preview["summary"] = (summary[:200] + "...") if summary else "Information available"
            preview["has_details"] = True

            # Try to set image via Wikimedia
            images = []
            max_attempts = config.WIKIMEDIA_RETRY_ATTEMPTS

            for attempt in range(max_attempts):
                try:
                    images = self.get_wikimedia_images(title or city_name, limit=1)
                    if images:
                        img_url = images[0].get("url", "")
                        if img_url and img_url != "/static/default_city.jpg":
                            preview["image"] = images[0]
                            logger.info(f"Successfully set image for {city_name} on attempt {attempt + 1}")
                            break

                    # If no images found, wait before retry (exponential-ish backoff)
                    if attempt < max_attempts - 1:
                        wait_time = 1 * (attempt + 1)
                        logger.info(f"No images found for {city_name} on attempt {attempt + 1}, retrying in {wait_time}s")
                        time.sleep(wait_time)

                except Exception as e:
                    logger.warning(f"Image fetch attempt {attempt + 1} failed for {city_name}: {e}")
                    if attempt == max_attempts - 1:
                        logger.error(f"All image fetch attempts failed for {city_name}")

        # Tagline (always populate, cached inside)
        try:
            tagline_obj = self.get_city_tagline(city_name)
            preview["tagline"] = tagline_obj.get("tagline")
            preview["tagline_source"] = tagline_obj.get("source")
        except Exception as e:
            logger.debug(f"Failed to fetch tagline for {city_name}: {e}")
            preview["tagline"] = "A beautiful city worth exploring"
            preview["tagline_source"] = "default"

        # Save to local cache
        LOCAL_CITY_CACHE[city_name] = preview
        save_local_cache()
        return preview

    # ---------- CITY DETAILS ----------
    def get_city_details(self, city_name: str) -> Dict:
        details = {
            "id": self._generate_city_id(city_name),
            "name": city_name,
            "coordinates": None,
            "sections": [],
            "landmarks": [],
            "images": [],
            "sources": [],
            "map": {},
            "last_updated": time.time(),
            "image": {"url": "/static/default_city.jpg", "source": "placeholder"},
            "static_map": None,
            "tagline": None,
            "tagline_source": None
        }

        coords = self.get_coordinates(city_name)
        if coords:
            details["coordinates"] = {"lat": coords[0], "lon": coords[1]}

        # If still no coordinates, attempt to fetch from Wikipedia coordinates endpoint as last effort
        if not details["coordinates"]:
            try:
                wiki_coords = self.get_coordinates_from_wikipedia(city_name)
                if wiki_coords:
                    details["coordinates"] = {"lat": wiki_coords["lat"], "lon": wiki_coords["lon"]}
            except Exception:
                pass

        # ALWAYS generate map config, even without coordinates
        details["map"] = self.map_provider.get_map_config(city_name, details.get("coordinates"))
        details["static_map"] = self.map_provider.generate_static_map_url(details.get("coordinates"), width=800, height=400)

        page_data, title = self.fetch_wikipedia_page(city_name)
        if page_data and title:
            details["name"] = page_data.get("title", city_name)
            details["wikipedia_summary"] = (page_data.get("summary") or "")[:1000]
            fullurl = page_data.get("fullurl", "")
            if fullurl:
                details["sources"].append(fullurl)

            # Structure sections properly
            sections_data = []
            for section_title, content in page_data.get("sections", {}).items():
                if content.strip():
                    sections_data.append({
                        "title": section_title,
                        "content": content[:400] + ("..." if len(content) > 400 else "")
                    })
            details["sections"] = sections_data[:4]

            # Enhanced image fetching for details with retry logic
            images = []
            max_attempts = config.WIKIMEDIA_RETRY_ATTEMPTS

            for attempt in range(max_attempts):
                try:
                    images = self.get_wikimedia_images(title, limit=6)
                    if images:
                        details["images"] = images
                        details["image"] = images[0]  # Use first image as main image
                        logger.info(f"Successfully set {len(images)} images for {city_name} details")
                        break

                    # If no images found, wait before retry
                    if attempt < max_attempts - 1:
                        time.sleep(1 * (attempt + 1))

                except Exception as e:
                    logger.warning(f"Details image fetch attempt {attempt + 1} failed for {city_name}: {e}")
                    if attempt == max_attempts - 1:
                        logger.error(f"All details image fetch attempts failed for {city_name}")

        # Tagline
        try:
            tagline_obj = self.get_city_tagline(city_name)
            details["tagline"] = tagline_obj.get("tagline")
            details["tagline_source"] = tagline_obj.get("source")
        except Exception as e:
            details["tagline"] = "A beautiful city worth exploring"
            details["tagline_source"] = "default"

        return details

    def _generate_city_id(self, city_name: str) -> str:
        return re.sub(r'[^a-z0-9-]', '', city_name.lower().replace(' ', '-'))

# -------------------- DATA --------------------
REGIONS = ["Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"]

WORLD_CITIES = [
    # EUROPE 
    {"name":"Paris","country":"France","region":"Europe"},
    {"name":"London","country":"United Kingdom","region":"Europe"},
    {"name":"Rome","country":"Italy","region":"Europe"},
    {"name":"Barcelona","country":"Spain","region":"Europe"},
    {"name":"Amsterdam","country":"Netherlands","region":"Europe"},
    {"name":"Berlin","country":"Germany","region":"Europe"},
    {"name":"Prague","country":"Czech Republic","region":"Europe"},
    {"name":"Vienna","country":"Austria","region":"Europe"},
    {"name":"Budapest","country":"Hungary","region":"Europe"},
    {"name":"Lisbon","country":"Portugal","region":"Europe"},
    {"name":"Madrid","country":"Spain","region":"Europe"},
    {"name":"Florence","country":"Italy","region":"Europe"},
    {"name":"Venice","country":"Italy","region":"Europe"},
    {"name":"Milan","country":"Italy","region":"Europe"},
    {"name":"Naples","country":"Italy","region":"Europe"},
    {"name":"Brussels","country":"Belgium","region":"Europe"},
    {"name":"Bruges","country":"Belgium","region":"Europe"},
    {"name":"Dublin","country":"Ireland","region":"Europe"},
    {"name":"Edinburgh","country":"Scotland","region":"Europe"},
    {"name":"Glasgow","country":"Scotland","region":"Europe"},
    {"name":"Manchester","country":"England","region":"Europe"},
    {"name":"Liverpool","country":"England","region":"Europe"},
    {"name":"Birmingham","country":"England","region":"Europe"},
    {"name":"Athens","country":"Greece","region":"Europe"},
    {"name":"Santorini","country":"Greece","region":"Europe"},
    {"name":"Mykonos","country":"Greece","region":"Europe"},
    {"name":"Crete","country":"Greece","region":"Europe"},
    {"name":"Istanbul","country":"Turkey","region":"Europe"},
    {"name":"Cappadocia","country":"Turkey","region":"Europe"},
    {"name":"Antalya","country":"Turkey","region":"Europe"},
    {"name":"Moscow","country":"Russia","region":"Europe"},
    {"name":"St Petersburg","country":"Russia","region":"Europe"},
    {"name":"Warsaw","country":"Poland","region":"Europe"},
    {"name":"Krakow","country":"Poland","region":"Europe"},
    {"name":"Copenhagen","country":"Denmark","region":"Europe"},
    {"name":"Stockholm","country":"Sweden","region":"Europe"},
    {"name":"Oslo","country":"Norway","region":"Europe"},
    {"name":"Helsinki","country":"Finland","region":"Europe"},
    {"name":"Reykjavik","country":"Iceland","region":"Europe"},
    {"name":"Zurich","country":"Switzerland","region":"Europe"},
    {"name":"Geneva","country":"Switzerland","region":"Europe"},
    {"name":"Lucerne","country":"Switzerland","region":"Europe"},
    {"name":"Munich","country":"Germany","region":"Europe"},
    {"name":"Hamburg","country":"Germany","region":"Europe"},
    {"name":"Frankfurt","country":"Germany","region":"Europe"},
    {"name":"Cologne","country":"Germany","region":"Europe"},
    {"name":"Dresden","country":"Germany","region":"Europe"},
    {"name":"Salzburg","country":"Austria","region":"Europe"},
    {"name":"Innsbruck","country":"Austria","region":"Europe"},
    {"name":"Ljubljana","country":"Slovenia","region":"Europe"},
    {"name":"Zagreb","country":"Croatia","region":"Europe"},
    {"name":"Dubrovnik","country":"Croatia","region":"Europe"},
    {"name":"Split","country":"Croatia","region":"Europe"},
    {"name":"Sarajevo","country":"Bosnia Herzegovina","region":"Europe"},
    {"name":"Belgrade","country":"Serbia","region":"Europe"},
    {"name":"Bucharest","country":"Romania","region":"Europe"},
    {"name":"Sofia","country":"Bulgaria","region":"Europe"},
    {"name":"Tallinn","country":"Estonia","region":"Europe"},
    {"name":"Riga","country":"Latvia","region":"Europe"},
    {"name":"Vilnius","country":"Lithuania","region":"Europe"},
    {"name":"Kiev","country":"Ukraine","region":"Europe"},
    {"name":"Lviv","country":"Ukraine","region":"Europe"},
    {"name":"Minsk","country":"Belarus","region":"Europe"},
    {"name":"Monaco","country":"Monaco","region":"Europe"},
    {"name":"Luxembourg City","country":"Luxembourg","region":"Europe"},
    {"name":"Valletta","country":"Malta","region":"Europe"},
    {"name":"Nicosia","country":"Cyprus","region":"Europe"},
    {"name":"Andorra la Vella","country":"Andorra","region":"Europe"},
    {"name":"San Marino","country":"San Marino","region":"Europe"},
    {"name":"Vaduz","country":"Liechtenstein","region":"Europe"},
    {"name":"Porto","country":"Portugal","region":"Europe"},
    {"name":"Seville","country":"Spain","region":"Europe"},
    {"name":"Granada","country":"Spain","region":"Europe"},
    {"name":"Valencia","country":"Spain","region":"Europe"},
    {"name":"Bilbao","country":"Spain","region":"Europe"},
    {"name":"Marseille","country":"France","region":"Europe"},
    {"name":"Lyon","country":"France","region":"Europe"},
    {"name":"Nice","country":"France","region":"Europe"},
    {"name":"Cannes","country":"France","region":"Europe"},
    {"name":"St Tropez","country":"France","region":"Europe"},
    {"name":"Bordeaux","country":"France","region":"Europe"},
    {"name":"Strasbourg","country":"France","region":"Europe"},
    {"name":"Versailles","country":"France","region":"Europe"},
    {"name":"Mont St Michel","country":"France","region":"Europe"},
    {"name":"Cinque Terre","country":"Italy","region":"Europe"},
    {"name":"Pisa","country":"Italy","region":"Europe"},
    {"name":"Siena","country":"Italy","region":"Europe"},
    {"name":"Verona","country":"Italy","region":"Europe"},
    {"name":"Lake Como","country":"Italy","region":"Europe"},
    {"name":"Amalfi Coast","country":"Italy","region":"Europe"},
    {"name":"Sicily","country":"Italy","region":"Europe"},
    {"name":"Palermo","country":"Italy","region":"Europe"},
    {"name":"Matera","country":"Italy","region":"Europe"},
    {"name":"Bologna","country":"Italy","region":"Europe"},
    {"name":"Turin","country":"Italy","region":"Europe"},
    {"name":"Genoa","country":"Italy","region":"Europe"},
    {"name":"Lake Garda","country":"Italy","region":"Europe"},
    {"name":"Dolomites","country":"Italy","region":"Europe"},
    {"name":"Scottish Highlands","country":"Scotland","region":"Europe"},
    {"name":"Stonehenge","country":"England","region":"Europe"},
    {"name":"Bath","country":"England","region":"Europe"},
    {"name":"Oxford","country":"England","region":"Europe"},
    {"name":"Cambridge","country":"England","region":"Europe"},
    {"name":"York","country":"England","region":"Europe"},
    {"name":"Canterbury","country":"England","region":"Europe"},
    {"name":"Stratford-upon-Avon","country":"England","region":"Europe"},
    {"name":"Lake District","country":"England","region":"Europe"},
    {"name":"Cotswolds","country":"England","region":"Europe"},
    {"name":"Windsor","country":"England","region":"Europe"},
    {"name":"Cardiff","country":"Wales","region":"Europe"},
    {"name":"Swansea","country":"Wales","region":"Europe"},
    {"name":"Belfast","country":"Northern Ireland","region":"Europe"},
    {"name":"Giant's Causeway","country":"Northern Ireland","region":"Europe"},
    {"name":"Galway","country":"Ireland","region":"Europe"},
    {"name":"Cork","country":"Ireland","region":"Europe"},
    {"name":"Cliffs of Moher","country":"Ireland","region":"Europe"},
    {"name":"Ring of Kerry","country":"Ireland","region":"Europe"},
    {"name":"Killarney","country":"Ireland","region":"Europe"},
    {"name":"Dingle Peninsula","country":"Ireland","region":"Europe"},
    {"name":"Rotterdam","country":"Netherlands","region":"Europe"},
    {"name":"Utrecht","country":"Netherlands","region":"Europe"},
    {"name":"The Hague","country":"Netherlands","region":"Europe"},
    {"name":"Maastricht","country":"Netherlands","region":"Europe"},
    {"name":"Ghent","country":"Belgium","region":"Europe"},
    {"name":"Antwerp","country":"Belgium","region":"Europe"},
    {"name":"Liege","country":"Belgium","region":"Europe"},
    {"name":"Luxembourg","country":"Luxembourg","region":"Europe"},
    {"name":"Bern","country":"Switzerland","region":"Europe"},
    {"name":"Basel","country":"Switzerland","region":"Europe"},
    {"name":"Lausanne","country":"Switzerland","region":"Europe"},
    {"name":"Interlaken","country":"Switzerland","region":"Europe"},
    {"name":"Zermatt","country":"Switzerland","region":"Europe"},
    {"name":"St Moritz","country":"Switzerland","region":"Europe"},
    {"name":"Jungfrau Region","country":"Switzerland","region":"Europe"},
    {"name":"Lake Geneva","country":"Switzerland","region":"Europe"},
    {"name":"Salzburg","country":"Austria","region":"Europe"},
    {"name":"Graz","country":"Austria","region":"Europe"},
    {"name":"Linz","country":"Austria","region":"Europe"},
    {"name":"Hallstatt","country":"Austria","region":"Europe"},
    {"name":"Tyrol","country":"Austria","region":"Europe"},
    {"name":"Voralberg","country":"Austria","region":"Europe"},
    {"name":"Bavarian Alps","country":"Germany","region":"Europe"},
    {"name":"Black Forest","country":"Germany","region":"Europe"},
    {"name":"Rhine Valley","country":"Germany","region":"Europe"},
    {"name":"Neuschwanstein","country":"Germany","region":"Europe"},
    {"name":"Heidelberg","country":"Germany","region":"Europe"},
    {"name":"Bremen","country":"Germany","region":"Europe"},
    {"name":"Stuttgart","country":"Germany","region":"Europe"},
    {"name":"Dusseldorf","country":"Germany","region":"Europe"},
    {"name":"Leipzig","country":"Germany","region":"Europe"},
    {"name":"Nuremberg","country":"Germany","region":"Europe"},
    {"name":"Weimar","country":"Germany","region":"Europe"},
    {"name":"Bamberg","country":"Germany","region":"Europe"},
    {"name":"Rothenburg","country":"Germany","region":"Europe"},
    {"name":"Fussen","country":"Germany","region":"Europe"},
    {"name":"Garmisch","country":"Germany","region":"Europe"},
    {"name":"Berchtesgaden","country":"Germany","region":"Europe"},
    {"name":"Rugen Island","country":"Germany","region":"Europe"},
    {"name":"Sylt","country":"Germany","region":"Europe"},
    {"name":"Usedom","country":"Germany","region":"Europe"},
    {"name":"Fehmarn","country":"Germany","region":"Europe"},
    {"name":"Heligoland","country":"Germany","region":"Europe"},
    {"name":"Baltic Coast","country":"Germany","region":"Europe"},
    {"name":"North Sea Coast","country":"Germany","region":"Europe"},

    # NORTH AMERICA
    {"name":"New York City","country":"USA","region":"North America"},
    {"name":"Los Angeles","country":"USA","region":"North America"},
    {"name":"Chicago","country":"USA","region":"North America"},
    {"name":"Miami","country":"USA","region":"North America"},
    {"name":"Las Vegas","country":"USA","region":"North America"},
    {"name":"San Francisco","country":"USA","region":"North America"},
    {"name":"Washington DC","country":"USA","region":"North America"},
    {"name":"Boston","country":"USA","region":"North America"},
    {"name":"Seattle","country":"USA","region":"North America"},
    {"name":"New Orleans","country":"USA","region":"North America"},
    {"name":"San Diego","country":"USA","region":"North America"},
    {"name":"Orlando","country":"USA","region":"North America"},
    {"name":"Honolulu","country":"USA","region":"North America"},
    {"name":"Phoenix","country":"USA","region":"North America"},
    {"name":"Philadelphia","country":"USA","region":"North America"},
    {"name":"Atlanta","country":"USA","region":"North America"},
    {"name":"Houston","country":"USA","region":"North America"},
    {"name":"Dallas","country":"USA","region":"North America"},
    {"name":"Denver","country":"USA","region":"North America"},
    {"name":"Nashville","country":"USA","region":"North America"},
    {"name":"Austin","country":"USA","region":"North America"},
    {"name":"Portland","country":"USA","region":"North America"},
    {"name":"San Antonio","country":"USA","region":"North America"},
    {"name":"Salt Lake City","country":"USA","region":"North America"},
    {"name":"Charleston","country":"USA","region":"North America"},
    {"name":"Savannah","country":"USA","region":"North America"},
    {"name":"Santa Fe","country":"USA","region":"North America"},
    {"name":"Sedona","country":"USA","region":"North America"},
    {"name":"Aspen","country":"USA","region":"North America"},
    {"name":"Vail","country":"USA","region":"North America"},
    {"name":"Park City","country":"USA","region":"North America"},
    {"name":"Jackson Hole","country":"USA","region":"North America"},
    {"name":"Lake Tahoe","country":"USA","region":"North America"},
    {"name":"Napa Valley","country":"USA","region":"North America"},
    {"name":"Sonoma","country":"USA","region":"North America"},
    {"name":"Monterey","country":"USA","region":"North America"},
    {"name":"Carmel","country":"USA","region":"North America"},
    {"name":"Santa Barbara","country":"USA","region":"North America"},
    {"name":"Malibu","country":"USA","region":"North America"},
    {"name":"Palm Springs","country":"USA","region":"North America"},
    {"name":"San Juan Islands","country":"USA","region":"North America"},
    {"name":"Key West","country":"USA","region":"North America"},
    {"name":"Maui","country":"USA","region":"North America"},
    {"name":"Kauai","country":"USA","region":"North America"},
    {"name":"Big Island","country":"USA","region":"North America"},
    {"name":"Oahu","country":"USA","region":"North America"},
    {"name":"Anchorage","country":"USA","region":"North America"},
    {"name":"Fairbanks","country":"USA","region":"North America"},
    {"name":"Juneau","country":"USA","region":"North America"},
    {"name":"Sitka","country":"USA","region":"North America"},
    {"name":"Ketchikan","country":"USA","region":"North America"},
    {"name":"Toronto","country":"Canada","region":"North America"},
    {"name":"Vancouver","country":"Canada","region":"North America"},
    {"name":"Montreal","country":"Canada","region":"North America"},
    {"name":"Quebec City","country":"Canada","region":"North America"},
    {"name":"Calgary","country":"Canada","region":"North America"},
    {"name":"Ottawa","country":"Canada","region":"North America"},
    {"name":"Victoria","country":"Canada","region":"North America"},
    {"name":"Banff","country":"Canada","region":"North America"},
    {"name":"Whistler","country":"Canada","region":"North America"},
    {"name":"Jasper","country":"Canada","region":"North America"},
    {"name":"Niagara Falls","country":"Canada","region":"North America"},
    {"name":"Halifax","country":"Canada","region":"North America"},
    {"name":"St John's","country":"Canada","region":"North America"},
    {"name":"Charlottetown","country":"Canada","region":"North America"},
    {"name":"Whitehorse","country":"Canada","region":"North America"},
    {"name":"Yellowknife","country":"Canada","region":"North America"},
    {"name":"Iqaluit","country":"Canada","region":"North America"},
    {"name":"Mexico City","country":"Mexico","region":"North America"},
    {"name":"Cancun","country":"Mexico","region":"North America"},
    {"name":"Playa del Carmen","country":"Mexico","region":"North America"},
    {"name":"Tulum","country":"Mexico","region":"North America"},
    {"name":"Cabo San Lucas","country":"Mexico","region":"North America"},
    {"name":"Puerto Vallarta","country":"Mexico","region":"North America"},
    {"name":"Guadalajara","country":"Mexico","region":"North America"},
    {"name":"Monterrey","country":"Mexico","region":"North America"},
    {"name":"Oaxaca","country":"Mexico","region":"North America"},
    {"name":"San Miguel de Allende","country":"Mexico","region":"North America"},
    {"name":"Guanajuato","country":"Mexico","region":"North America"},
    {"name":"Merida","country":"Mexico","region":"North America"},
    {"name":"Puebla","country":"Mexico","region":"North America"},
    {"name":"Havana","country":"Cuba","region":"North America"},
    {"name":"Varadero","country":"Cuba","region":"North America"},
    {"name":"Trinidad","country":"Cuba","region":"North America"},
    {"name":"Santiago de Cuba","country":"Cuba","region":"North America"},
    {"name":"Nassau","country":"Bahamas","region":"North America"},
    {"name":"Freeport","country":"Bahamas","region":"North America"},
    {"name":"Punta Cana","country":"Dominican Republic","region":"North America"},
    {"name":"Santo Domingo","country":"Dominican Republic","region":"North America"},
    {"name":"San Juan","country":"Puerto Rico","region":"North America"},
    {"name":"Kingston","country":"Jamaica","region":"North America"},
    {"name":"Montego Bay","country":"Jamaica","region":"North America"},
    {"name":"Ocho Rios","country":"Jamaica","region":"North America"},
    {"name":"Negril","country":"Jamaica","region":"North America"},
    {"name":"Bridgetown","country":"Barbados","region":"North America"},
    {"name":"St George's","country":"Grenada","region":"North America"},
    {"name":"Castries","country":"St Lucia","region":"North America"},
    {"name":"Basseterre","country":"St Kitts","region":"North America"},
    {"name":"Philipsburg","country":"St Maarten","region":"North America"},
    {"name":"Willemstad","country":"Curacao","region":"North America"},
    {"name":"Oranjestad","country":"Aruba","region":"North America"},
    {"name":"Belmopan","country":"Belize","region":"North America"},
    {"name":"Belize City","country":"Belize","region":"North America"},
    {"name":"San Jose","country":"Costa Rica","region":"North America"},
    {"name":"Liberia","country":"Costa Rica","region":"North America"},
    {"name":"Managua","country":"Nicaragua","region":"North America"},
    {"name":"Granada","country":"Nicaragua","region":"North America"},
    {"name":"San Salvador","country":"El Salvador","region":"North America"},
    {"name":"Tegucigalpa","country":"Honduras","region":"North America"},
    {"name":"Guatemala City","country":"Guatemala","region":"North America"},
    {"name":"Antigua Guatemala","country":"Guatemala","region":"North America"},
    {"name":"Panama City","country":"Panama","region":"North America"},
    {"name":"Bocas del Toro","country":"Panama","region":"North America"},

    # SOUTH AMERICA
    {"name":"Rio de Janeiro","country":"Brazil","region":"South America"},
    {"name":"Sao Paulo","country":"Brazil","region":"South America"},
    {"name":"Buenos Aires","country":"Argentina","region":"South America"},
    {"name":"Lima","country":"Peru","region":"South America"},
    {"name":"Bogota","country":"Colombia","region":"South America"},
    {"name":"Santiago","country":"Chile","region":"South America"},
    {"name":"Quito","country":"Ecuador","region":"South America"},
    {"name":"Caracas","country":"Venezuela","region":"South America"},
    {"name":"La Paz","country":"Bolivia","region":"South America"},
    {"name":"Montevideo","country":"Uruguay","region":"South America"},
    {"name":"Asuncion","country":"Paraguay","region":"South America"},
    {"name":"Georgetown","country":"Guyana","region":"South America"},
    {"name":"Paramaribo","country":"Suriname","region":"South America"},
    {"name":"Cayenne","country":"French Guiana","region":"South America"},
    {"name":"Salvador","country":"Brazil","region":"South America"},
    {"name":"Brasilia","country":"Brazil","region":"South America"},
    {"name":"Fortaleza","country":"Brazil","region":"South America"},
    {"name":"Recife","country":"Brazil","region":"South America"},
    {"name":"Manaus","country":"Brazil","region":"South America"},
    {"name":"Florianopolis","country":"Brazil","region":"South America"},
    {"name":"Cordoba","country":"Argentina","region":"South America"},
    {"name":"Mendoza","country":"Argentina","region":"South America"},
    {"name":"Bariloche","country":"Argentina","region":"South America"},
    {"name":"Salta","country":"Argentina","region":"South America"},
    {"name":"Iguazu Falls","country":"Argentina","region":"South America"},
    {"name":"Ushuaia","country":"Argentina","region":"South America"},
    {"name":"Cusco","country":"Peru","region":"South America"},
    {"name":"Machu Picchu","country":"Peru","region":"South America"},
    {"name":"Arequipa","country":"Peru","region":"South America"},
    {"name":"Trujillo","country":"Peru","region":"South America"},
    {"name":"Iquitos","country":"Peru","region":"South America"},
    {"name":"Medellin","country":"Colombia","region":"South America"},
    {"name":"Cartagena","country":"Colombia","region":"South America"},
    {"name":"Cali","country":"Colombia","region":"South America"},
    {"name":"Santa Marta","country":"Colombia","region":"South America"},
    {"name":"San Andres","country":"Colombia","region":"South America"},
    {"name":"Valparaiso","country":"Chile","region":"South America"},
    {"name":"Pucon","country":"Chile","region":"South America"},
    {"name":"San Pedro de Atacama","country":"Chile","region":"South America"},
    {"name":"Puerto Natales","country":"Chile","region":"South America"},
    {"name":"Easter Island","country":"Chile","region":"South America"},
    {"name":"Guayaquil","country":"Ecuador","region":"South America"},
    {"name":"Cuenca","country":"Ecuador","region":"South America"},
    {"name":"Galapagos Islands","country":"Ecuador","region":"South America"},
    {"name":"Banhos","country":"Ecuador","region":"South America"},
    {"name":"Mindo","country":"Ecuador","region":"South America"},
    {"name":"Santa Cruz","country":"Bolivia","region":"South America"},
    {"name":"Sucre","country":"Bolivia","region":"South America"},
    {"name":"Potosi","country":"Bolivia","region":"South America"},
    {"name":"Uyuni","country":"Bolivia","region":"South America"},
    {"name":"Punta del Este","country":"Uruguay","region":"South America"},
    {"name":"Colonia del Sacramento","country":"Uruguay","region":"South America"},
    {"name":"Ciudad del Este","country":"Paraguay","region":"South America"},
    {"name":"Encarnacion","country":"Paraguay","region":"South America"},
    {"name":"Angel Falls","country":"Venezuela","region":"South America"},
    {"name":"Margarita Island","country":"Venezuela","region":"South America"},
    {"name":"Los Roques","country":"Venezuela","region":"South America"},
    {"name":"Devil's Island","country":"French Guiana","region":"South America"},
    {"name":"Kaieteur Falls","country":"Guyana","region":"South America"},
    {"name":"Paramaribo","country":"Suriname","region":"South America"},

    # ASIA
    {"name":"Tokyo","country":"Japan","region":"Asia"},
    {"name":"Kyoto","country":"Japan","region":"Asia"},
    {"name":"Osaka","country":"Japan","region":"Asia"},
    {"name":"Hiroshima","country":"Japan","region":"Asia"},
    {"name":"Nagoya","country":"Japan","region":"Asia"},
    {"name":"Yokohama","country":"Japan","region":"Asia"},
    {"name":"Sapporo","country":"Japan","region":"Asia"},
    {"name":"Fukuoka","country":"Japan","region":"Asia"},
    {"name":"Nara","country":"Japan","region":"Asia"},
    {"name":"Kobe","country":"Japan","region":"Asia"},
    {"name":"Kanazawa","country":"Japan","region":"Asia"},
    {"name":"Hakone","country":"Japan","region":"Asia"},
    {"name":"Nikko","country":"Japan","region":"Asia"},
    {"name":"Kamakura","country":"Japan","region":"Asia"},
    {"name":"Takayama","country":"Japan","region":"Asia"},
    {"name":"Matsumoto","country":"Japan","region":"Asia"},
    {"name":"Beppu","country":"Japan","region":"Asia"},
    {"name":"Seoul","country":"South Korea","region":"Asia"},
    {"name":"Busan","country":"South Korea","region":"Asia"},
    {"name":"Jeju Island","country":"South Korea","region":"Asia"},
    {"name":"Incheon","country":"South Korea","region":"Asia"},
    {"name":"Daegu","country":"South Korea","region":"Asia"},
    {"name":"Gyeongju","country":"South Korea","region":"Asia"},
    {"name":"Beijing","country":"China","region":"Asia"},
    {"name":"Shanghai","country":"China","region":"Asia"},
    {"name":"Hong Kong","country":"China","region":"Asia"},
    {"name":"Guangzhou","country":"China","region":"Asia"},
    {"name":"Shenzhen","country":"China","region":"Asia"},
    {"name":"Chengdu","country":"China","region":"Asia"},
    {"name":"Xi'an","country":"China","region":"Asia"},
    {"name":"Hangzhou","country":"China","region":"Asia"},
    {"name":"Suzhou","country":"China","region":"Asia"},
    {"name":"Nanjing","country":"China","region":"Asia"},
    {"name":"Macau","country":"China","region":"Asia"},
    {"name":"Lhasa","country":"China","region":"Asia"},
    {"name":"Guilin","country":"China","region":"Asia"},
    {"name":"Kunming","country":"China","region":"Asia"},
    {"name":"Dali","country":"China","region":"Asia"},
    {"name":"Lijiang","country":"China","region":"Asia"},
    {"name":"Zhangjiajie","country":"China","region":"Asia"},
    {"name":"Harbin","country":"China","region":"Asia"},
    {"name":"Urumqi","country":"China","region":"Asia"},
    {"name":"Taipei","country":"Taiwan","region":"Asia"},
    {"name":"Kaohsiung","country":"Taiwan","region":"Asia"},
    {"name":"Taichung","country":"Taiwan","region":"Asia"},
    {"name":"Tainan","country":"Taiwan","region":"Asia"},
    {"name":"Hualien","country":"Taiwan","region":"Asia"},
    {"name":"Bangkok","country":"Thailand","region":"Asia"},
    {"name":"Phuket","country":"Thailand","region":"Asia"},
    {"name":"Chiang Mai","country":"Thailand","region":"Asia"},
    {"name":"Pattaya","country":"Thailand","region":"Asia"},
    {"name":"Krabi","country":"Thailand","region":"Asia"},
    {"name":"Koh Samui","country":"Thailand","region":"Asia"},
    {"name":"Ayutthaya","country":"Thailand","region":"Asia"},
    {"name":"Hua Hin","country":"Thailand","region":"Asia"},
    {"name":"Singapore","country":"Singapore","region":"Asia"},
    {"name":"Kuala Lumpur","country":"Malaysia","region":"Asia"},
    {"name":"Penang","country":"Malaysia","region":"Asia"},
    {"name":"Langkawi","country":"Malaysia","region":"Asia"},
    {"name":"Malacca","country":"Malaysia","region":"Asia"},
    {"name":"Kota Kinabalu","country":"Malaysia","region":"Asia"},
    {"name":"Kuching","country":"Malaysia","region":"Asia"},
    {"name":"Manila","country":"Philippines","region":"Asia"},
    {"name":"Cebu","country":"Philippines","region":"Asia"},
    {"name":"Boracay","country":"Philippines","region":"Asia"},
    {"name":"Palawan","country":"Philippines","region":"Asia"},
    {"name":"Bohol","country":"Philippines","region":"Asia"},
    {"name":"Davao","country":"Philippines","region":"Asia"},
    {"name":"Jakarta","country":"Indonesia","region":"Asia"},
    {"name":"Bali","country":"Indonesia","region":"Asia"},
    {"name":"Yogyakarta","country":"Indonesia","region":"Asia"},
    {"name":"Lombok","country":"Indonesia","region":"Asia"},
    {"name":"Komodo","country":"Indonesia","region":"Asia"},
    {"name":"Surabaya","country":"Indonesia","region":"Asia"},
    {"name":"Bandung","country":"Indonesia","region":"Asia"},
    {"name":"Medan","country":"Indonesia","region":"Asia"},
    {"name":"Hanoi","country":"Vietnam","region":"Asia"},
    {"name":"Ho Chi Minh City","country":"Vietnam","region":"Asia"},
    {"name":"Da Nang","country":"Vietnam","region":"Asia"},
    {"name":"Hue","country":"Vietnam","region":"Asia"},
    {"name":"Hoi An","country":"Vietnam","region":"Asia"},
    {"name":"Nha Trang","country":"Vietnam","region":"Asia"},
    {"name":"Ha Long Bay","country":"Vietnam","region":"Asia"},
    {"name":"Sapa","country":"Vietnam","region":"Asia"},
    {"name":"Phu Quoc","country":"Vietnam","region":"Asia"},
    {"name":"Phnom Penh","country":"Cambodia","region":"Asia"},
    {"name":"Siem Reap","country":"Cambodia","region":"Asia"},
    {"name":"Sihanoukville","country":"Cambodia","region":"Asia"},
    {"name":"Battambang","country":"Cambodia","region":"Asia"},
    {"name":"Vientiane","country":"Laos","region":"Asia"},
    {"name":"Luang Prabang","country":"Laos","region":"Asia"},
    {"name":"Vang Vieng","country":"Laos","region":"Asia"},
    {"name":"Yangon","country":"Myanmar","region":"Asia"},
    {"name":"Mandalay","country":"Myanmar","region":"Asia"},
    {"name":"Bagan","country":"Myanmar","region":"Asia"},
    {"name":"Inle Lake","country":"Myanmar","region":"Asia"},
    {"name":"New Delhi","country":"India","region":"Asia"},
    {"name":"Mumbai","country":"India","region":"Asia"},
    {"name":"Goa","country":"India","region":"Asia"},
    {"name":"Kerala","country":"India","region":"Asia"},
    {"name":"Rajasthan","country":"India","region":"Asia"},
    {"name":"Varanasi","country":"India","region":"Asia"},
    {"name":"Agra","country":"India","region":"Asia"},
    {"name":"Jaipur","country":"India","region":"Asia"},
    {"name":"Kolkata","country":"India","region":"Asia"},
    {"name":"Chennai","country":"India","region":"Asia"},
    {"name":"Hyderabad","country":"India","region":"Asia"},
    {"name":"Bangalore","country":"India","region":"Asia"},
    {"name":"Pune","country":"India","region":"Asia"},
    {"name":"Udaipur","country":"India","region":"Asia"},
    {"name":"Jodhpur","country":"India","region":"Asia"},
    {"name":"Jaisalmer","country":"India","region":"Asia"},
    {"name":"Amritsar","country":"India","region":"Asia"},
    {"name":"Shimla","country":"India","region":"Asia"},
    {"name":"Darjeeling","country":"India","region":"Asia"},
    {"name":"Leh","country":"India","region":"Asia"},
    {"name":"Kathmandu","country":"Nepal","region":"Asia"},
    {"name":"Pokhara","country":"Nepal","region":"Asia"},
    {"name":"Chitwan","country":"Nepal","region":"Asia"},
    {"name":"Lumbini","country":"Nepal","region":"Asia"},
    {"name":"Thimphu","country":"Bhutan","region":"Asia"},
    {"name":"Paro","country":"Bhutan","region":"Asia"},
    {"name":"Punakha","country":"Bhutan","region":"Asia"},
    {"name":"Colombo","country":"Sri Lanka","region":"Asia"},
    {"name":"Kandy","country":"Sri Lanka","region":"Asia"},
    {"name":"Galle","country":"Sri Lanka","region":"Asia"},
    {"name":"Sigiriya","country":"Sri Lanka","region":"Asia"},
    {"name":"Male","country":"Maldives","region":"Asia"},
    {"name":"Dhaka","country":"Bangladesh","region":"Asia"},
    {"name":"Chittagong","country":"Bangladesh","region":"Asia"},
    {"name":"Cox's Bazar","country":"Bangladesh","region":"Asia"},
    {"name":"Islamabad","country":"Pakistan","region":"Asia"},
    {"name":"Karachi","country":"Pakistan","region":"Asia"},
    {"name":"Lahore","country":"Pakistan","region":"Asia"},
    {"name":"Rawalpindi","country":"Pakistan","region":"Asia"},
    {"name":"Peshawar","country":"Pakistan","region":"Asia"},
    {"name":"Multan","country":"Pakistan","region":"Asia"},
    {"name":"Quetta","country":"Pakistan","region":"Asia"},
    {"name":"Gilgit","country":"Pakistan","region":"Asia"},
    {"name":"Skardu","country":"Pakistan","region":"Asia"},
    {"name":"Hunza Valley","country":"Pakistan","region":"Asia"},
    {"name":"Fairy Meadows","country":"Pakistan","region":"Asia"},
    {"name":"Kashmir","country":"Pakistan","region":"Asia"},

    # MIDDLE EAST
    {"name":"Dubai","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Abu Dhabi","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Sharjah","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Ras Al Khaimah","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Doha","country":"Qatar","region":"Middle East"},
    {"name":"Manama","country":"Bahrain","region":"Middle East"},
    {"name":"Kuwait City","country":"Kuwait","region":"Middle East"},
    {"name":"Muscat","country":"Oman","region":"Middle East"},
    {"name":"Salalah","country":"Oman","region":"Middle East"},
    {"name":"Riyadh","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Jeddah","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Mecca","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Medina","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Dammam","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Sana'a","country":"Yemen","region":"Middle East"},
    {"name":"Aden","country":"Yemen","region":"Middle East"},
    {"name":"Tehran","country":"Iran","region":"Middle East"},
    {"name":"Mashhad","country":"Iran","region":"Middle East"},
    {"name":"Isfahan","country":"Iran","region":"Middle East"},
    {"name":"Shiraz","country":"Iran","region":"Middle East"},
    {"name":"Tabriz","country":"Iran","region":"Middle East"},
    {"name":"Yazd","country":"Iran","region":"Middle East"},
    {"name":"Baghdad","country":"Iraq","region":"Middle East"},
    {"name":"Basra","country":"Iraq","region":"Middle East"},
    {"name":"Erbil","country":"Iraq","region":"Middle East"},
    {"name":"Amman","country":"Jordan","region":"Middle East"},
    {"name":"Petra","country":"Jordan","region":"Middle East"},
    {"name":"Aqaba","country":"Jordan","region":"Middle East"},
    {"name":"Jerash","country":"Jordan","region":"Middle East"},
    {"name":"Wadi Rum","country":"Jordan","region":"Middle East"},
    {"name":"Beirut","country":"Lebanon","region":"Middle East"},
    {"name":"Byblos","country":"Lebanon","region":"Middle East"},
    {"name":"Baalbek","country":"Lebanon","region":"Middle East"},
    {"name":"Damascus","country":"Syria","region":"Middle East"},
    {"name":"Aleppo","country":"Syria","region":"Middle East"},
    {"name":"Palmyra","country":"Syria","region":"Middle East"},
    {"name":"Bethlehem","country":"Palestine","region":"Middle East"},
    {"name":"Ramallah","country":"Palestine","region":"Middle East"},
    {"name":"Gaza","country":"Palestine","region":"Middle East"},

    # AFRICA
    {"name":"Cape Town","country":"South Africa","region":"Africa"},
    {"name":"Johannesburg","country":"South Africa","region":"Africa"},
    {"name":"Durban","country":"South Africa","region":"Africa"},
    {"name":"Pretoria","country":"South Africa","region":"Africa"},
    {"name":"Port Elizabeth","country":"South Africa","region":"Africa"},
    {"name":"Bloemfontein","country":"South Africa","region":"Africa"},
    {"name":"Kruger National Park","country":"South Africa","region":"Africa"},
    {"name":"Garden Route","country":"South Africa","region":"Africa"},
    {"name":"Winelands","country":"South Africa","region":"Africa"},
    {"name":"Sun City","country":"South Africa","region":"Africa"},
    {"name":"Marrakech","country":"Morocco","region":"Africa"},
    {"name":"Casablanca","country":"Morocco","region":"Africa"},
    {"name":"Fez","country":"Morocco","region":"Africa"},
    {"name":"Tangier","country":"Morocco","region":"Africa"},
    {"name":"Rabat","country":"Morocco","region":"Africa"},
    {"name":"Essaouira","country":"Morocco","region":"Africa"},
    {"name":"Chefchaouen","country":"Morocco","region":"Africa"},
    {"name":"Agadir","country":"Morocco","region":"Africa"},
    {"name":"Cairo","country":"Egypt","region":"Africa"},
    {"name":"Alexandria","country":"Egypt","region":"Africa"},
    {"name":"Luxor","country":"Egypt","region":"Africa"},
    {"name":"Aswan","country":"Egypt","region":"Africa"},
    {"name":"Sharm el Sheikh","country":"Egypt","region":"Africa"},
    {"name":"Hurghada","country":"Egypt","region":"Africa"},
    {"name":"Giza","country":"Egypt","region":"Africa"},
    {"name":"Dahab","country":"Egypt","region":"Africa"},
    {"name":"Nairobi","country":"Kenya","region":"Africa"},
    {"name":"Mombasa","country":"Kenya","region":"Africa"},
    {"name":"Maasai Mara","country":"Kenya","region":"Africa"},
    {"name":"Amboseli","country":"Kenya","region":"Africa"},
    {"name":"Tsavo","country":"Kenya","region":"Africa"},
    {"name":"Lake Nakuru","country":"Kenya","region":"Africa"},
    {"name":"Lamu Island","country":"Kenya","region":"Africa"},
    {"name":"Dar es Salaam","country":"Tanzania","region":"Africa"},
    {"name":"Zanzibar","country":"Tanzania","region":"Africa"},
    {"name":"Arusha","country":"Tanzania","region":"Africa"},
    {"name":"Serengeti","country":"Tanzania","region":"Africa"},
    {"name":"Ngorongoro","country":"Tanzania","region":"Africa"},
    {"name":"Kilimanjaro","country":"Tanzania","region":"Africa"},
    {"name":"Stone Town","country":"Tanzania","region":"Africa"},
    {"name":"Kampala","country":"Uganda","region":"Africa"},
    {"name":"Entebbe","country":"Uganda","region":"Africa"},
    {"name":"Jinja","country":"Uganda","region":"Africa"},
    {"name":"Kigali","country":"Rwanda","region":"Africa"},
    {"name":"Addis Ababa","country":"Ethiopia","region":"Africa"},
    {"name":"Lalibela","country":"Ethiopia","region":"Africa"},
    {"name":"Axum","country":"Ethiopia","region":"Africa"},
    {"name":"Gondar","country":"Ethiopia","region":"Africa"},
    {"name":"Bahir Dar","country":"Ethiopia","region":"Africa"},
    {"name":"Djibouti City","country":"Djibouti","region":"Africa"},
    {"name":"Mogadishu","country":"Somalia","region":"Africa"},
    {"name":"Hargeisa","country":"Somalia","region":"Africa"},
    {"name":"Accra","country":"Ghana","region":"Africa"},
    {"name":"Kumasi","country":"Ghana","region":"Africa"},
    {"name":"Cape Coast","country":"Ghana","region":"Africa"},
    {"name":"Lagos","country":"Nigeria","region":"Africa"},
    {"name":"Abuja","country":"Nigeria","region":"Africa"},
    {"name":"Port Harcourt","country":"Nigeria","region":"Africa"},
    {"name":"Kano","country":"Nigeria","region":"Africa"},
    {"name":"Dakar","country":"Senegal","region":"Africa"},
    {"name":"Saint Louis","country":"Senegal","region":"Africa"},
    {"name":"Bamako","country":"Mali","region":"Africa"},
    {"name":"Timbuktu","country":"Mali","region":"Africa"},
    {"name":"Ouagadougou","country":"Burkina Faso","region":"Africa"},
    {"name":"Abidjan","country":"Ivory Coast","region":"Africa"},
    {"name":"Yamoussoukro","country":"Ivory Coast","region":"Africa"},
    {"name":"Monrovia","country":"Liberia","region":"Africa"},
    {"name":"Freetown","country":"Sierra Leone","region":"Africa"},
    {"name":"Conakry","country":"Guinea","region":"Africa"},
    {"name":"Bissau","country":"Guinea-Bissau","region":"Africa"},
    {"name":"Praia","country":"Cape Verde","region":"Africa"},
    {"name":"Mindelo","country":"Cape Verde","region":"Africa"},
    {"name":"Luanda","country":"Angola","region":"Africa"},
    {"name":"Lobito","country":"Angola","region":"Africa"},
    {"name":"Windhoek","country":"Namibia","region":"Africa"},
    {"name":"Swakopmund","country":"Namibia","region":"Africa"},
    {"name":"Walvis Bay","country":"Namibia","region":"Africa"},
    {"name":"Etosha National Park","country":"Namibia","region":"Africa"},
    {"name":"Sossusvlei","country":"Namibia","region":"Africa"},
    {"name":"Gaborone","country":"Botswana","region":"Africa"},
    {"name":"Maun","country":"Botswana","region":"Africa"},
    {"name":"Okavango Delta","country":"Botswana","region":"Africa"},
    {"name":"Chobe National Park","country":"Botswana","region":"Africa"},
    {"name":"Kalahari Desert","country":"Botswana","region":"Africa"},
    {"name":"Harare","country":"Zimbabwe","region":"Africa"},
    {"name":"Bulawayo","country":"Zimbabwe","region":"Africa"},
    {"name":"Victoria Falls","country":"Zimbabwe","region":"Africa"},
    {"name":"Great Zimbabwe","country":"Zimbabwe","region":"Africa"},
    {"name":"Lusaka","country":"Zambia","region":"Africa"},
    {"name":"Livingstone","country":"Zambia","region":"Africa"},
    {"name":"Lake Kariba","country":"Zambia","region":"Africa"},
    {"name":"South Luangwa","country":"Zambia","region":"Africa"},
    {"name":"Lilongwe","country":"Malawi","region":"Africa"},
    {"name":"Blantyre","country":"Malawi","region":"Africa"},
    {"name":"Lake Malawi","country":"Malawi","region":"Africa"},
    {"name":"Maputo","country":"Mozambique","region":"Africa"},
    {"name":"Beira","country":"Mozambique","region":"Africa"},
    {"name":"Bazaruto Archipelago","country":"Mozambique","region":"Africa"},
    {"name":"Quelimane","country":"Mozambique","region":"Africa"},
    {"name":"Nampula","country":"Mozambique","region":"Africa"},
    {"name":"Antananarivo","country":"Madagascar","region":"Africa"},
    {"name":"Nosy Be","country":"Madagascar","region":"Africa"},
    {"name":"Morondava","country":"Madagascar","region":"Africa"},
    {"name":"Fort Dauphin","country":"Madagascar","region":"Africa"},
    {"name":"Saint Denis","country":"Reunion","region":"Africa"},
    {"name":"Mauritius","country":"Mauritius","region":"Africa"},
    {"name":"Port Louis","country":"Mauritius","region":"Africa"},
    {"name":"Seychelles","country":"Seychelles","region":"Africa"},
    {"name":"Victoria","country":"Seychelles","region":"Africa"},
    {"name":"Comoros","country":"Comoros","region":"Africa"},
    {"name":"Moroni","country":"Comoros","region":"Africa"},

    # OCEANIA
    {"name":"Sydney","country":"Australia","region":"Oceania"},
    {"name":"Melbourne","country":"Australia","region":"Oceania"},
    {"name":"Brisbane","country":"Australia","region":"Oceania"},
    {"name":"Perth","country":"Australia","region":"Oceania"},
    {"name":"Adelaide","country":"Australia","region":"Oceania"},
    {"name":"Gold Coast","country":"Australia","region":"Oceania"},
    {"name":"Cairns","country":"Australia","region":"Oceania"},
    {"name":"Darwin","country":"Australia","region":"Oceania"},
    {"name":"Hobart","country":"Australia","region":"Oceania"},
    {"name":"Canberra","country":"Australia","region":"Oceania"},
    {"name":"Alice Springs","country":"Australia","region":"Oceania"},
    {"name":"Uluru","country":"Australia","region":"Oceania"},
    {"name":"Great Barrier Reef","country":"Australia","region":"Oceania"},
    {"name":"Whitsunday Islands","country":"Australia","region":"Oceania"},
    {"name":"Fraser Island","country":"Australia","region":"Oceania"},
    {"name":"Kangaroo Island","country":"Australia","region":"Oceania"},
    {"name":"Tasmania","country":"Australia","region":"Oceania"},
    {"name":"Blue Mountains","country":"Australia","region":"Oceania"},
    {"name":"Hunter Valley","country":"Australia","region":"Oceania"},
    {"name":"Margaret River","country":"Australia","region":"Oceania"},
    {"name":"Barossa Valley","country":"Australia","region":"Oceania"},
    {"name":"Yarra Valley","country":"Australia","region":"Oceania"},
    {"name":"Auckland","country":"New Zealand","region":"Oceania"},
    {"name":"Wellington","country":"New Zealand","region":"Oceania"},
    {"name":"Christchurch","country":"New Zealand","region":"Oceania"},
    {"name":"Queenstown","country":"New Zealand","region":"Oceania"},
    {"name":"Rotorua","country":"New Zealand","region":"Oceania"},
    {"name":"Dunedin","country":"New Zealand","region":"Oceania"},
    {"name":"Napier","country":"New Zealand","region":"Oceania"},
    {"name":"Nelson","country":"New Zealand","region":"Oceania"},
    {"name":"Taupo","country":"New Zealand","region":"Oceania"},
    {"name":"Milford Sound","country":"New Zealand","region":"Oceania"},
    {"name":"Bay of Islands","country":"New Zealand","region":"Oceania"},
    {"name":"Coromandel Peninsula","country":"New Zealand","region":"Oceania"},
    {"name":"Fiordland","country":"New Zealand","region":"Oceania"},
    {"name":"Abel Tasman","country":"New Zealand","region":"Oceania"},
    {"name":"Tongariro","country":"New Zealand","region":"Oceania"},
    {"name":"Mount Cook","country":"New Zealand","region":"Oceania"},
    {"name":"Wanaka","country":"New Zealand","region":"Oceania"},
    {"name":"Te Anau","country":"New Zealand","region":"Oceania"},
    {"name":"Paihia","country":"New Zealand","region":"Oceania"},
    {"name":"Suva","country":"Fiji","region":"Oceania"},
    {"name":"Nadi","country":"Fiji","region":"Oceania"},
    {"name":"Denarau Island","country":"Fiji","region":"Oceania"},
    {"name":"Mamanuca Islands","country":"Fiji","region":"Oceania"},
    {"name":"Yasawa Islands","country":"Fiji","region":"Oceania"},
    {"name":"Port Vila","country":"Vanuatu","region":"Oceania"},
    {"name":"Luganville","country":"Vanuatu","region":"Oceania"},
    {"name":"Noumea","country":"New Caledonia","region":"Oceania"},
    {"name":"Honiara","country":"Solomon Islands","region":"Oceania"},
    {"name":"Port Moresby","country":"Papua New Guinea","region":"Oceania"},
    {"name":"Lae","country":"Papua New Guinea","region":"Oceania"},
    {"name":"Madang","country":"Papua New Guinea","region":"Oceania"},
    {"name":"Rabaul","country":"Papua New Guinea","region":"Oceania"},
    {"name":"Goroka","country":"Papua New Guinea","region":"Oceania"},
    {"name":"Mount Hagen","country":"Papua New Guinea","region":"Oceania"},
    {"name":"Apia","country":"Samoa","region":"Oceania"},
    {"name":"Pago Pago","country":"American Samoa","region":"Oceania"},
    {"name":"Nuku'alofa","country":"Tonga","region":"Oceania"},
    {"name":"Tarawa","country":"Kiribati","region":"Oceania"},
    {"name":"Funafuti","country":"Tuvalu","region":"Oceania"},
    {"name":"Majuro","country":"Marshall Islands","region":"Oceania"},
    {"name":"Palikir","country":"Micronesia","region":"Oceania"},
    {"name":"Yaren","country":"Nauru","region":"Oceania"},
    {"name":"South Tarawa","country":"Kiribati","region":"Oceania"},
]

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)
data_provider = CityDataProvider()

# Global variable to store all cities data
ALL_CITIES_DATA: List[Dict[str, Any]] = []
CITIES_LOADED = False
CITIES_LOADING = False

def load_all_cities():
    """Load all cities data in the background"""
    global ALL_CITIES_DATA, CITIES_LOADED, CITIES_LOADING

    if CITIES_LOADED or CITIES_LOADING:
        return

    CITIES_LOADING = True
    logger.info(f"Starting to load all {len(WORLD_CITIES)} cities...")

    def load_task():
        global ALL_CITIES_DATA, CITIES_LOADED, CITIES_LOADING

        try:
            # Use more workers to load faster
            with ThreadPoolExecutor(max_workers=min(16, max(2, len(WORLD_CITIES)))) as executor:
                futures = {executor.submit(data_provider.get_city_preview, c["name"]): c for c in WORLD_CITIES}
                loaded_data = []

                for i, future in enumerate(as_completed(futures)):
                    city = futures[future]
                    try:
                        d = future.result(timeout=45)  # Increased timeout for image fetching
                        d["region"] = city.get("region")
                        d["country"] = city.get("country")
                        # Ensure map exists: if no static_map, attempt to create one from coordinates or placeholder
                        coords = d.get("coordinates")
                        if not d.get("static_map"):
                            if coords:
                                d["static_map"] = data_provider.map_provider.generate_static_map_url(coords, width=400, height=200)
                            else:
                                d["static_map"] = data_provider.map_provider.generate_static_map_url(None, width=400, height=200)
                        loaded_data.append(d)

                        # Log progress every 20 cities to see image loading
                        if (i + 1) % 20 == 0:
                            loaded_with_images = sum(1 for city in loaded_data if city.get("image", {}).get("url") != "/static/default_city.jpg")
                            logger.info(f"Loaded {i + 1}/{len(WORLD_CITIES)} cities ({loaded_with_images} with images)")

                    except Exception as e:
                        logger.warning(f"Failed to get preview for {city['name']}: {e}")
                        # Add basic city info even if preview fails
                        fallback = {
                            "id": data_provider._generate_city_id(city["name"]),
                            "name": city["name"],
                            "display_name": city["name"],
                            "summary": "Information available",
                            "has_details": False,
                            "region": city["region"],
                            "country": city["country"],
                            "image": {"url": "/static/default_city.jpg", "source": "placeholder"},
                        }
                        # Add tagline via provider
                        try:
                            tag = data_provider.get_city_tagline(city["name"])
                            fallback["tagline"] = tag.get("tagline")
                            fallback["tagline_source"] = tag.get("source")
                        except Exception:
                            fallback["tagline"] = "A beautiful city worth exploring"
                            fallback["tagline_source"] = "default"

                        fallback["static_map"] = data_provider.map_provider.generate_static_map_url(None, width=400, height=200)
                        loaded_data.append(fallback)

            ALL_CITIES_DATA = loaded_data
            CITIES_LOADED = True
            CITIES_LOADING = False

            # Log final image statistics
            cities_with_images = sum(1 for city in ALL_CITIES_DATA if city.get("image", {}).get("url") != "/static/default_city.jpg")
            logger.info(f"Successfully loaded all {len(ALL_CITIES_DATA)} cities ({cities_with_images} with proper images)")

        except Exception as e:
            logger.error(f"Failed to load cities: {e}")
            CITIES_LOADING = False

    # Start loading in background thread
    threading.Thread(target=load_task, daemon=True).start()

def preload_popular_cities():
    """Preload only the most popular cities initially"""
    def task():
        logger.info("Preloading popular cities...")
        top_cities = WORLD_CITIES[:config.PRELOAD_TOP_CITIES]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(data_provider.get_city_preview, c["name"]): c["name"] for c in top_cities}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=45)
                    has_image = result.get("image", {}).get("url") != "/static/default_city.jpg"
                    logger.info(f"Preloaded {result['name']} - Image: {'Yes' if has_image else 'No'}")
                except Exception as e:
                    logger.warning(f"Preload failed for city: {e}")
        logger.info("Popular cities preloaded")
        # Start loading all cities after popular ones are done
        load_all_cities()
    threading.Thread(target=task, daemon=True).start()

@app.route("/")
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/api/health")
def health():
    cities_with_images = 0
    if CITIES_LOADED:
        cities_with_images = sum(1 for city in ALL_CITIES_DATA if city.get("image", {}).get("url") != "/static/default_city.jpg")

    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "cities_loaded": CITIES_LOADED,
        "cities_loading": CITIES_LOADING,
        "total_cities": len(WORLD_CITIES),
        "cities_with_images": cities_with_images
    })

@app.route("/api/image-stats")
def get_image_stats():
    """Debug endpoint to check image loading statistics"""
    if not CITIES_LOADED:
        return jsonify({"error": "Cities still loading"}), 423

    cities_with_images = sum(1 for city in ALL_CITIES_DATA
                           if city.get("image", {}).get("url") != "/static/default_city.jpg")
    cities_without_images = len(ALL_CITIES_DATA) - cities_with_images

    # Get sample of cities without images
    no_image_cities = [city["name"] for city in ALL_CITIES_DATA
                      if city.get("image", {}).get("url") == "/static/default_city.jpg"][:10]

    return jsonify({
        "total_cities": len(ALL_CITIES_DATA),
        "cities_with_images": cities_with_images,
        "cities_without_images": cities_without_images,
        "coverage_percentage": round((cities_with_images / len(ALL_CITIES_DATA)) * 100, 2) if ALL_CITIES_DATA else 0.0,
        "sample_no_image_cities": no_image_cities
    })

@app.route("/api/debug-images/<path:city_name>")
def debug_images(city_name):
    """Debug endpoint to check image fetching for a specific city"""
    from urllib.parse import unquote
    city_name = unquote(city_name)

    # Test Wikipedia page lookup
    page_data, title = data_provider.fetch_wikipedia_page(city_name)
    wiki_status = "Found" if page_data else "Not found"

    # Test image fetching
    images = []
    if title:
        images = data_provider.get_wikimedia_images(title, limit=6)

    return jsonify({
        "city": city_name,
        "wikipedia_page": wiki_status,
        "wikipedia_title": title,
        "images_found": len(images),
        "image_urls": [img.get("url") for img in images],
        "current_preview_image": LOCAL_CITY_CACHE.get(city_name, {}).get("image", {})
    })

@app.route("/api/debug-map/<path:city_name>")
def debug_map(city_name):
    """Debug endpoint to check map configuration for a specific city"""
    from urllib.parse import unquote
    city_name = unquote(city_name)

    # Test coordinate fetching
    coords_data = data_provider.get_coordinates(city_name)
    coordinates = None
    if coords_data:
        coordinates = {"lat": coords_data[0], "lon": coords_data[1]}

    # Test map configuration
    map_config = data_provider.map_provider.get_map_config(city_name, coordinates)

    # Test static map URL
    static_map_url = data_provider.map_provider.generate_static_map_url(coordinates)

    return jsonify({
        "city": city_name,
        "coordinates_found": coords_data is not None,
        "coordinates": coordinates,
        "coordinates_validation": data_provider.map_provider._validate_coordinates(coordinates) if coordinates else False,
        "map_config": map_config,
        "static_map_url": static_map_url,
        "tile_provider": config.MAP_TILE_PROVIDER,
        "has_mapbox_token": bool(config.MAPBOX_ACCESS_TOKEN)
    })

@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    city_name = request.json.get("city") if request.json else None
    cities_to_clear = [city_name] if city_name else None

    clear_city_cache(cities_to_clear)

    if city_name:
        return jsonify({"success": True, "message": f"Cache cleared for {city_name}"})
    else:
        return jsonify({"success": True, "message": "All cache cleared"})

@app.route("/api/reload-city", methods=["POST"])
def reload_city():
    city_name = request.json.get("city") if request.json else None
    if not city_name:
        return jsonify({"success": False, "error": "City name required"}), 400

    # Clear cache for this city
    clear_city_cache([city_name])

    # Force reload
    preview = data_provider.get_city_preview(city_name)
    details = data_provider.get_city_details(city_name)

    return jsonify({
        "success": True,
        "message": f"Reloaded {city_name}",
        "preview": preview,
        "details": details
    })

@app.route("/api/cities")
def get_cities():
    # If all cities are loaded, return them all at once
    if CITIES_LOADED:
        logger.info(f"Returning all {len(ALL_CITIES_DATA)} loaded cities")
        return jsonify({
            "success": True,
            "data": ALL_CITIES_DATA,
            "pagination": {
                "page": 1,
                "limit": len(ALL_CITIES_DATA),
                "total": len(ALL_CITIES_DATA),
                "pages": 1
            }
        })

    # If still loading, return what we have so far + remaining basic cities
    if CITIES_LOADING:
        current_data = ALL_CITIES_DATA.copy()
        loaded_names = {city["name"] for city in current_data}

        # Add basic info for cities not yet loaded
        for city in WORLD_CITIES:
            if city["name"] not in loaded_names:
                # Try to include tagline right away via provider (fast because cached)
                tagline_obj = {}
                try:
                    tagline_obj = data_provider.get_city_tagline(city["name"])
                except Exception:
                    tagline_obj = {"tagline": "A beautiful city worth exploring", "source": "default"}

                current_data.append({
                    "id": data_provider._generate_city_id(city["name"]),
                    "name": city["name"],
                    "display_name": city["name"],
                    "summary": "Loading...",
                    "has_details": False,
                    "region": city["region"],
                    "country": city["country"],
                    "image": {"url": "/static/default_city.jpg", "source": "placeholder"},
                    "tagline": tagline_obj.get("tagline"),
                    "tagline_source": tagline_obj.get("source"),
                    "static_map": data_provider.map_provider.generate_static_map_url(None, width=400, height=200)
                })

        logger.info(f"Returning {len(current_data)} cities (loading in progress)")
        return jsonify({
            "success": True,
            "data": current_data,
            "loading": True,
            "pagination": {
                "page": 1,
                "limit": len(current_data),
                "total": len(current_data),
                "pages": 1
            }
        })

    # If not started loading yet, start loading and return basic cities
    load_all_cities()
    basic_data = []
    for city in WORLD_CITIES:
        tagline_obj = {}
        try:
            tagline_obj = data_provider.get_city_tagline(city["name"])
        except Exception:
            tagline_obj = {"tagline": "A beautiful city worth exploring", "source": "default"}

        basic_data.append({
            "id": data_provider._generate_city_id(city["name"]),
            "name": city["name"],
            "display_name": city["name"],
            "summary": "Loading...",
            "has_details": False,
            "region": city["region"],
            "country": city["country"],
            "image": {"url": "/static/default_city.jpg", "source": "placeholder"},
            "tagline": tagline_obj.get("tagline"),
            "tagline_source": tagline_obj.get("source"),
            "static_map": data_provider.map_provider.generate_static_map_url(None, width=400, height=200)
        })

    logger.info(f"Returning {len(basic_data)} basic cities (starting load)")
    return jsonify({
        "success": True,
        "data": basic_data,
        "loading": True,
        "pagination": {
            "page": 1,
            "limit": len(basic_data),
            "total": len(basic_data),
            "pages": 1
        }
    })

@app.route("/api/cities/<path:city_name>")
def get_city(city_name):
    from urllib.parse import unquote
    city_name = unquote(city_name)
    exact = next((c for c in WORLD_CITIES if c["name"].lower() == city_name.lower()), None)
    if not exact:
        return jsonify({"success": False, "error": "City not found"}), 404

    details = data_provider.get_city_details(exact["name"])
    details["region"] = exact["region"]
    details["country"] = exact["country"]
    return jsonify({"success": True, "data": details})

@app.route("/api/city-desc/<path:city_name>")
def get_city_desc(city_name):
    """Return a short tagline-style description for a given city."""
    from urllib.parse import unquote
    city_name = unquote(city_name)

    try:
        result = data_provider.get_city_tagline(city_name)
        # Try to enrich with region/country info if available
        city_info = next((c for c in WORLD_CITIES if c["name"].lower() == city_name.lower()), None)
        if city_info:
            result["country"] = city_info.get("country")
            result["region"] = city_info.get("region")
        return jsonify(result)
    except Exception as e:
        logger.error(f"City tagline fetch failed for {city_name}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/search")
def search_cities():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify({"success": False, "error": "Query too short"}), 400

    # Search through loaded cities first, then fallback to basic search
    matches = []
    if CITIES_LOADED:
        matches = [c for c in ALL_CITIES_DATA if q in c["name"].lower()][:10]
    else:
        city_matches = [c for c in WORLD_CITIES if q in c["name"].lower()][:10]
        with ThreadPoolExecutor(max_workers=min(4, len(city_matches))) as executor:
            futures = {executor.submit(data_provider.get_city_preview, c["name"]): c for c in city_matches}
            for future in as_completed(futures):
                city = futures[future]
                try:
                    d = future.result(timeout=45)
                    d["region"] = city["region"]
                    d["country"] = city["country"]
                    matches.append(d)
                except Exception as e:
                    logger.warning(f"Search failed for {city['name']}: {e}")

    return jsonify({"success": True, "data": matches})

@app.route("/api/cities-list")
def get_cities_list():
    return jsonify({"success": True, "data": WORLD_CITIES})

@app.route("/api/regions")
def get_regions():
    return jsonify({"success": True, "data": REGIONS})

@app.route("/api/map/config")
def get_map_config():
    city_name = request.args.get("city", "")
    coords = None
    if city_name:
        c = data_provider.get_coordinates(city_name)
        if c:
            coords = {"lat": c[0], "lon": c[1]}
    map_conf = data_provider.map_provider.get_map_config(city_name, coords)
    return jsonify({"success": True, "data": map_conf})

# Debug middleware to log API requests
@app.after_request
def after_request(response):
    if request.path.startswith('/api/'):
        logger.info(f"API {request.method} {request.path} - Status: {response.status_code}")
    return response

if __name__ == "__main__":
    logger.info(f"Starting Flask server with {len(WORLD_CITIES)} cities")
    preload_popular_cities()
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True)
