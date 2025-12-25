import os
import re
import json
import time
import logging
import threading
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import requests
import wikipediaapi
import diskcache
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dataclasses import dataclass
from urllib.parse import quote_plus, unquote
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -------------------- ENHANCED CONFIG --------------------
@dataclass
class Config:
    UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours
    MAX_IMAGE_WORKERS: int = int(os.getenv("MAX_IMAGE_WORKERS", "16"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    FLASK_PORT: int = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    MAPBOX_ACCESS_TOKEN: str = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    MAP_TILE_PROVIDER: str = os.getenv("MAP_TILE_PROVIDER", "openstreetmap")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/diskcache")
    PRELOAD_TOP_CITIES: int = int(os.getenv("PRELOAD_TOP_CITIES", "50"))
    LOCAL_CACHE_FILE: str = os.getenv("LOCAL_CACHE_FILE", "/tmp/cities_data.json")
    MAX_WIKIMEDIA_FILES_TO_SCAN: int = 30
    MAX_IMAGES_PER_REQUEST: int = 8
    WIKIMEDIA_RETRY_ATTEMPTS: int = 5
    ENABLE_FALLBACK_IMAGES: bool = os.getenv("ENABLE_FALLBACK_IMAGES", "True").lower() == "true"
    FALLBACK_IMAGE_SERVICE: str = os.getenv("FALLBACK_IMAGE_SERVICE", "pixabay")  # "pixabay", "pexels", "unsplash"
    PIXABAY_API_KEY: str = os.getenv("PIXABAY_API_KEY", "")
    PEXELS_API_KEY: str = os.getenv("PEXELS_API_KEY", "")
    USE_PERSISTENT_CACHE: bool = os.getenv("USE_PERSISTENT_CACHE", "True").lower() == "true"
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))

config = Config()

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/city_api.log')
    ]
)
logger = logging.getLogger(__name__)

# -------------------- ENHANCED CACHE --------------------
class EnhancedCache:
    def __init__(self):
        self.cache = None
        self.local_cache = {}
        self.load_local_cache()
        self.init_disk_cache()
    
    def load_local_cache(self):
        """Load local cache from file"""
        try:
            if os.path.exists(config.LOCAL_CACHE_FILE):
                with open(config.LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
                    self.local_cache = json.load(f)
                logger.info(f"Loaded {len(self.local_cache)} cities from local cache")
        except Exception as e:
            logger.warning(f"Failed to load local cache: {e}")
            self.local_cache = {}
    
    def save_local_cache(self):
        """Save local cache to file"""
        try:
            os.makedirs(os.path.dirname(config.LOCAL_CACHE_FILE), exist_ok=True)
            with open(config.LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.local_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.local_cache)} cities to local cache")
        except Exception as e:
            logger.error(f"Failed to save local cache: {e}")
    
    def init_disk_cache(self):
        """Initialize disk cache"""
        try:
            if config.USE_PERSISTENT_CACHE:
                cache_dir = config.CACHE_DIR
                os.makedirs(cache_dir, exist_ok=True)
                self.cache = diskcache.Cache(cache_dir)
                logger.info(f"Disk cache initialized at: {cache_dir}")
            else:
                self.cache = diskcache.Cache()  # Memory cache
                logger.info("Using in-memory cache")
        except Exception as e:
            logger.warning(f"Failed to initialize disk cache, using memory: {e}")
            self.cache = diskcache.Cache()
    
    def get(self, city_name):
        """Get city data from cache"""
        return self.local_cache.get(city_name)
    
    def set(self, city_name, data):
        """Set city data in cache"""
        self.local_cache[city_name] = data
        self.save_local_cache()
    
    def clear(self, city_names=None):
        """Clear cache for specific cities or all"""
        if city_names is None:
            self.local_cache.clear()
            if self.cache:
                self.cache.clear()
            logger.info("Cleared all cache")
        else:
            for city_name in city_names:
                if city_name in self.local_cache:
                    del self.local_cache[city_name]
            logger.info(f"Cleared cache for {len(city_names)} cities")
        self.save_local_cache()

cache_manager = EnhancedCache()

# -------------------- IMAGE FALLBACK PROVIDER --------------------
class ImageFallbackProvider:
    def __init__(self):
        self.services = []
        
        # Configure available services
        if config.PIXABAY_API_KEY:
            self.services.append(self.pixabay_fallback)
        
        if config.PEXELS_API_KEY:
            self.services.append(self.pexels_fallback)
        
        if config.UNSPLASH_ACCESS_KEY:
            self.services.append(self.unsplash_fallback)
        
        # Always include Wikipedia as fallback
        self.services.append(self.wikipedia_fallback)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def pixabay_fallback(self, query):
        """Fetch images from Pixabay"""
        try:
            url = "https://pixabay.com/api/"
            params = {
                "key": config.PIXABAY_API_KEY,
                "q": f"{query} city",
                "image_type": "photo",
                "orientation": "horizontal",
                "per_page": 5,
                "safesearch": True
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                images = []
                for hit in data.get("hits", [])[:3]:
                    images.append({
                        "url": hit.get("webformatURL") or hit.get("largeImageURL"),
                        "title": f"{query} city",
                        "description": f"Cityscape of {query}",
                        "source": "pixabay",
                        "width": hit.get("webformatWidth"),
                        "height": hit.get("webformatHeight"),
                        "photographer": hit.get("user", "Unknown")
                    })
                return images if images else None
        except Exception as e:
            logger.debug(f"Pixabay fallback failed for {query}: {e}")
        return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def pexels_fallback(self, query):
        """Fetch images from Pexels"""
        try:
            url = f"https://api.pexels.com/v1/search"
            headers = {"Authorization": config.PEXELS_API_KEY}
            params = {"query": f"{query} city", "per_page": 5, "orientation": "landscape"}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                images = []
                for photo in data.get("photos", [])[:3]:
                    images.append({
                        "url": photo.get("src", {}).get("large") or photo.get("src", {}).get("medium"),
                        "title": f"{query} city",
                        "description": f"Cityscape of {query}",
                        "source": "pexels",
                        "width": photo.get("width"),
                        "height": photo.get("height"),
                        "photographer": photo.get("photographer", "Unknown")
                    })
                return images if images else None
        except Exception as e:
            logger.debug(f"Pexels fallback failed for {query}: {e}")
        return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def unsplash_fallback(self, query):
        """Fetch images from Unsplash"""
        try:
            url = "https://api.unsplash.com/search/photos"
            headers = {"Authorization": f"Client-ID {config.UNSPLASH_ACCESS_KEY}"}
            params = {"query": f"{query} city", "per_page": 5, "orientation": "landscape"}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                images = []
                for result in data.get("results", [])[:3]:
                    images.append({
                        "url": result.get("urls", {}).get("regular"),
                        "title": f"{query} city",
                        "description": result.get("description") or f"Cityscape of {query}",
                        "source": "unsplash",
                        "width": result.get("width"),
                        "height": result.get("height"),
                        "photographer": result.get("user", {}).get("name", "Unknown")
                    })
                return images if images else None
        except Exception as e:
            logger.debug(f"Unsplash fallback failed for {query}: {e}")
        return None
    
    def wikipedia_fallback(self, query):
        """Use Wikipedia commons search as ultimate fallback"""
        try:
            # Try Wikimedia Commons API
            url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"{query} city",
                "gsrnamespace": "6",
                "gsrlimit": 5,
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": 800,
                "format": "json"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                images = []
                for page in data.get("query", {}).get("pages", {}).values():
                    iinfo = page.get("imageinfo", [{}])[0]
                    if iinfo.get("url"):
                        images.append({
                            "url": iinfo.get("url"),
                            "title": page.get("title", "").replace("File:", ""),
                            "description": f"Image of {query} from Wikimedia Commons",
                            "source": "wikimedia_commons",
                            "width": iinfo.get("width"),
                            "height": iinfo.get("height")
                        })
                return images[:3] if images else None
        except Exception as e:
            logger.debug(f"Wikimedia Commons fallback failed: {e}")
        return None
    
    def get_fallback_images(self, city_name):
        """Try multiple fallback services until we get images"""
        if not config.ENABLE_FALLBACK_IMAGES:
            return None
        
        for service in self.services:
            try:
                images = service(city_name)
                if images:
                    logger.info(f"Found {len(images)} fallback images for {city_name} via {service.__name__}")
                    return images
            except Exception as e:
                logger.debug(f"Fallback service {service.__name__} failed: {e}")
                continue
        
        return None

# -------------------- MAP PROVIDER --------------------
class MapProvider:
    def __init__(self):
        self.tile_providers = {
            "openstreetmap": {
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attribution": "© OpenStreetMap contributors"
            },
            "carto": {
                "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                "attribution": "© OpenStreetMap & CARTO"
            }
        }
        
        if config.MAPBOX_ACCESS_TOKEN:
            self.tile_providers["mapbox"] = {
                "url": f"https://api.mapbox.com/styles/v1/mapbox/light-v10/tiles/{{z}}/{{x}}/{{y}}?access_token={config.MAPBOX_ACCESS_TOKEN}",
                "attribution": "© Mapbox & OpenStreetMap"
            }
    
    def get_map_config(self, city_name, coordinates=None):
        provider_key = config.MAP_TILE_PROVIDER if config.MAP_TILE_PROVIDER in self.tile_providers else "openstreetmap"
        provider_config = self.tile_providers[provider_key]
        
        map_config = {
            "tile_provider": provider_key,
            "tile_url": provider_config["url"],
            "attribution": provider_config["attribution"],
            "zoom": 12,
            "min_zoom": 2,
            "max_zoom": 18,
            "center": {"lat": 0, "lon": 0},
            "marker": None
        }
        
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
    
    def generate_static_map_url(self, coordinates, width=600, height=400):
        if not coordinates or not self._validate_coordinates(coordinates):
            # Generate generic world map as fallback
            return "https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}.png?api_key=YOUR_KEY"
        
        lat, lon = coordinates["lat"], coordinates["lon"]
        
        # OpenStreetMap static map via staticmap.org
        try:
            zoom = 10 if abs(lat) < 30 else 8  # Adjust zoom based on latitude
            return f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={width}x{height}&markers={lat},{lon},lightblue"
        except Exception:
            pass
        
        return None
    
    def _validate_coordinates(self, coordinates):
        try:
            lat = float(coordinates.get("lat", 0))
            lon = float(coordinates.get("lon", 0))
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except (TypeError, ValueError):
            return False

# -------------------- ENHANCED CITY DATA PROVIDER --------------------
class EnhancedCityDataProvider:
    def __init__(self):
        user_agent = "TravelTTO/1.0 (https://www.traveltto.com; contact@traveltto.com)"
        self.geolocator = Nominatim(
            user_agent=user_agent,
            timeout=config.REQUEST_TIMEOUT
        )
        # Add rate limiting
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
        
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent=user_agent,
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        self.map_provider = MapProvider()
        self.image_fallback = ImageFallbackProvider()
        
        # City name normalization cache
        self.city_aliases = self._load_city_aliases()
    
    def _load_city_aliases(self):
        """Load common city name variations"""
        return {
            "nyc": "New York",
            "new york city": "New York",
            "la": "Los Angeles",
            "sf": "San Francisco",
            "dc": "Washington DC",
            "washington d.c.": "Washington DC",
            "london uk": "London",
            "paris france": "Paris",
            "tokyo japan": "Tokyo",
            # Add more aliases as needed
        }
    
    def normalize_city_name(self, city_name):
        """Normalize city name with aliases"""
        normalized = city_name.strip()
        
        # Check aliases
        if normalized.lower() in self.city_aliases:
            return self.city_aliases[normalized.lower()]
        
        # Clean up common issues
        if normalized.endswith(","):
            normalized = normalized[:-1].strip()
        
        # Capitalize properly
        words = normalized.split()
        capitalized_words = []
        for word in words:
            if word.lower() in ['and', 'or', 'the', 'of', 'de', 'la', 'el']:
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())
        
        return ' '.join(capitalized_words)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_coordinates(self, city_name):
        """Get coordinates with multiple fallback strategies"""
        normalized_name = self.normalize_city_name(city_name)
        
        # Strategy 1: Try from world cities list first
        for city in WORLD_CITIES:
            if city["name"].lower() == normalized_name.lower():
                if "lat" in city and "lon" in city:
                    logger.info(f"Found coordinates for {normalized_name} from city list")
                    return (city["lat"], city["lon"], {"source": "predefined"})
        
        # Strategy 2: Try with country
        country = self._get_country_for_city(normalized_name)
        if country:
            try:
                query = f"{normalized_name}, {country}"
                location = self.geocode(query, exactly_one=True, addressdetails=True)
                if location:
                    logger.info(f"Found coordinates for {normalized_name} with country")
                    return (location.latitude, location.longitude, location.raw)
            except Exception as e:
                logger.debug(f"Geocoding with country failed: {e}")
        
        # Strategy 3: Try without country
        try:
            location = self.geocode(normalized_name, exactly_one=True, addressdetails=True)
            if location:
                logger.info(f"Found coordinates for {normalized_name} without country")
                return (location.latitude, location.longitude, location.raw)
        except Exception as e:
            logger.debug(f"Geocoding without country failed: {e}")
        
        # Strategy 4: Try Wikipedia API
        try:
            coords = self._get_wikidata_coordinates(normalized_name)
            if coords:
                logger.info(f"Found coordinates for {normalized_name} from Wikidata")
                return (coords["lat"], coords["lon"], {"source": "wikidata"})
        except Exception as e:
            logger.debug(f"Wikidata coordinates failed: {e}")
        
        logger.warning(f"No coordinates found for {normalized_name}")
        return None
    
    def _get_wikidata_coordinates(self, city_name):
        """Get coordinates from Wikidata"""
        try:
            # Search for entity
            search_url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "search": city_name,
                "language": "en",
                "format": "json",
                "type": "item"
            }
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for result in data.get("search", []):
                    entity_id = result.get("id")
                    
                    # Get coordinates for this entity
                    entity_url = "https://www.wikidata.org/wiki/Special:EntityData/"
                    entity_response = requests.get(f"{entity_url}{entity_id}.json", timeout=10)
                    
                    if entity_response.status_code == 200:
                        entity_data = entity_response.json()
                        claims = entity_data.get("entities", {}).get(entity_id, {}).get("claims", {})
                        
                        # Look for coordinate claims (P625)
                        if "P625" in claims:
                            coords = claims["P625"][0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
                            if coords:
                                return {
                                    "lat": coords.get("latitude"),
                                    "lon": coords.get("longitude")
                                }
        except Exception as e:
            logger.debug(f"Wikidata coordinate fetch failed: {e}")
        
        return None
    
    def _get_country_for_city(self, city_name):
        """Get country name for city"""
        for city in WORLD_CITIES:
            if city["name"].lower() == city_name.lower():
                return city.get("country")
        return None
    
    def get_wikipedia_data(self, city_name):
        """Get Wikipedia data with improved error handling"""
        try:
            normalized_name = self.normalize_city_name(city_name)
            
            # Try exact match
            page = self.wiki.page(normalized_name)
            if page.exists() and page.ns == 0:
                return self._extract_wikipedia_data(page)
            
            # Try with common suffixes
            variations = [
                f"{normalized_name} (city)",
                f"{normalized_name}, {self._get_country_for_city(normalized_name)}",
                f"{normalized_name} City",
                normalized_name.split(',')[0].strip()
            ]
            
            for variation in variations:
                if not variation:
                    continue
                page = self.wiki.page(variation)
                if page.exists() and page.ns == 0:
                    return self._extract_wikipedia_data(page)
            
            # Try search
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": f"{normalized_name} city",
                "format": "json"
            }
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for result in data.get("query", {}).get("search", [])[:3]:
                    page = self.wiki.page(result.get("title"))
                    if page.exists() and page.ns == 0:
                        return self._extract_wikipedia_data(page)
        
        except Exception as e:
            logger.warning(f"Wikipedia fetch failed for {city_name}: {e}")
        
        return None
    
    def _extract_wikipedia_data(self, page):
        """Extract structured data from Wikipedia page"""
        sections = {}
        
        def extract_sections(section, depth=0):
            if depth > 2:  # Limit depth
                return
            
            title = section.title.strip()
            text = section.text.strip()
            
            if title and text and depth <= 2:
                # Clean text
                text = re.sub(r'\[\d+\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if title not in ["See also", "References", "External links", "Notes"]:
                    sections[title] = text[:500]  # Limit length
            
            for subsection in section.sections:
                extract_sections(subsection, depth + 1)
        
        # Extract main sections
        for section in page.sections:
            extract_sections(section)
        
        return {
            "title": page.title,
            "summary": page.summary[:1000] if page.summary else "",
            "url": page.fullurl,
            "sections": sections,
            "categories": list(page.categories.keys())[:5] if hasattr(page, 'categories') else []
        }
    
    def get_images(self, city_name, limit=6):
        """Get images from multiple sources with fallbacks"""
        normalized_name = self.normalize_city_name(city_name)
        images = []
        
        # Try Wikipedia images first
        wiki_data = self.get_wikipedia_data(normalized_name)
        if wiki_data:
            wiki_images = self._get_wikipedia_images(wiki_data["title"], limit)
            images.extend(wiki_images[:limit])
        
        # If not enough images, try fallback services
        if len(images) < limit and config.ENABLE_FALLBACK_IMAGES:
            fallback_images = self.image_fallback.get_fallback_images(normalized_name)
            if fallback_images:
                images.extend(fallback_images[:limit - len(images)])
        
        # Ensure unique images
        seen_urls = set()
        unique_images = []
        for img in images:
            if img.get("url") and img["url"] not in seen_urls:
                seen_urls.add(img["url"])
                unique_images.append(img)
        
        # Add default if still empty
        if not unique_images:
            unique_images.append({
                "url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                "title": "Cityscape",
                "description": "Beautiful city view",
                "source": "default",
                "width": 800,
                "height": 600
            })
        
        return unique_images[:limit]
    
    def _get_wikipedia_images(self, page_title, limit):
        """Get images from Wikipedia page"""
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": page_title,
                "prop": "pageimages|images",
                "pithumbsize": 800,
                "imlimit": 20,
                "format": "json",
                "piprop": "thumbnail|name"
            }
            
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                images = []
                
                for page in pages.values():
                    # Main page image
                    if "thumbnail" in page:
                        thumb = page["thumbnail"]
                        if thumb.get("source"):
                            images.append({
                                "url": thumb["source"],
                                "title": f"Main image of {page_title}",
                                "description": f"Featured image",
                                "source": "wikipedia",
                                "width": thumb.get("width", 800),
                                "height": thumb.get("height", 600)
                            })
                    
                    # Additional images
                    for img in page.get("images", [])[:10]:
                        if img["title"].startswith("File:"):
                            # Get image info
                            img_params = {
                                "action": "query",
                                "titles": img["title"],
                                "prop": "imageinfo",
                                "iiprop": "url|size|mime",
                                "iiurlwidth": 800,
                                "format": "json"
                            }
                            
                            img_response = requests.get(api_url, params=img_params, timeout=10)
                            if img_response.status_code == 200:
                                img_data = img_response.json()
                                img_pages = img_data.get("query", {}).get("pages", {})
                                for img_page in img_pages.values():
                                    iinfo = img_page.get("imageinfo", [{}])[0]
                                    if iinfo.get("url"):
                                        images.append({
                                            "url": iinfo["url"],
                                            "title": img_page["title"].replace("File:", ""),
                                            "description": f"Image from {page_title}",
                                            "source": "wikipedia",
                                            "width": iinfo.get("width"),
                                            "height": iinfo.get("height")
                                        })
                    
                    if len(images) >= limit:
                        break
                
                return images[:limit]
        
        except Exception as e:
            logger.warning(f"Wikipedia image fetch failed: {e}")
        
        return []
    
    def get_city_tagline(self, city_name):
        """Get a catchy tagline for the city"""
        taglines = {
            "Paris": "The City of Light",
            "New York": "The City That Never Sleeps",
            "London": "The Big Smoke",
            "Tokyo": "The Eastern Capital",
            "Rome": "The Eternal City",
            "Dubai": "The City of Gold",
            "Venice": "The Floating City",
            "Amsterdam": "The Venice of the North",
            "Barcelona": "The City of Gaudí",
            "Istanbul": "Where East Meets West",
            "Rio de Janeiro": "The Marvelous City",
            "Sydney": "The Harbour City",
            "Cape Town": "The Mother City",
            "Bangkok": "The City of Angels",
            "Moscow": "The Third Rome",
            "Beijing": "The Northern Capital",
            "Seoul": "The Morning Calm",
            "Mumbai": "The City of Dreams",
            "Los Angeles": "The City of Angels",
            "Chicago": "The Windy City",
            "San Francisco": "The Golden City",
            "Las Vegas": "Sin City",
            "Miami": "The Magic City",
            "New Orleans": "The Big Easy",
            "Boston": "The Cradle of Liberty"
        }
        
        normalized_name = self.normalize_city_name(city_name)
        
        # Check exact match
        if normalized_name in taglines:
            return taglines[normalized_name]
        
        # Check partial match
        for key, value in taglines.items():
            if key.lower() in normalized_name.lower():
                return value
        
        # Generate from Wikipedia summary
        try:
            wiki_data = self.get_wikipedia_data(normalized_name)
            if wiki_data and wiki_data.get("summary"):
                summary = wiki_data["summary"]
                # Take first sentence or first 10 words
                sentences = summary.split('. ')
                if sentences:
                    first_sentence = sentences[0]
                    words = first_sentence.split()[:8]
                    return ' '.join(words) + '...'
        except Exception as e:
            logger.debug(f"Tagline generation failed: {e}")
        
        # Default tagline
        return "A beautiful destination worth exploring"
    
    def get_city_preview(self, city_name):
        """Get quick preview of city"""
        normalized_name = self.normalize_city_name(city_name)
        
        # Check cache first
        cached = cache_manager.get(normalized_name)
        if cached:
            return cached
        
        # Get basic info
        coords = self.get_coordinates(normalized_name)
        tagline = self.get_city_tagline(normalized_name)
        images = self.get_images(normalized_name, 1)
        
        # Construct preview
        preview = {
            "id": hashlib.md5(normalized_name.encode()).hexdigest()[:8],
            "name": normalized_name,
            "display_name": normalized_name,
            "tagline": tagline,
            "coordinates": {"lat": coords[0], "lon": coords[1]} if coords else None,
            "image": images[0] if images else {
                "url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
                "title": "City View",
                "source": "default"
            },
            "static_map": self.map_provider.generate_static_map_url(
                {"lat": coords[0], "lon": coords[1]} if coords else None,
                width=400, height=200
            ),
            "has_details": True
        }
        
        # Add region and country
        for city in WORLD_CITIES:
            if city["name"].lower() == normalized_name.lower():
                preview["region"] = city.get("region", "Unknown")
                preview["country"] = city.get("country", "Unknown")
                break
        
        # Cache the result
        cache_manager.set(normalized_name, preview)
        
        return preview
    
    def get_city_details(self, city_name):
        """Get detailed city information"""
        normalized_name = self.normalize_city_name(city_name)
        
        # Get all data
        coords = self.get_coordinates(normalized_name)
        wiki_data = self.get_wikipedia_data(normalized_name)
        images = self.get_images(normalized_name, config.MAX_IMAGES_PER_REQUEST)
        tagline = self.get_city_tagline(normalized_name)
        
        # Construct details
        details = {
            "id": hashlib.md5(normalized_name.encode()).hexdigest()[:8],
            "name": normalized_name,
            "coordinates": {"lat": coords[0], "lon": coords[1]} if coords else None,
            "map": self.map_provider.get_map_config(normalized_name, 
                   {"lat": coords[0], "lon": coords[1]} if coords else None),
            "static_map": self.map_provider.generate_static_map_url(
                {"lat": coords[0], "lon": coords[1]} if coords else None,
                width=800, height=400
            ),
            "images": images,
            "tagline": tagline,
            "summary": wiki_data.get("summary", "")[:1500] if wiki_data else "",
            "wikipedia_url": wiki_data.get("url") if wiki_data else "",
            "sections": [],
            "last_updated": time.time()
        }
        
        # Add sections from Wikipedia
        if wiki_data and wiki_data.get("sections"):
            for title, content in wiki_data["sections"].items():
                details["sections"].append({
                    "title": title,
                    "content": content[:1000]
                })
        
        # Add region and country
        for city in WORLD_CITIES:
            if city["name"].lower() == normalized_name.lower():
                details["region"] = city.get("region", "Unknown")
                details["country"] = city.get("country", "Unknown")
                break
        
        return details

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

REGIONS = ["Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"]

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)

# Initialize data provider
data_provider = EnhancedCityDataProvider()

# Global state
ALL_CITIES_DATA = []
CITIES_LOADED = False
CITIES_LOADING = False

def load_all_cities_parallel():
    """Load all cities in parallel with progress tracking"""
    global ALL_CITIES_DATA, CITIES_LOADED, CITIES_LOADING
    
    if CITIES_LOADED or CITIES_LOADING:
        return
    
    CITIES_LOADING = True
    logger.info(f"Starting parallel load of {len(WORLD_CITIES)} cities...")
    
    def load_batch(cities_batch):
        """Load a batch of cities"""
        batch_results = []
        with ThreadPoolExecutor(max_workers=min(config.MAX_CONCURRENT_REQUESTS, len(cities_batch))) as executor:
            futures = {executor.submit(data_provider.get_city_preview, city["name"]): city for city in cities_batch}
            
            for future in as_completed(futures):
                city = futures[future]
                try:
                    preview = future.result(timeout=30)
                    # Add region and country
                    preview["region"] = city.get("region", "Unknown")
                    preview["country"] = city.get("country", "Unknown")
                    batch_results.append(preview)
                except Exception as e:
                    logger.warning(f"Failed to load {city['name']}: {e}")
                    # Create minimal preview
                    batch_results.append({
                        "id": hashlib.md5(city['name'].encode()).hexdigest()[:8],
                        "name": city["name"],
                        "display_name": city["name"],
                        "tagline": "Loading...",
                        "image": {"url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?ixlib=rb-4.0.3&auto=format&fit=crop&w-400&q=80", "source": "default"},
                        "region": city.get("region", "Unknown"),
                        "country": city.get("country", "Unknown"),
                        "has_details": False
                    })
        
        return batch_results
    
    try:
        # Split cities into batches
        total_cities = len(WORLD_CITIES)
        batch_size = config.BATCH_SIZE
        all_results = []
        
        for i in range(0, total_cities, batch_size):
            batch = WORLD_CITIES[i:i + batch_size]
            logger.info(f"Loading batch {i//batch_size + 1}/{(total_cities + batch_size - 1)//batch_size}")
            
            batch_results = load_batch(batch)
            all_results.extend(batch_results)
            
            # Log progress
            cities_with_images = sum(1 for c in all_results if c.get("image", {}).get("source") != "default")
            logger.info(f"Progress: {len(all_results)}/{total_cities} cities loaded ({cities_with_images} with images)")
        
        ALL_CITIES_DATA = all_results
        CITIES_LOADED = True
        CITIES_LOADING = False
        
        # Final statistics
        cities_with_images = sum(1 for c in ALL_CITIES_DATA if c.get("image", {}).get("source") != "default")
        logger.info(f"✅ Loaded all {len(ALL_CITIES_DATA)} cities ({cities_with_images} with proper images)")
        
    except Exception as e:
        logger.error(f"Failed to load cities: {e}")
        CITIES_LOADING = False

def preload_popular_cities():
    """Preload top cities for quick initial display"""
    def task():
        logger.info("Preloading popular cities...")
        top_cities = WORLD_CITIES[:min(config.PRELOAD_TOP_CITIES, len(WORLD_CITIES))]
        
        for city in top_cities:
            try:
                preview = data_provider.get_city_preview(city["name"])
                preview["region"] = city.get("region", "Unknown")
                preview["country"] = city.get("country", "Unknown")
                
                # Add to global data
                if not any(c["name"] == preview["name"] for c in ALL_CITIES_DATA):
                    ALL_CITIES_DATA.append(preview)
                
                logger.debug(f"Preloaded {city['name']}")
            except Exception as e:
                logger.warning(f"Preload failed for {city['name']}: {e}")
        
        logger.info(f"Preloaded {len(top_cities)} popular cities")
        
        # Start full load in background
        load_all_cities_parallel()
    
    threading.Thread(target=task, daemon=True).start()

# -------------------- API ROUTES --------------------
@app.route("/")
def home():
    return jsonify({
        "message": "TravelTTO City API",
        "status": "online",
        "version": "2.0",
        "endpoints": {
            "/api/cities": "Get all cities",
            "/api/cities/<name>": "Get city details",
            "/api/search": "Search cities",
            "/api/regions": "Get all regions",
            "/api/health": "Health check",
            "/api/stats": "Get statistics"
        },
        "frontend": "https://www.traveltto.com"
    })

@app.route("/api/health")
def health():
    cities_with_images = 0
    if CITIES_LOADED:
        cities_with_images = sum(1 for c in ALL_CITIES_DATA if c.get("image", {}).get("source") != "default")
    
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "cities": {
            "total": len(WORLD_CITIES),
            "loaded": CITIES_LOADED,
            "loading": CITIES_LOADING,
            "in_cache": len(ALL_CITIES_DATA),
            "with_images": cities_with_images
        },
        "cache": {
            "size": len(cache_manager.local_cache),
            "path": config.LOCAL_CACHE_FILE
        }
    })

@app.route("/api/cities")
def get_cities():
    # Pagination parameters
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 100))
    region = request.args.get("region", "").strip()
    
    # Start loading if not already started
    if not CITIES_LOADING and not CITIES_LOADED:
        preload_popular_cities()
    
    # Filter by region if specified
    if region:
        filtered = [c for c in ALL_CITIES_DATA if c.get("region", "").lower() == region.lower()]
    else:
        filtered = ALL_CITIES_DATA
    
    # Apply pagination
    total = len(filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated = filtered[start_idx:end_idx]
    
    return jsonify({
        "success": True,
        "data": paginated,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit,
            "has_more": end_idx < total
        },
        "loading": not CITIES_LOADED,
        "filters": {
            "region": region if region else None
        }
    })

@app.route("/api/cities/<path:city_name>")
def get_city(city_name):
    decoded_name = unquote(city_name)
    
    # Check if city exists in our list
    city_exists = any(c["name"].lower() == decoded_name.lower() for c in WORLD_CITIES)
    if not city_exists:
        return jsonify({
            "success": False,
            "error": "City not found",
            "suggestions": [c["name"] for c in WORLD_CITIES if decoded_name.lower() in c["name"].lower()][:5]
        }), 404
    
    # Get detailed information
    try:
        details = data_provider.get_city_details(decoded_name)
        
        # Add region and country
        for city in WORLD_CITIES:
            if city["name"].lower() == decoded_name.lower():
                details["region"] = city.get("region", "Unknown")
                details["country"] = city.get("country", "Unknown")
                break
        
        return jsonify({
            "success": True,
            "data": details
        })
    
    except Exception as e:
        logger.error(f"Error fetching city {decoded_name}: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch city details",
            "message": str(e)
        }), 500

@app.route("/api/search")
def search_cities():
    query = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 20))
    
    if len(query) < 2:
        return jsonify({"success": False, "error": "Query must be at least 2 characters"}), 400
    
    # Search in loaded cities
    matches = []
    if CITIES_LOADED:
        matches = [c for c in ALL_CITIES_DATA if query in c["name"].lower() or query in c.get("tagline", "").lower()]
    else:
        # Search in world cities list
        for city in WORLD_CITIES:
            if query in city["name"].lower():
                try:
                    preview = data_provider.get_city_preview(city["name"])
                    preview["region"] = city.get("region", "Unknown")
                    preview["country"] = city.get("country", "Unknown")
                    matches.append(preview)
                except Exception:
                    continue
    
    # Sort by relevance (exact matches first)
    matches.sort(key=lambda x: (
        0 if query == x["name"].lower() else 1 if query in x["name"].lower() else 2
    ))
    
    return jsonify({
        "success": True,
        "data": matches[:limit],
        "total": len(matches),
        "query": query
    })

@app.route("/api/regions")
def get_regions():
    return jsonify({
        "success": True,
        "data": REGIONS,
        "count": len(REGIONS)
    })

@app.route("/api/stats")
def get_stats():
    cities_with_images = 0
    cities_with_coords = 0
    
    if CITIES_LOADED:
        cities_with_images = sum(1 for c in ALL_CITIES_DATA if c.get("image", {}).get("source") != "default")
        cities_with_coords = sum(1 for c in ALL_CITIES_DATA if c.get("coordinates"))
    
    return jsonify({
        "success": True,
        "data": {
            "total_cities": len(WORLD_CITIES),
            "loaded_cities": len(ALL_CITIES_DATA),
            "cities_with_images": cities_with_images,
            "cities_with_coords": cities_with_coords,
            "cities_without_images": len(WORLD_CITIES) - cities_with_images,
            "regions": len(REGIONS),
            "cache_size": len(cache_manager.local_cache)
        }
    })

@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    # Optional: Add authentication in production
    cache_manager.clear()
    return jsonify({
        "success": True,
        "message": "Cache cleared successfully"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# For local development
if __name__ == "__main__":
    port = config.FLASK_PORT
    logger.info(f"🚀 Starting TravelTTO API with {len(WORLD_CITIES)} cities on port {port}")
    preload_popular_cities()
    app.run(host="0.0.0.0", port=port, debug=config.FLASK_DEBUG, threaded=True)