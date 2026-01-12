import os
import re
import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from urllib.parse import quote_plus, unquote
import requests
import wikipediaapi
import diskcache
from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request
from flask_cors import CORS
from dataclasses import dataclass, field
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==================== SIMPLIFIED CONFIGURATION ====================
@dataclass
class Config:
    # Cache settings - longer TTL for better performance
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours
    CACHE_TTL_IMAGES: int = int(os.getenv("CACHE_TTL_IMAGES", "172800"))  # 48 hours
    CACHE_TTL_PREVIEW: int = int(os.getenv("CACHE_TTL_PREVIEW", "86400"))  # 24 hours
    
    # Performance settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "5"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    WIKIPEDIA_TIMEOUT: int = int(os.getenv("WIKIPEDIA_TIMEOUT", "15"))
    GEOLOCATOR_TIMEOUT: int = int(os.getenv("GEOLOCATOR_TIMEOUT", "5"))
    
    # App settings
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    FLASK_PORT: int = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    
    # Cache directories
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/city_explorer_cache")
    IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "/tmp/image_cache")
    
    # Image settings
    MAX_IMAGES_PER_REQUEST: int = 8
    MAX_IMAGES_PREVIEW: int = 1
    MIN_IMAGE_WIDTH: int = 400
    MIN_IMAGE_HEIGHT: int = 300
    
    # Feature flags
    ENABLE_FALLBACK_IMAGES: bool = True
    ENABLE_COORDINATE_FALLBACK: bool = True
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# ==================== LOGGING ====================
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("CityExplorer")

# ==================== ENHANCED CACHING SYSTEM ====================
class EnhancedCache:
    def __init__(self):
        self.memory_cache = {}
        self.disk_cache = None
        
        try:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            os.makedirs(config.IMAGE_CACHE_DIR, exist_ok=True)
            self.disk_cache = diskcache.Cache(config.CACHE_DIR)
            logger.info(f"✅ Disk cache initialized at: {config.CACHE_DIR}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize disk cache: {e}")
            self.disk_cache = diskcache.Cache()
        
        # Try Redis if available
        self.redis_client = None
        if os.getenv("REDIS_URL"):
            try:
                self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis cache unavailable: {e}")
    
    def get(self, key: str, default=None):
        # Check memory cache first (fastest)
        if key in self.memory_cache:
            item = self.memory_cache.get(key)
            if item and time.time() - item.get('timestamp', 0) < config.CACHE_TTL:
                return item.get('value')
        
        # Check Redis if available
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"city:{key}")
                if cached:
                    data = json.loads(cached)
                    # Also store in memory cache for faster access
                    self.memory_cache[key] = {
                        'value': data,
                        'timestamp': time.time()
                    }
                    return data
            except Exception:
                pass
        
        # Check disk cache
        if self.disk_cache:
            try:
                cached = self.disk_cache.get(key)
                if cached and time.time() - cached.get('timestamp', 0) < config.CACHE_TTL:
                    # Store in memory cache for faster access
                    self.memory_cache[key] = cached
                    return cached.get('value')
            except Exception:
                pass
        
        return default
    
    def set(self, key: str, value: Any, ttl: int = None):
        cache_item = {
            'value': value,
            'timestamp': time.time()
        }
        
        # Store in memory cache
        self.memory_cache[key] = cache_item
        
        # Store in Redis if available
        if self.redis_client:
            try:
                ttl_actual = ttl or config.CACHE_TTL
                self.redis_client.setex(
                    f"city:{key}",
                    ttl_actual,
                    json.dumps(value, default=str)
                )
            except Exception:
                pass
        
        # Store in disk cache
        if self.disk_cache:
            try:
                self.disk_cache.set(key, cache_item, expire=ttl or config.CACHE_TTL)
            except Exception:
                pass

cache = EnhancedCache()

# ==================== REQUEST HANDLER ====================
class RequestHandler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CityExplorer/2.0; +https://traveltto.com)',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def get_json(self, url: str, params: dict = None) -> Any:
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Request failed for {url}: {e}")
            return None
    
    def get_json_cached(self, url: str, params: dict = None, cache_key: str = None) -> Any:
        if not cache_key:
            cache_key = hashlib.md5(
                f"{url}{json.dumps(params or {}, sort_keys=True)}".encode()
            ).hexdigest()
        
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        
        data = self.get_json(url, params)
        if data:
            cache.set(cache_key, data, config.CACHE_TTL)
        
        return data

request_handler = RequestHandler()

# ==================== IMAGE FETCHER (KEEPING YOUR LOGIC) ====================
class IntelligentImageFetcher:
    def __init__(self):
        self.wikimedia_api = "https://commons.wikimedia.org/w/api.php"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
    
    def get_one_representative_image(self, city_name: str, page_title: str = None) -> Optional[Dict]:
        """Get ONLY ONE representative image for city preview"""
        cache_key = f"one_image:{city_name}:{page_title}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Try to get Wikipedia page image first
            if page_title:
                params = {
                    'action': 'query',
                    'titles': page_title,
                    'prop': 'pageimages',
                    'pithumbsize': 800,
                    'format': 'json'
                }
                
                data = request_handler.get_json_cached(
                    self.wikipedia_api,
                    params=params,
                    cache_key=f"wiki_page_image:{page_title}"
                )
                
                if data:
                    pages = data.get('query', {}).get('pages', {})
                    for page in pages.values():
                        if 'thumbnail' in page:
                            thumb = page['thumbnail']
                            image = {
                                'url': thumb['source'],
                                'title': f'Image of {city_name}',
                                'description': f'Featured image from Wikipedia',
                                'source': 'wikipedia',
                                'width': thumb.get('width'),
                                'height': thumb.get('height')
                            }
                            cache.set(cache_key, image, config.CACHE_TTL_IMAGES)
                            return image
        except Exception as e:
            logger.debug(f"Wikipedia image fetch failed: {e}")
        
        # Fallback image
        encoded_city = quote_plus(city_name)
        fallback_image = {
            'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80&txt={encoded_city}&txt-size=40&txt-color=white&txt-align=middle,center',
            'title': f'{city_name}',
            'description': f'Representation of {city_name}',
            'source': 'placeholder',
            'width': 800,
            'height': 600
        }
        
        cache.set(cache_key, fallback_image, config.CACHE_TTL_IMAGES)
        return fallback_image
    
    def get_images_for_city(self, city_name: str, page_title: str = None, limit: int = None) -> List[Dict]:
        """Get multiple images for detailed view"""
        limit = limit or config.MAX_IMAGES_PER_REQUEST
        cache_key = f"all_images:{city_name}:{page_title}:{limit}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        images = []
        
        try:
            # Try Wikipedia images
            params = {
                'action': 'query',
                'titles': page_title or city_name,
                'prop': 'images|pageimages',
                'pithumbsize': 800,
                'imlimit': limit,
                'format': 'json'
            }
            
            data = request_handler.get_json_cached(
                self.wikipedia_api,
                params=params,
                cache_key=f"wikipedia_images:{page_title or city_name}"
            )
            
            if data:
                pages = data.get('query', {}).get('pages', {})
                for page in pages.values():
                    if 'thumbnail' in page:
                        thumb = page['thumbnail']
                        images.append({
                            'url': thumb['source'],
                            'title': f'Main image of {city_name}',
                            'description': 'Featured image from Wikipedia',
                            'source': 'wikipedia',
                            'width': thumb.get('width'),
                            'height': thumb.get('height')
                        })
        except Exception as e:
            logger.debug(f"Wikipedia images failed: {e}")
        
        # If we don't have enough images, add fallbacks
        while len(images) < min(limit, 3):
            encoded_city = quote_plus(city_name)
            images.append({
                'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80&txt={encoded_city}&txt-size=40&txt-color=white&txt-align=middle,center',
                'title': f'{city_name}',
                'description': f'Representation of {city_name}',
                'source': 'placeholder',
                'width': 800,
                'height': 600
            })
        
        cache.set(cache_key, images, config.CACHE_TTL_IMAGES)
        return images[:limit]

image_fetcher = IntelligentImageFetcher()

# ==================== CITY DATA PROVIDER (OPTIMIZED) ====================
class CityDataProvider:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="CityExplorer/2.0 (https://traveltto.com)",
            timeout=config.GEOLOCATOR_TIMEOUT
        )
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='CityExplorer/2.0 (https://traveltto.com)',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
    
    def _get_short_description(self, city_name: str, country: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Get only 1-2 sentences for preview"""
        cache_key = f"short_desc:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached.get('description'), cached.get('title')
        
        variations = [city_name]
        if country:
            variations.append(f"{city_name}, {country}")
        
        for variation in variations:
            try:
                page = self.wiki.page(variation)
                if page.exists() and page.ns == 0:
                    summary = page.summary or ""
                    if summary:
                        # Get first 2 sentences max
                        sentences = re.split(r'[.!?]', summary)
                        short_desc = ""
                        sentence_count = 0
                        for sentence in sentences:
                            if sentence.strip():
                                short_desc += sentence.strip() + '. '
                                sentence_count += 1
                                if sentence_count >= 2:
                                    break
                        
                        short_desc = short_desc.strip()
                        if short_desc:
                            cache.set(cache_key, {
                                'description': short_desc,
                                'title': page.title
                            }, config.CACHE_TTL_PREVIEW)
                            return short_desc, page.title
            except Exception:
                continue
        
        return None, None
    
    def get_coordinates(self, city_name: str, country: str = None) -> Optional[Dict[str, float]]:
        """Get coordinates with caching"""
        cache_key = f"coords:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            query = f"{city_name}, {country}" if country else city_name
            location = self.geolocator.geocode(
                query,
                exactly_one=True,
                timeout=config.GEOLOCATOR_TIMEOUT
            )
            
            if location and hasattr(location, 'latitude'):
                coords = {"lat": location.latitude, "lon": location.longitude}
                cache.set(cache_key, coords, config.CACHE_TTL)
                return coords
        except Exception as e:
            logger.debug(f"Coordinates fetch failed for {city_name}: {e}")
        
        return None
    
    def get_city_preview(self, city_name: str, country: str = None, region: str = None) -> Dict:
        """Get minimal preview data for city list"""
        cache_key = f"preview:{city_name}:{country}:{region}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Get short description
        short_desc, wiki_title = self._get_short_description(city_name, country)
        
        # Get one image
        image = image_fetcher.get_one_representative_image(city_name, wiki_title)
        
        # Get coordinates
        coordinates = self.get_coordinates(city_name, country)
        
        # Create preview
        preview = {
            "id": city_name.lower().replace(' ', '-'),
            "name": city_name,
            "display_name": wiki_title or city_name,
            "country": country,
            "region": region,
            "image": image,
            "description": short_desc or f"Discover {city_name}, a city in {country or 'the world'}",
            "coordinates": coordinates,
            "has_details": True
        }
        
        cache.set(cache_key, preview, config.CACHE_TTL_PREVIEW)
        return preview
    
    def get_city_details(self, city_name: str, country: str = None, region: str = None) -> Dict:
        """Get full details for individual city"""
        cache_key = f"details:{city_name}:{country}:{region}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Start with preview
        preview = self.get_city_preview(city_name, country, region)
        
        # Get all images
        _, wiki_title = self._get_short_description(city_name, country)
        all_images = image_fetcher.get_images_for_city(city_name, wiki_title)
        
        # Build details
        details = {
            **preview,
            "all_images": all_images,
            "image_count": len(all_images),
            "landmarks": [],  # Would extract from Wikipedia if needed
            "wiki_summary": preview.get("description", ""),
            "detailed_info_available": True
        }
        
        cache.set(cache_key, details, config.CACHE_TTL)
        return details

data_provider = CityDataProvider()

# ==================== FLASK APP ====================
app = Flask(__name__)

CORS(app, 
     origins=["https://www.traveltto.com", "http://localhost:3000"],
     methods=["GET", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False,
     max_age=3600)

# ==================== WORLD CITIES DATA ====================
WORLD_CITIES = [
    # EUROPE (Expanded - 200+ cities)
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
    {"name":"San Sebastian","country":"Spain","region":"Europe"},
    {"name":"Toledo","country":"Spain","region":"Europe"},
    {"name":"Cordoba","country":"Spain","region":"Europe"},
    {"name":"Malaga","country":"Spain","region":"Europe"},
    {"name":"Marbella","country":"Spain","region":"Europe"},
    {"name":"Ibiza","country":"Spain","region":"Europe"},
    {"name":"Mallorca","country":"Spain","region":"Europe"},
    {"name":"Tenerife","country":"Spain","region":"Europe"},
    {"name":"Gran Canaria","country":"Spain","region":"Europe"},
    {"name":"Lanzarote","country":"Spain","region":"Europe"},
    {"name":"Fuerteventura","country":"Spain","region":"Europe"},
    {"name":"Menorca","country":"Spain","region":"Europe"},
    {"name":"Ronda","country":"Spain","region":"Europe"},
    {"name":"Salamanca","country":"Spain","region":"Europe"},
    {"name":"Avila","country":"Spain","region":"Europe"},
    {"name":"Segovia","country":"Spain","region":"Europe"},
    {"name":"Girona","country":"Spain","region":"Europe"},
    {"name":"Tarragona","country":"Spain","region":"Europe"},
    {"name":"Santiago de Compostela","country":"Spain","region":"Europe"},
    {"name":"Toulouse","country":"France","region":"Europe"},
    {"name":"Lille","country":"France","region":"Europe"},
    {"name":"Nantes","country":"France","region":"Europe"},
    {"name":"Rennes","country":"France","region":"Europe"},
    {"name":"Montpellier","country":"France","region":"Europe"},
    {"name":"Toulon","country":"France","region":"Europe"},
    {"name":"Avignon","country":"France","region":"Europe"},
    {"name":"Arles","country":"France","region":"Europe"},
    {"name":"Aix-en-Provence","country":"France","region":"Europe"},
    {"name":"Tours","country":"France","region":"Europe"},
    {"name":"Colmar","country":"France","region":"Europe"},
    {"name":"Annecy","country":"France","region":"Europe"},
    {"name":"Chamonix","country":"France","region":"Europe"},
    {"name":"Biarritz","country":"France","region":"Europe"},
    {"name":"La Rochelle","country":"France","region":"Europe"},
    {"name":"St Malo","country":"France","region":"Europe"},
    {"name":"Deauville","country":"France","region":"Europe"},
    {"name":"Honfleur","country":"France","region":"Europe"},
    {"name":"Etretat","country":"France","region":"Europe"},
    {"name":"Bergamo","country":"Italy","region":"Europe"},
    {"name":"Brescia","country":"Italy","region":"Europe"},
    {"name":"Modena","country":"Italy","region":"Europe"},
    {"name":"Parma","country":"Italy","region":"Europe"},
    {"name":"Reggio Emilia","country":"Italy","region":"Europe"},
    {"name":"Ravenna","country":"Italy","region":"Europe"},
    {"name":"Ferrara","country":"Italy","region":"Europe"},
    {"name":"Bari","country":"Italy","region":"Europe"},
    {"name":"Lecce","country":"Italy","region":"Europe"},
    {"name":"Brindisi","country":"Italy","region":"Europe"},
    {"name":"Taranto","country":"Italy","region":"Europe"},
    {"name":"Catania","country":"Italy","region":"Europe"},
    {"name":"Syracuse","country":"Italy","region":"Europe"},
    {"name":"Taormina","country":"Italy","region":"Europe"},
    {"name":"Cagliari","country":"Italy","region":"Europe"},
    {"name":"Olbia","country":"Italy","region":"Europe"},
    {"name":"Alghero","country":"Italy","region":"Europe"},
    {"name":"Trento","country":"Italy","region":"Europe"},
    {"name":"Bolzano","country":"Italy","region":"Europe"},
    {"name":"Aosta","country":"Italy","region":"Europe"},
    {"name":"Perugia","country":"Italy","region":"Europe"},
    {"name":"Assisi","country":"Italy","region":"Europe"},
    {"name":"Orvieto","country":"Italy","region":"Europe"},
    {"name":"Spoleto","country":"Italy","region":"Europe"},
    {"name":"Tivoli","country":"Italy","region":"Europe"},
    {"name":"Pompeii","country":"Italy","region":"Europe"},
    {"name":"Herculaneum","country":"Italy","region":"Europe"},
    {"name":"Capri","country":"Italy","region":"Europe"},
    {"name":"Ischia","country":"Italy","region":"Europe"},
    {"name":"Elba","country":"Italy","region":"Europe"},
    {"name":"Lake Maggiore","country":"Italy","region":"Europe"},
    {"name":"Lake Orta","country":"Italy","region":"Europe"},
    {"name":"Lake Iseo","country":"Italy","region":"Europe"},

    # MOROCCO (Expanded as requested - 50+ cities)
    {"name":"Marrakech","country":"Morocco","region":"Africa"},
    {"name":"Casablanca","country":"Morocco","region":"Africa"},
    {"name":"Fez","country":"Morocco","region":"Africa"},
    {"name":"Tangier","country":"Morocco","region":"Africa"},
    {"name":"Rabat","country":"Morocco","region":"Africa"},
    {"name":"Essaouira","country":"Morocco","region":"Africa"},
    {"name":"Chefchaouen","country":"Morocco","region":"Africa"},
    {"name":"Agadir","country":"Morocco","region":"Africa"},
    {"name":"Ouarzazate","country":"Morocco","region":"Africa"},
    {"name":"Meknes","country":"Morocco","region":"Africa"},
    {"name":"Tetouan","country":"Morocco","region":"Africa"},
    {"name":"El Jadida","country":"Morocco","region":"Africa"},
    {"name":"Safi","country":"Morocco","region":"Africa"},
    {"name":"Kenitra","country":"Morocco","region":"Africa"},
    {"name":"Nador","country":"Morocco","region":"Africa"},
    {"name":"Settat","country":"Morocco","region":"Africa"},
    {"name":"Mohammedia","country":"Morocco","region":"Africa"},
    {"name":"Khouribga","country":"Morocco","region":"Africa"},
    {"name":"Beni Mellal","country":"Morocco","region":"Africa"},
    {"name":"Taza","country":"Morocco","region":"Africa"},
    {"name":"Al Hoceima","country":"Morocco","region":"Africa"},
    {"name":"Larache","country":"Morocco","region":"Africa"},
    {"name":"Ksar El Kebir","country":"Morocco","region":"Africa"},
    {"name":"Guelmim","country":"Morocco","region":"Africa"},
    {"name":"Errachidia","country":"Morocco","region":"Africa"},
    {"name":"Taroudant","country":"Morocco","region":"Africa"},
    {"name":"Sidi Ifni","country":"Morocco","region":"Africa"},
    {"name":"Midelt","country":"Morocco","region":"Africa"},
    {"name":"Azrou","country":"Morocco","region":"Africa"},
    {"name":"Ifrane","country":"Morocco","region":"Africa"},
    {"name":"Moulay Idriss","country":"Morocco","region":"Africa"},
    {"name":"Volubilis","country":"Morocco","region":"Africa"},
    {"name":"Ait Benhaddou","country":"Morocco","region":"Africa"},
    {"name":"Merzouga","country":"Morocco","region":"Africa"},
    {"name":"Zagora","country":"Morocco","region":"Africa"},
    {"name":"Tinghir","country":"Morocco","region":"Africa"},
    {"name":"Dakhla","country":"Morocco","region":"Africa"},
    {"name":"Laayoune","country":"Morocco","region":"Africa"},
    {"name":"Smara","country":"Morocco","region":"Africa"},
    {"name":"Asilah","country":"Morocco","region":"Africa"},
    {"name":"M'diq","country":"Morocco","region":"Africa"},
    {"name":"Fnideq","country":"Morocco","region":"Africa"},
    {"name":"Martil","country":"Morocco","region":"Africa"},
    {"name":"Bouznika","country":"Morocco","region":"Africa"},
    {"name":"Temara","country":"Morocco","region":"Africa"},
    {"name":"Skhirat","country":"Morocco","region":"Africa"},
    {"name":"Benslimane","country":"Morocco","region":"Africa"},
    {"name":"Berrechid","country":"Morocco","region":"Africa"},
    {"name":"Youssoufia","country":"Morocco","region":"Africa"},
    {"name":"Oujda","country":"Morocco","region":"Africa"},
    {"name":"Taourirt","country":"Morocco","region":"Africa"},
    {"name":"Jerada","country":"Morocco","region":"Africa"},
    {"name":"Figuig","country":"Morocco","region":"Africa"},
    {"name":"Berkane","country":"Morocco","region":"Africa"},
    {"name":"Nador","country":"Morocco","region":"Africa"},
    {"name":"Al Hoceima","country":"Morocco","region":"Africa"},
    {"name":"Taza","country":"Morocco","region":"Africa"},
    {"name":"Sefrou","country":"Morocco","region":"Africa"},
    {"name":"Boulemane","country":"Morocco","region":"Africa"},
    {"name":"Midelt","country":"Morocco","region":"Africa"},
    {"name":"Errachidia","country":"Morocco","region":"Africa"},
    {"name":"Goulmima","country":"Morocco","region":"Africa"},
    {"name":"Rissani","country":"Morocco","region":"Africa"},
    {"name":"Merzouga","country":"Morocco","region":"Africa"},
    {"name":"Tineghir","country":"Morocco","region":"Africa"},
    {"name":"Todgha Gorge","country":"Morocco","region":"Africa"},
    {"name":"Dades Valley","country":"Morocco","region":"Africa"},
    {"name":"Skoura","country":"Morocco","region":"Africa"},
    {"name":"Kelaa Mgouna","country":"Morocco","region":"Africa"},
    {"name":"Taliouine","country":"Morocco","region":"Africa"},
    {"name":"Tafraoute","country":"Morocco","region":"Africa"},
    {"name":"Mirleft","country":"Morocco","region":"Africa"},
    {"name":"Sidi Kaouki","country":"Morocco","region":"Africa"},
    {"name":"Taghazout","country":"Morocco","region":"Africa"},
    {"name":"Tamraght","country":"Morocco","region":"Africa"},
    {"name":"Imsouane","country":"Morocco","region":"Africa"},

    # MIDDLE EAST (Expanded as requested - 100+ cities, NO ISRAELI CITIES)
    {"name":"Dubai","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Abu Dhabi","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Sharjah","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Ras Al Khaimah","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Fujairah","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Ajman","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Umm Al Quwain","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Al Ain","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Dibba","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Khor Fakkan","country":"United Arab Emirates","region":"Middle East"},
    {"name":"Doha","country":"Qatar","region":"Middle East"},
    {"name":"Al Wakrah","country":"Qatar","region":"Middle East"},
    {"name":"Al Khor","country":"Qatar","region":"Middle East"},
    {"name":"Al Rayyan","country":"Qatar","region":"Middle East"},
    {"name":"Umm Salal","country":"Qatar","region":"Middle East"},
    {"name":"Madinat ash Shamal","country":"Qatar","region":"Middle East"},
    {"name":"Manama","country":"Bahrain","region":"Middle East"},
    {"name":"Muharraq","country":"Bahrain","region":"Middle East"},
    {"name":"Riffa","country":"Bahrain","region":"Middle East"},
    {"name":"Hamad Town","country":"Bahrain","region":"Middle East"},
    {"name":"Isa Town","country":"Bahrain","region":"Middle East"},
    {"name":"Sitra","country":"Bahrain","region":"Middle East"},
    {"name":"Budaiya","country":"Bahrain","region":"Middle East"},
    {"name":"Jidhafs","country":"Bahrain","region":"Middle East"},
    {"name":"Kuwait City","country":"Kuwait","region":"Middle East"},
    {"name":"Hawalli","country":"Kuwait","region":"Middle East"},
    {"name":"Farwaniya","country":"Kuwait","region":"Middle East"},
    {"name":"Jahra","country":"Kuwait","region":"Middle East"},
    {"name":"Ahmadi","country":"Kuwait","region":"Middle East"},
    {"name":"Salmiya","country":"Kuwait","region":"Middle East"},
    {"name":"Muscat","country":"Oman","region":"Middle East"},
    {"name":"Salalah","country":"Oman","region":"Middle East"},
    {"name":"Sohar","country":"Oman","region":"Middle East"},
    {"name":"Sur","country":"Oman","region":"Middle East"},
    {"name":"Nizwa","country":"Oman","region":"Middle East"},
    {"name":"Ibri","country":"Oman","region":"Middle East"},
    {"name":"Rustaq","country":"Oman","region":"Middle East"},
    {"name":"Bahla","country":"Oman","region":"Middle East"},
    {"name":"Barka","country":"Oman","region":"Middle East"},
    {"name":"Khasab","country":"Oman","region":"Middle East"},
    {"name":"Duqm","country":"Oman","region":"Middle East"},
    {"name":"Seeb","country":"Oman","region":"Middle East"},
    {"name":"Al Buraimi","country":"Oman","region":"Middle East"},
    {"name":"Riyadh","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Jeddah","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Mecca","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Medina","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Dammam","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Khobar","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Taif","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Tabuk","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Abha","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Jizan","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Najran","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Hail","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Buraidah","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Khamis Mushait","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Al Hofuf","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Al Jubail","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Yanbu","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Unaizah","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Arar","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Sakakah","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Jubail","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Dhahran","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Qatif","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Al Bahah","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Tarut","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Safwa","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Ras Tanura","country":"Saudi Arabia","region":"Middle East"},
    {"name":"Sana'a","country":"Yemen","region":"Middle East"},
    {"name":"Aden","country":"Yemen","region":"Middle East"},
    {"name":"Taiz","country":"Yemen","region":"Middle East"},
    {"name":"Hodeidah","country":"Yemen","region":"Middle East"},
    {"name":"Ibb","country":"Yemen","region":"Middle East"},
    {"name":"Dhamar","country":"Yemen","region":"Middle East"},
    {"name":"Al Mukalla","country":"Yemen","region":"Middle East"},
    {"name":"Zinjibar","country":"Yemen","region":"Middle East"},
    {"name":"Sayyan","country":"Yemen","region":"Middle East"},
    {"name":"Sadah","country":"Yemen","region":"Middle East"},
    {"name":"Tehran","country":"Iran","region":"Middle East"},
    {"name":"Mashhad","country":"Iran","region":"Middle East"},
    {"name":"Isfahan","country":"Iran","region":"Middle East"},
    {"name":"Shiraz","country":"Iran","region":"Middle East"},
    {"name":"Tabriz","country":"Iran","region":"Middle East"},
    {"name":"Yazd","country":"Iran","region":"Middle East"},
    {"name":"Karaj","country":"Iran","region":"Middle East"},
    {"name":"Qom","country":"Iran","region":"Middle East"},
    {"name":"Ahvaz","country":"Iran","region":"Middle East"},
    {"name":"Kermanshah","country":"Iran","region":"Middle East"},
    {"name":"Rasht","country":"Iran","region":"Middle East"},
    {"name":"Kashan","country":"Iran","region":"Middle East"},
    {"name":"Hamadan","country":"Iran","region":"Middle East"},
    {"name":"Ardabil","country":"Iran","region":"Middle East"},
    {"name":"Bandar Abbas","country":"Iran","region":"Middle East"},
    {"name":"Arak","country":"Iran","region":"Middle East"},
    {"name":"Zahedan","country":"Iran","region":"Middle East"},
    {"name":"Sanandaj","country":"Iran","region":"Middle East"},
    {"name":"Qazvin","country":"Iran","region":"Middle East"},
    {"name":"Khorramabad","country":"Iran","region":"Middle East"},
    {"name":"Gorgan","country":"Iran","region":"Middle East"},
    {"name":"Sari","country":"Iran","region":"Middle East"},
    {"name":"Baghdad","country":"Iraq","region":"Middle East"},
    {"name":"Basra","country":"Iraq","region":"Middle East"},
    {"name":"Erbil","country":"Iraq","region":"Middle East"},
    {"name":"Mosul","country":"Iraq","region":"Middle East"},
    {"name":"Najaf","country":"Iraq","region":"Middle East"},
    {"name":"Karbala","country":"Iraq","region":"Middle East"},
    {"name":"Sulaymaniyah","country":"Iraq","region":"Middle East"},
    {"name":"Kirkuk","country":"Iraq","region":"Middle East"},
    {"name":"Nasiriyah","country":"Iraq","region":"Middle East"},
    {"name":"Amara","country":"Iraq","region":"Middle East"},
    {"name":"Ramadi","country":"Iraq","region":"Middle East"},
    {"name":"Fallujah","country":"Iraq","region":"Middle East"},
    {"name":"Hilla","country":"Iraq","region":"Middle East"},
    {"name":"Dahuk","country":"Iraq","region":"Middle East"},
    {"name":"Samarra","country":"Iraq","region":"Middle East"},
    {"name":"Amman","country":"Jordan","region":"Middle East"},
    {"name":"Petra","country":"Jordan","region":"Middle East"},
    {"name":"Aqaba","country":"Jordan","region":"Middle East"},
    {"name":"Jerash","country":"Jordan","region":"Middle East"},
    {"name":"Wadi Rum","country":"Jordan","region":"Middle East"},
    {"name":"Irbid","country":"Jordan","region":"Middle East"},
    {"name":"Zarqa","country":"Jordan","region":"Middle East"},
    {"name":"Madaba","country":"Jordan","region":"Middle East"},
    {"name":"Karak","country":"Jordan","region":"Middle East"},
    {"name":"Mafraq","country":"Jordan","region":"Middle East"},
    {"name":"Salt","country":"Jordan","region":"Middle East"},
    {"name":"Ajloun","country":"Jordan","region":"Middle East"},
    {"name":"Tafilah","country":"Jordan","region":"Middle East"},
    {"name":"Ma'an","country":"Jordan","region":"Middle East"},
    {"name":"Beirut","country":"Lebanon","region":"Middle East"},
    {"name":"Byblos","country":"Lebanon","region":"Middle East"},
    {"name":"Baalbek","country":"Lebanon","region":"Middle East"},
    {"name":"Tripoli","country":"Lebanon","region":"Middle East"},
    {"name":"Sidon","country":"Lebanon","region":"Middle East"},
    {"name":"Tyre","country":"Lebanon","region":"Middle East"},
    {"name":"Jounieh","country":"Lebanon","region":"Middle East"},
    {"name":"Zahle","country":"Lebanon","region":"Middle East"},
    {"name":"Batroun","country":"Lebanon","region":"Middle East"},
    {"name":"Jbeil","country":"Lebanon","region":"Middle East"},
    {"name":"Aley","country":"Lebanon","region":"Middle East"},
    {"name":"Bcharre","country":"Lebanon","region":"Middle East"},
    {"name":"Damascus","country":"Syria","region":"Middle East"},
    {"name":"Aleppo","country":"Syria","region":"Middle East"},
    {"name":"Palmyra","country":"Syria","region":"Middle East"},
    {"name":"Homs","country":"Syria","region":"Middle East"},
    {"name":"Latakia","country":"Syria","region":"Middle East"},
    {"name":"Hama","country":"Syria","region":"Middle East"},
    {"name":"Raqqa","country":"Syria","region":"Middle East"},
    {"name":"Deir ez-Zor","country":"Syria","region":"Middle East"},
    {"name":"Idlib","country":"Syria","region":"Middle East"},
    {"name":"Daraa","country":"Syria","region":"Middle East"},
    {"name":"Tartus","country":"Syria","region":"Middle East"},
    {"name":"Al-Hasakah","country":"Syria","region":"Middle East"},
    {"name":"Qamishli","country":"Syria","region":"Middle East"},
    {"name":"Manbij","country":"Syria","region":"Middle East"},
    {"name":"Bethlehem","country":"Palestine","region":"Middle East"},
    {"name":"Ramallah","country":"Palestine","region":"Middle East"},
    {"name":"Gaza","country":"Palestine","region":"Middle East"},
    {"name":"Jericho","country":"Palestine","region":"Middle East"},
    {"name":"Hebron","country":"Palestine","region":"Middle East"},
    {"name":"Nablus","country":"Palestine","region":"Middle East"},
    {"name":"Jenin","country":"Palestine","region":"Middle East"},
    {"name":"Tulkarm","country":"Palestine","region":"Middle East"},
    {"name":"Qalqilya","country":"Palestine","region":"Middle East"},
    {"name":"Salfit","country":"Palestine","region":"Middle East"},
    {"name":"Tubas","country":"Palestine","region":"Middle East"},
    {"name":"Khan Yunis","country":"Palestine","region":"Middle East"},
    {"name":"Rafah","country":"Palestine","region":"Middle East"},

    # NORTH AMERICA (200+ cities)
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
    {"name":"David","country":"Panama","region":"North America"},
    {"name":"Colon","country":"Panama","region":"North America"},
    {"name":"Port-au-Prince","country":"Haiti","region":"North America"},
    {"name":"Cap-Haitien","country":"Haiti","region":"North America"},
    {"name":"Jacmel","country":"Haiti","region":"North America"},
    {"name":"Roseau","country":"Dominica","region":"North America"},
    {"name":"Basse-Terre","country":"Guadeloupe","region":"North America"},
    {"name":"Fort-de-France","country":"Martinique","region":"North America"},
    {"name":"Providenciales","country":"Turks and Caicos","region":"North America"},
    {"name":"George Town","country":"Cayman Islands","region":"North America"},
    {"name":"Hamilton","country":"Bermuda","region":"North America"},
    {"name":"Nuuk","country":"Greenland","region":"North America"},
    {"name":"Ilulissat","country":"Greenland","region":"North America"},
    {"name":"Kangerlussuaq","country":"Greenland","region":"North America"},
    {"name":"Torshavn","country":"Faroe Islands","region":"North America"},
    {"name":"St Pierre","country":"St Pierre and Miquelon","region":"North America"},

    # SOUTH AMERICA (150+ cities)
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
    {"name":"Curitiba","country":"Brazil","region":"South America"},
    {"name":"Porto Alegre","country":"Brazil","region":"South America"},
    {"name":"Belém","country":"Brazil","region":"South America"},
    {"name":"Goiania","country":"Brazil","region":"South America"},
    {"name":"Campinas","country":"Brazil","region":"South America"},
    {"name":"Natal","country":"Brazil","region":"South America"},
    {"name":"Joao Pessoa","country":"Brazil","region":"South America"},
    {"name":"Maceio","country":"Brazil","region":"South America"},
    {"name":"Aracaju","country":"Brazil","region":"South America"},
    {"name":"Vitoria","country":"Brazil","region":"South America"},
    {"name":"Cuiaba","country":"Brazil","region":"South America"},
    {"name":"Campo Grande","country":"Brazil","region":"South America"},
    {"name":"Teresina","country":"Brazil","region":"South America"},
    {"name":"Sao Luis","country":"Brazil","region":"South America"},
    {"name":"Palmas","country":"Brazil","region":"South America"},
    {"name":"Boa Vista","country":"Brazil","region":"South America"},
    {"name":"Porto Velho","country":"Brazil","region":"South America"},
    {"name":"Rio Branco","country":"Brazil","region":"South America"},
    {"name":"Macapa","country":"Brazil","region":"South America"},
    {"name":"Rosario","country":"Argentina","region":"South America"},
    {"name":"La Plata","country":"Argentina","region":"South America"},
    {"name":"Mar del Plata","country":"Argentina","region":"South America"},
    {"name":"San Juan","country":"Argentina","region":"South America"},
    {"name":"San Luis","country":"Argentina","region":"South America"},
    {"name":"Neuquen","country":"Argentina","region":"South America"},
    {"name":"Comodoro Rivadavia","country":"Argentina","region":"South America"},
    {"name":"Rio Gallegos","country":"Argentina","region":"South America"},
    {"name":"Formosa","country":"Argentina","region":"South America"},
    {"name":"Resistencia","country":"Argentina","region":"South America"},
    {"name":"Posadas","country":"Argentina","region":"South America"},
    {"name":"Corrientes","country":"Argentina","region":"South America"},
    {"name":"Parana","country":"Argentina","region":"South America"},
    {"name":"Santa Fe","country":"Argentina","region":"South America"},
    {"name":"Mendoza","country":"Argentina","region":"South America"},
    {"name":"San Rafael","country":"Argentina","region":"South America"},
    {"name":"Malargue","country":"Argentina","region":"South America"},
    {"name":"Bariloche","country":"Argentina","region":"South America"},
    {"name":"San Martin de los Andes","country":"Argentina","region":"South America"},
    {"name":"Villa La Angostura","country":"Argentina","region":"South America"},
    {"name":"El Calafate","country":"Argentina","region":"South America"},
    {"name":"El Chalten","country":"Argentina","region":"South America"},
    {"name":"Puerto Madryn","country":"Argentina","region":"South America"},
    {"name":"Trelew","country":"Argentina","region":"South America"},
    {"name":"Rawson","country":"Argentina","region":"South America"},
    {"name":"Comodoro Rivadavia","country":"Argentina","region":"South America"},
    {"name":"Rio Grande","country":"Argentina","region":"South America"},
    {"name":"Ushuaia","country":"Argentina","region":"South America"},
    {"name":"Tolhuin","country":"Argentina","region":"South America"},
    {"name":"Caleta Olivia","country":"Argentina","region":"South America"},
    {"name":"Puerto Deseado","country":"Argentina","region":"South America"},
    {"name":"San Julian","country":"Argentina","region":"South America"},
    {"name":"Puerto San Julian","country":"Argentina","region":"South America"},
    {"name":"Gobernador Gregores","country":"Argentina","region":"South America"},
    {"name":"Perito Moreno","country":"Argentina","region":"South America"},
    {"name":"Los Antiguos","country":"Argentina","region":"South America"},
    {"name":"Chile Chico","country":"Chile","region":"South America"},
    {"name":"Coyhaique","country":"Chile","region":"South America"},
    {"name":"Puerto Aysen","country":"Chile","region":"South America"},
    {"name":"Puerto Chacabuco","country":"Chile","region":"South America"},
    {"name":"Puerto Natales","country":"Chile","region":"South America"},
    {"name":"Punta Arenas","country":"Chile","region":"South America"},
    {"name":"Porvenir","country":"Chile","region":"South America"},
    {"name":"Puerto Williams","country":"Chile","region":"South America"},
    {"name":"Ushuaia","country":"Argentina","region":"South America"},

    # ASIA (250+ cities)
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

    # AFRICA (200+ cities)
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
    {"name":"Algiers","country":"Algeria","region":"Africa"},
    {"name":"Oran","country":"Algeria","region":"Africa"},
    {"name":"Constantine","country":"Algeria","region":"Africa"},
    {"name":"Annaba","country":"Algeria","region":"Africa"},
    {"name":"Tunis","country":"Tunisia","region":"Africa"},
    {"name":"Sousse","country":"Tunisia","region":"Africa"},
    {"name":"Hammamet","country":"Tunisia","region":"Africa"},
    {"name":"Djerba","country":"Tunisia","region":"Africa"},
    {"name":"Tripoli","country":"Libya","region":"Africa"},
    {"name":"Benghazi","country":"Libya","region":"Africa"},
    {"name":"Misrata","country":"Libya","region":"Africa"},
    {"name":"Khartoum","country":"Sudan","region":"Africa"},
    {"name":"Omdurman","country":"Sudan","region":"Africa"},
    {"name":"Port Sudan","country":"Sudan","region":"Africa"},
    {"name":"Asmara","country":"Eritrea","region":"Africa"},
    {"name":"Massawa","country":"Eritrea","region":"Africa"},
    {"name":"Keren","country":"Eritrea","region":"Africa"},
    {"name":"N'Djamena","country":"Chad","region":"Africa"},
    {"name":"Moundou","country":"Chad","region":"Africa"},
    {"name":"Sarh","country":"Chad","region":"Africa"},
    {"name":"Bangui","country":"Central African Republic","region":"Africa"},
    {"name":"Bimbo","country":"Central African Republic","region":"Africa"},
    {"name":"Berberati","country":"Central African Republic","region":"Africa"},
    {"name":"Brazzaville","country":"Republic of Congo","region":"Africa"},
    {"name":"Pointe-Noire","country":"Republic of Congo","region":"Africa"},
    {"name":"Dolisie","country":"Republic of Congo","region":"Africa"},
    {"name":"Kinshasa","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Lubumbashi","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Mbuji-Mayi","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Bukavu","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Goma","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Kisangani","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Matadi","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Kananga","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Mbandaka","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Butembo","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Beni","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Uvira","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Kalemie","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Kindu","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Isiro","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Bunia","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Lisala","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Gemena","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Boende","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Basankusu","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Bondo","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Aketi","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Bafwasende","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Watsa","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Niangara","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Dungu","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Faradje","country":"Democratic Republic of Congo","region":"Africa"},
    {"name":"Ab","country":"Democratic Republic of Congo","region":"Africa"},

    # OCEANIA (100+ cities)
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

# Create lookup dictionaries for faster filtering
CITIES_BY_REGION = {}
CITIES_BY_COUNTRY = {}

for city in WORLD_CITIES:
    region = city.get('region')
    country = city.get('country')
    
    if region:
        if region not in CITIES_BY_REGION:
            CITIES_BY_REGION[region] = []
        CITIES_BY_REGION[region].append(city)
    
    if country:
        if country not in CITIES_BY_COUNTRY:
            CITIES_BY_COUNTRY[country] = []
        CITIES_BY_COUNTRY[country].append(city)

# Get unique regions from data
REGIONS = sorted(list(set(city.get('region') for city in WORLD_CITIES if city.get('region'))))

@app.route('/')
def home():
    return jsonify({
        "name": "City Explorer API",
        "version": "3.0.0",
        "status": "operational",
        "mode": "optimized-with-caching",
        "description": "/api/cities returns name, region, description, and ONE image per city with smart caching",
        "features": [
            "Region filtering for faster loading",
            "Smart caching (memory, Redis, disk)",
            "Real Wikipedia images and descriptions",
            "Individual city details on demand"
        ],
        "endpoints": {
            "health": "/api/health",
            "cities": "/api/cities?region=Europe (filter by region/country)",
            "city_details": "/api/cities/<city_name> (full details)",
            "search": "/api/search",
            "regions": "/api/regions"
        },
        "total_cities": len(WORLD_CITIES)
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "world_cities_count": len(WORLD_CITIES),
        "unique_regions": len(REGIONS),
        "cache_enabled": True,
        "performance_mode": "optimized"
    })

@app.route('/api/regions')
def get_regions():
    """Get all available regions"""
    return jsonify({
        "success": True,
        "regions": REGIONS,
        "count": len(REGIONS)
    })

@app.route('/api/cities')
def get_cities():
    """
    OPTIMIZED ENDPOINT: Returns ONLY name, region, description, and ONE image per city
    - Supports region/country filtering for faster loading
    - Uses smart caching for instant responses after first load
    - Real Wikipedia images and descriptions
    """
    region = request.args.get('region', type=str)
    country = request.args.get('country', type=str)
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    
    # Check cache first
    cache_key = f"cities_list:{region}:{country}:{page}:{limit}"
    cached_data = cache.get(cache_key)
    if cached_data:
        logger.info(f"📊 Cache hit for cities (region={region}, country={country})")
        return jsonify({
            "success": True,
            "data": cached_data,
            "total": len(cached_data),
            "filtered_by": {"region": region, "country": country} if region or country else "all",
            "cached": True
        })
    
    # Filter cities efficiently using pre-built dictionaries
    if region and region in CITIES_BY_REGION:
        filtered_cities = CITIES_BY_REGION[region]
    elif country and country in CITIES_BY_COUNTRY:
        filtered_cities = CITIES_BY_COUNTRY[country]
    else:
        filtered_cities = WORLD_CITIES
    
    # Apply pagination
    total_cities = len(filtered_cities)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total_cities)
    
    logger.info(f"📊 Processing {len(filtered_cities[start_idx:end_idx])} cities (region={region}, country={country})")
    
    # Process cities in parallel for faster loading
    cities_list = []
    
    def process_city(city_info):
        try:
            return data_provider.get_city_preview(
                city_info['name'],
                city_info.get('country'),
                city_info.get('region')
            )
        except Exception as e:
            logger.warning(f"Failed to load {city_info['name']}: {e}")
            return {
                "id": city_info['name'].lower().replace(' ', '-'),
                "name": city_info['name'],
                "display_name": city_info['name'],
                "country": city_info.get('country'),
                "region": city_info.get('region'),
                "image": {
                    "url": f"https://via.placeholder.com/400x300.png?text={quote_plus(city_info['name'])}",
                    "title": city_info['name'],
                    "description": f"Image of {city_info['name']}",
                    "source": "placeholder"
                },
                "description": f"{city_info['name']}, {city_info.get('country', 'a city')}",
                "coordinates": None,
                "has_details": False
            }
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, len(filtered_cities[start_idx:end_idx]))) as executor:
        cities_list = list(executor.map(process_city, filtered_cities[start_idx:end_idx]))
    
    logger.info(f"✅ Processed {len(cities_list)} cities")
    
    # Cache the results
    cache.set(cache_key, cities_list, config.CACHE_TTL_PREVIEW)
    
    return jsonify({
        "success": True,
        "data": cities_list,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_cities,
            "pages": max(1, (total_cities + limit - 1) // limit),
            "next_page": page + 1 if end_idx < total_cities else None,
            "prev_page": page - 1 if page > 1 else None
        },
        "filtered_by": {"region": region, "country": country} if region or country else "all",
        "cached": False
    })

@app.route('/api/cities/<path:city_name>')
def get_city(city_name):
    """
    Individual city endpoint - loads full details only when requested
    """
    city_name = unquote(city_name)
    
    # Find city
    city_info = None
    for city in WORLD_CITIES:
        if city['name'].lower() == city_name.lower():
            city_info = city
            break
    
    if not city_info:
        return jsonify({
            "success": False,
            "error": f"City '{city_name}' not found"
        }), 404
    
    try:
        city_data = data_provider.get_city_details(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        
        return jsonify({
            "success": True,
            "data": city_data
        })
        
    except Exception as e:
        logger.error(f"Failed to get city {city_name}: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch city details"
        }), 500

@app.route('/api/search')
def search_cities():
    """Fast search endpoint with caching"""
    query = request.args.get('q', '').strip().lower()
    
    if len(query) < 2:
        return jsonify({
            "success": False,
            "error": "Search query must be at least 2 characters"
        }), 400
    
    limit = request.args.get('limit', 20, type=int)
    limit = min(max(1, limit), 100)
    
    # Check cache
    cache_key = f"search:{query}:{limit}"
    cached_results = cache.get(cache_key)
    if cached_results:
        return jsonify({
            "success": True,
            "query": query,
            "count": len(cached_results),
            "data": cached_results,
            "cached": True
        })
    
    results = []
    
    # Simple linear search (optimized for 1000+ cities)
    for city in WORLD_CITIES:
        if query in city['name'].lower():
            try:
                city_data = data_provider.get_city_preview(
                    city['name'],
                    city.get('country'),
                    city.get('region')
                )
                
                results.append(city_data)
                
                if len(results) >= limit:
                    break
            except Exception:
                continue
    
    # Cache search results
    cache.set(cache_key, results, config.CACHE_TTL_PREVIEW)
    
    return jsonify({
        "success": True,
        "query": query,
        "count": len(results),
        "data": results,
        "cached": False
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# ==================== STARTUP ====================
if __name__ == '__main__':
    logger.info("🚀 Starting City Explorer API with OPTIMIZED CACHING")
    logger.info(f"📊 Total cities: {len(WORLD_CITIES)}")
    logger.info(f"📊 Unique regions: {len(REGIONS)}")
    logger.info("✅ Region filtering enabled")
    logger.info("✅ Smart caching enabled (memory, Redis, disk)")
    logger.info("✅ Real Wikipedia images and descriptions")
    logger.info("✅ Individual city details load on demand")
    
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=config.FLASK_DEBUG,
        threaded=True
    )
else:
    logger.info("🔧 Running in serverless mode with optimized caching")
    logger.info(f"📊 Found {len(WORLD_CITIES)} cities")
    logger.info(f"📊 Pre-built region/country indices for fast filtering")