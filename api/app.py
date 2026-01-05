import os
import re
import json
import time
import logging
import threading
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
from datetime import datetime, timedelta
from collections import defaultdict
from urllib.parse import quote_plus, unquote, urlparse
import requests
import wikipediaapi
import diskcache
from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from dataclasses import dataclass, field
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, RateLimitException
import backoff

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")
    MAPBOX_ACCESS_TOKEN: str = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "7200"))
    CACHE_TTL_IMAGES: int = int(os.getenv("CACHE_TTL_IMAGES", "86400"))
    CACHE_TTL_COORDS: int = int(os.getenv("CACHE_TTL_COORDS", "259200"))
    CACHE_TTL_PREVIEW: int = int(os.getenv("CACHE_TTL_PREVIEW", "3600"))
    
    MAX_IMAGE_WORKERS: int = int(os.getenv("MAX_IMAGE_WORKERS", "6"))
    MAX_DETAIL_WORKERS: int = int(os.getenv("MAX_DETAIL_WORKERS", "4"))
    MAX_PRELOAD_WORKERS: int = int(os.getenv("MAX_PRELOAD_WORKERS", "3"))
    
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    WIKIPEDIA_TIMEOUT: int = int(os.getenv("WIKIPEDIA_TIMEOUT", "20"))
    GEOLOCATOR_TIMEOUT: int = int(os.getenv("GEOLOCATOR_TIMEOUT", "10"))
    
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    FLASK_PORT: int = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    
    MAP_TILE_PROVIDER: str = os.getenv("MAP_TILE_PROVIDER", "openstreetmap")
    
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/city_explorer_cache")
    LOCAL_CACHE_FILE: str = os.getenv("LOCAL_CACHE_FILE", "/tmp/cities_data.json")
    IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "/tmp/image_cache")
    
    PRELOAD_TOP_CITIES: int = int(os.getenv("PRELOAD_TOP_CITIES", "8"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "20"))
    LAZY_LOADING: bool = os.getenv("LAZY_LOADING", "true").lower() == "true"
    
    MAX_WIKIMEDIA_FILES_TO_SCAN: int = 80
    MAX_IMAGES_PER_REQUEST: int = 8
    MAX_IMAGES_PREVIEW: int = 1
    WIKIMEDIA_RETRY_ATTEMPTS: int = 3
    MIN_IMAGE_WIDTH: int = 400
    MIN_IMAGE_HEIGHT: int = 300
    PREFERRED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.webp'])
    
    MIN_IMAGE_QUALITY_SCORE: int = 40
    REQUIRED_SUCCESS_RATE: float = 0.6
    
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    ENABLE_FALLBACK_IMAGES: bool = os.getenv("ENABLE_FALLBACK_IMAGES", "true").lower() == "true"
    ENABLE_COORDINATE_FALLBACK: bool = os.getenv("ENABLE_COORDINATE_FALLBACK", "true").lower() == "true"
    
    REQUESTS_PER_MINUTE: int = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
    WIKIMEDIA_RATE_LIMIT: int = int(os.getenv("WIKIMEDIA_RATE_LIMIT", "50"))
    
    MAX_TEXT_LENGTH: int = 1000000
    MAX_SUMMARY_LENGTH: int = 1000000
    MAX_SECTION_LENGTH: int = 1000000
    MAX_DETAILED_SUMMARY_LENGTH: int = 1000000

config = Config()

# ==================== ENHANCED LOGGING ====================
class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("CityExplorer")
logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter())
logger.addHandler(console_handler)

try:
    file_handler = logging.FileHandler('/tmp/city_explorer.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not set up file logging: {e}")

logger.info("🚀 City Explorer API Initializing with MINIMAL PREVIEW MODE...")

# ==================== TEXT PROCESSING HELPERS ====================
def clean_text(text: str) -> str:
    if not text:
        return ""
    
    cleaned = re.sub(r'\[\d+\]', '', text)
    cleaned = re.sub(r'\{\{.*?\}\}', '', cleaned)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

# ==================== ENHANCED CACHING SYSTEM ====================
class MultiLevelCache:
    def __init__(self):
        self.memory_cache = {}
        self.disk_cache = None
        self.redis_client = None
        self.hits = 0
        self.misses = 0
        
        try:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            os.makedirs(config.IMAGE_CACHE_DIR, exist_ok=True)
            self.disk_cache = diskcache.Cache(config.CACHE_DIR)
            logger.info(f"✅ Disk cache initialized at: {config.CACHE_DIR}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize disk cache: {e}")
            self.disk_cache = diskcache.Cache()
        
        if os.getenv("REDIS_URL"):
            try:
                self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis cache unavailable: {e}")
    
    def get(self, key: str, default=None):
        if key in self.memory_cache:
            item = self.memory_cache.get(key)
            if item and time.time() - item.get('timestamp', 0) < config.CACHE_TTL:
                self.hits += 1
                return item.get('value')
        
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"city:{key}")
                if cached:
                    self.hits += 1
                    data = json.loads(cached)
                    self.memory_cache[key] = {
                        'value': data,
                        'timestamp': time.time()
                    }
                    return data
            except Exception:
                pass
        
        if self.disk_cache:
            try:
                cached = self.disk_cache.get(key)
                if cached and time.time() - cached.get('timestamp', 0) < config.CACHE_TTL:
                    self.hits += 1
                    self.memory_cache[key] = cached
                    return cached.get('value')
            except Exception:
                pass
        
        self.misses += 1
        return default
    
    def set(self, key: str, value: Any, ttl: int = None):
        cache_item = {
            'value': value,
            'timestamp': time.time()
        }
        
        self.memory_cache[key] = cache_item
        
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
        
        if self.disk_cache:
            try:
                self.disk_cache.set(key, cache_item, expire=ttl or config.CACHE_TTL)
            except Exception:
                pass
    
    def delete(self, key: str):
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if self.redis_client:
            try:
                self.redis_client.delete(f"city:{key}")
            except Exception:
                pass
        
        if self.disk_cache:
            try:
                del self.disk_cache[key]
            except Exception:
                pass
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'memory_items': len(self.memory_cache),
            'disk_size': len(self.disk_cache) if self.disk_cache else 0
        }

cache = MultiLevelCache()

# ==================== ENHANCED REQUEST HANDLER ====================
class SmartRequestHandler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CityExplorer/2.0; +https://traveltto.com)',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.request_times = []
        self.failure_count = defaultdict(int)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.Timeout, 
                                       requests.exceptions.ConnectionError))
    )
    def get_with_retry(self, url: str, params: dict = None, headers: dict = None, 
                       timeout: int = None) -> requests.Response:
        start_time = time.time()
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or config.REQUEST_TIMEOUT
            )
            
            duration = time.time() - start_time
            self.request_times.append(duration)
            if len(self.request_times) > 100:
                self.request_times.pop(0)
            
            if duration > 5:
                logger.warning(f"Slow request: {url} took {duration:.2f}s")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            self.failure_count[url] += 1
            logger.error(f"Request failed for {url}: {e}")
            
            if self.failure_count[url] > 5:
                logger.warning(f"Circuit breaker triggered for {url}")
                raise
            
            raise
    
    def get_json_cached(self, url: str, params: dict = None, headers: dict = None, 
                        cache_key: str = None, ttl: int = None) -> Any:
        if not cache_key:
            cache_key = hashlib.md5(
                f"{url}{json.dumps(params or {}, sort_keys=True)}".encode()
            ).hexdigest()
        
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {url}")
            return cached
        
        try:
            response = self.get_with_retry(url, params, headers)
            data = response.json()
            
            cache.set(cache_key, data, ttl or config.CACHE_TTL)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            
            if cached is not None:
                logger.info(f"Using stale cache for {url}")
                return cached
            
            raise
    
    def get_performance_stats(self):
        if not self.request_times:
            return {}
        
        return {
            'total_requests': len(self.request_times),
            'avg_time': sum(self.request_times) / len(self.request_times),
            'max_time': max(self.request_times),
            'min_time': min(self.request_times),
            'failure_counts': dict(self.failure_count)
        }

request_handler = SmartRequestHandler()

# ==================== INTELLIGENT IMAGE FETCHER ====================
class IntelligentImageFetcher:
    def __init__(self):
        self.wikimedia_api = "https://commons.wikimedia.org/w/api.php"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        
    def calculate_image_quality(self, image_info: dict) -> int:
        score = 50
        
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        if width >= 1200 and height >= 800:
            score += 30
        elif width >= 800 and height >= 600:
            score += 20
        elif width >= 400 and height >= 300:
            score += 10
        
        url = image_info.get('url', '').lower()
        if any(fmt in url for fmt in ['.jpg', '.jpeg']):
            score += 5
        
        if width > 0 and height > 0:
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                score -= 10
        
        if image_info.get('source') == 'wikimedia':
            score += 5
        
        return min(100, max(0, score))
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def fetch_from_wikimedia(self, query: str, limit: int = 8) -> List[Dict]:
        images = []
        
        try:
            search_queries = [
                f'{query} city',
                f'{query} landscape',
                f'{query} aerial view',
                f'{query} cityscape',
                f'{query} tourism'
            ]
            
            for search_query in search_queries:
                if len(images) >= limit:
                    break
                    
                params = {
                    'action': 'query',
                    'generator': 'search',
                    'gsrsearch': search_query,
                    'gsrnamespace': '6',
                    'gsrlimit': 20,
                    'prop': 'imageinfo',
                    'iiprop': 'url|size|mime|extmetadata',
                    'iiurlwidth': 800,
                    'format': 'json'
                }
                
                data = request_handler.get_json_cached(
                    self.wikimedia_api,
                    params=params,
                    cache_key=f"wikimedia:{search_query}",
                    ttl=config.CACHE_TTL_IMAGES
                )
                
                for page in data.get('query', {}).get('pages', {}).values():
                    if 'imageinfo' in page:
                        info = page['imageinfo'][0]
                        
                        if self._is_relevant_image(info, query):
                            image_data = {
                                'url': info.get('thumburl') or info.get('url'),
                                'title': page.get('title', '').replace('File:', ''),
                                'description': self._extract_description(info),
                                'source': 'wikimedia',
                                'width': info.get('width'),
                                'height': info.get('height'),
                                'quality_score': self.calculate_image_quality(info),
                                'page_url': f"https://commons.wikimedia.org/wiki/{page.get('title', '')}"
                            }
                            
                            if image_data['url'] and image_data['quality_score'] >= config.MIN_IMAGE_QUALITY_SCORE:
                                images.append(image_data)
                                    
                                if len(images) >= limit:
                                    break
            
            images.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
        except Exception as e:
            logger.warning(f"Wikimedia fetch failed for {query}: {e}")
        
        return images[:limit]
    
    def fetch_from_wikipedia(self, page_title: str, limit: int = 6) -> List[Dict]:
        images = []
        
        try:
            params = {
                'action': 'query',
                'titles': page_title,
                'prop': 'images|pageimages',
                'pithumbsize': 1000,
                'imlimit': 50,
                'format': 'json'
            }
            
            data = request_handler.get_json_cached(
                self.wikipedia_api,
                params=params,
                cache_key=f"wikipedia_images:{page_title}",
                ttl=config.CACHE_TTL_IMAGES
            )
            
            pages = data.get('query', {}).get('pages', {})
            file_titles = []
            
            for page in pages.values():
                if 'thumbnail' in page:
                    thumb = page['thumbnail']
                    if thumb.get('source'):
                        images.append({
                            'url': thumb['source'],
                            'title': f'Main image of {page_title}',
                            'description': f'Featured image from Wikipedia',
                            'source': 'wikipedia',
                            'width': thumb.get('width'),
                            'height': thumb.get('height'),
                            'quality_score': 80,
                            'page_url': f"https://en.wikipedia.org/wiki/{quote_plus(page_title)}"
                        })
                
                for img in page.get('images', []):
                    title = img.get('title', '')
                    if title.startswith('File:'):
                        lower_title = title.lower()
                        if not any(x in lower_title for x in ['.svg', '.ogg', '.webm', '.tif']):
                            file_titles.append(title)
            
            batch_size = 10
            for i in range(0, min(len(file_titles), 30), batch_size):
                batch = file_titles[i:i + batch_size]
                titles_param = '|'.join(batch)
                
                params = {
                    'action': 'query',
                    'titles': titles_param,
                    'prop': 'imageinfo',
                    'iiprop': 'url|size|mime',
                    'iiurlwidth': 800,
                    'format': 'json'
                }
                
                batch_data = request_handler.get_json_cached(
                    self.wikipedia_api,
                    params=params,
                    cache_key=f"wikipedia_batch:{hashlib.md5(titles_param.encode()).hexdigest()}",
                    ttl=config.CACHE_TTL_IMAGES
                )
                
                for page in batch_data.get('query', {}).get('pages', {}).values():
                    if 'imageinfo' in page:
                        info = page['imageinfo'][0]
                        mime = info.get('mime', '')
                        
                        if mime.startswith('image/'):
                            image_data = {
                                'url': info.get('thumburl') or info.get('url'),
                                'title': page.get('title', '').replace('File:', ''),
                                'description': f'Image from {page_title}',
                                'source': 'wikipedia',
                                'width': info.get('width'),
                                'height': info.get('height'),
                                'quality_score': self.calculate_image_quality(info),
                                'page_url': f"https://en.wikipedia.org/wiki/{quote_plus(page_title)}"
                            }
                            
                            if image_data['quality_score'] >= config.MIN_IMAGE_QUALITY_SCORE:
                                images.append(image_data)
                
                if len(images) >= limit:
                    break
            
            seen_urls = set()
            unique_images = []
            for img in sorted(images, key=lambda x: x.get('quality_score', 0), reverse=True):
                if img['url'] not in seen_urls:
                    seen_urls.add(img['url'])
                    unique_images.append(img)
            
            return unique_images[:limit]
            
        except Exception as e:
            logger.warning(f"Wikipedia image fetch failed for {page_title}: {e}")
            return []
    
    def get_one_representative_image(self, city_name: str, page_title: str = None) -> Optional[Dict]:
        """Get ONLY ONE representative image for city preview"""
        cache_key = f"one_image:{city_name}:{page_title}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Try to get Wikipedia page image first (usually most representative)
            if page_title:
                params = {
                    'action': 'query',
                    'titles': page_title,
                    'prop': 'pageimages',
                    'pithumbsize': 1200,
                    'format': 'json'
                }
                
                data = request_handler.get_json_cached(
                    self.wikipedia_api,
                    params=params,
                    cache_key=f"wiki_page_image:{page_title}",
                    ttl=config.CACHE_TTL_IMAGES
                )
                
                pages = data.get('query', {}).get('pages', {})
                for page in pages.values():
                    if 'thumbnail' in page:
                        thumb = page['thumbnail']
                        image = {
                            'url': thumb['source'],
                            'title': f'Representative image of {city_name}',
                            'description': f'Featured image from Wikipedia',
                            'source': 'wikipedia',
                            'width': thumb.get('width'),
                            'height': thumb.get('height'),
                            'quality_score': 85,
                            'page_url': f"https://en.wikipedia.org/wiki/{quote_plus(page_title)}"
                        }
                        cache.set(cache_key, image, config.CACHE_TTL_IMAGES)
                        return image
            
            # If no Wikipedia image, try to get one good city image
            all_images = self.get_images_for_city(city_name, page_title, limit=3)
            if all_images:
                # Pick the best quality image
                best_image = max(all_images, key=lambda x: x.get('quality_score', 0))
                cache.set(cache_key, best_image, config.CACHE_TTL_IMAGES)
                return best_image
                
        except Exception as e:
            logger.debug(f"Single image fetch failed: {e}")
        
        return None
    
    def get_images_for_city(self, city_name: str, page_title: str = None, limit: int = None) -> List[Dict]:
        limit = limit or config.MAX_IMAGES_PER_REQUEST
        images = []
        seen_urls = set()
        
        # Try Wikipedia images
        if len(images) < limit:
            try:
                wiki_images = self.fetch_from_wikipedia(page_title or city_name, limit - len(images))
                for img in wiki_images:
                    if img['url'] not in seen_urls:
                        seen_urls.add(img['url'])
                        images.append(img)
                        
                        if len(images) >= limit:
                            break
            except Exception as e:
                logger.debug(f"Wikipedia images failed: {e}")
        
        # Try Wikimedia for cityscapes
        if len(images) < limit:
            try:
                wikimedia_images = self.fetch_from_wikimedia(city_name, limit - len(images))
                for img in wikimedia_images:
                    if img['url'] not in seen_urls:
                        seen_urls.add(img['url'])
                        images.append(img)
                        
                        if len(images) >= limit:
                            break
            except Exception as e:
                logger.debug(f"Wikimedia images failed: {e}")
        
        # Fallback
        if not images and config.ENABLE_FALLBACK_IMAGES:
            fallback = self.generate_fallback_image(city_name)
            images.append(fallback)
        
        return images[:limit]
    
    def generate_fallback_image(self, city_name: str) -> Dict:
        encoded_city = quote_plus(city_name)
        return {
            'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80&txt={encoded_city}&txt-size=40&txt-color=white&txt-align=middle,center',
            'title': f'{city_name}',
            'description': f'Representation of {city_name}',
            'source': 'placeholder',
            'width': 800,
            'height': 600,
            'quality_score': 30,
            'page_url': f'https://unsplash.com/s/photos/{encoded_city}-city'
        }
    
    def _is_relevant_image(self, image_info: dict, query: str) -> bool:
        mime = image_info.get('mime', '')
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        url = image_info.get('url', '').lower()
        
        if not mime.startswith('image/'):
            return False
        
        if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
            return False
        
        if not any(fmt in url for fmt in config.PREFERRED_IMAGE_FORMATS):
            return False
        
        if width > 0 and height > 0:
            ratio = width / height
            if ratio < 0.3 or ratio > 3.0:
                return False
        
        return True
    
    def _extract_description(self, image_info: dict) -> str:
        extmetadata = image_info.get('extmetadata', {})
        
        for field in ['ImageDescription', 'ObjectName', 'Caption']:
            if field in extmetadata:
                value = extmetadata[field].get('value', '')
                if isinstance(value, str) and value.strip():
                    clean_value = re.sub(r'<[^>]+>', '', value)
                    return clean_value
        
        return ""

image_fetcher = IntelligentImageFetcher()

# ==================== ENHANCED CITY DATA PROVIDER ====================
class EnhancedCityDataProvider:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="CityExplorer/2.0 (https://traveltto.com; contact@traveltto.com)",
            timeout=config.GEOLOCATOR_TIMEOUT
        )
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='CityExplorer/2.0 (https://traveltto.com)',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        self.map_provider = MapProvider()
        self.stats = {
            'previews_generated': 0,
            'details_generated': 0,
            'coordinates_found': 0,
            'coordinates_failed': 0,
            'wiki_found': 0,
            'wiki_failed': 0,
            'images_found': 0,
            'images_failed': 0
        }
    
    def get_coordinates_enhanced(self, city_name: str, country: str = None, 
                                region: str = None) -> Optional[Tuple[float, float, Dict]]:
        cache_key = f"coords:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            self.stats['coordinates_found'] += 1
            return cached
        
        strategies = [
            self._get_coordinates_nominatim,
            self._get_coordinates_wikipedia,
            self._get_coordinates_wikidata,
        ]
        
        for strategy in strategies:
            try:
                result = strategy(city_name, country, region)
                if result:
                    lat, lon, metadata = result
                    
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        cache.set(cache_key, result, config.CACHE_TTL_COORDS)
                        self.stats['coordinates_found'] += 1
                        
                        logger.info(f"✅ Coordinates found for {city_name}: {lat}, {lon}")
                        return result
            except Exception as e:
                logger.debug(f"Coordinate strategy failed for {city_name}: {e}")
                continue
        
        self.stats['coordinates_failed'] += 1
        logger.warning(f"❌ No coordinates found for {city_name}")
        return None
    
    def _get_coordinates_nominatim(self, city_name: str, country: str = None, 
                                  region: str = None) -> Optional[Tuple[float, float, Dict]]:
        queries = []
        
        if country:
            queries.append(f"{city_name}, {country}")
            if region:
                queries.append(f"{city_name}, {region}, {country}")
        
        queries.append(f"{city_name}")
        queries.append(f"{city_name} city")
        
        for query in queries:
            try:
                location = self.geolocator.geocode(
                    query,
                    exactly_one=True,
                    addressdetails=True,
                    language="en",
                    timeout=config.GEOLOCATOR_TIMEOUT
                )
                
                if location and hasattr(location, 'latitude'):
                    metadata = getattr(location, 'raw', {})
                    metadata['source'] = 'nominatim'
                    metadata['query_used'] = query
                    
                    return (location.latitude, location.longitude, metadata)
                    
            except Exception as e:
                logger.debug(f"Nominatim query failed for '{query}': {e}")
                continue
        
        return None
    
    def _get_coordinates_wikipedia(self, city_name: str, country: str = None, 
                                  region: str = None) -> Optional[Tuple[float, float, Dict]]:
        try:
            page = self.wiki.page(city_name)
            if not page.exists():
                if country:
                    page = self.wiki.page(f"{city_name}, {country}")
            
            if page.exists():
                api_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'titles': page.title,
                    'prop': 'coordinates',
                    'format': 'json'
                }
                
                data = request_handler.get_json_cached(
                    api_url,
                    params=params,
                    cache_key=f"wiki_coords:{page.title}"
                )
                
                pages = data.get('query', {}).get('pages', {})
                for page_data in pages.values():
                    coords = page_data.get('coordinates')
                    if coords and len(coords) > 0:
                        coord = coords[0]
                        return (
                            coord['lat'],
                            coord['lon'],
                            {'source': 'wikipedia', 'page_title': page.title, 'page_url': f"https://en.wikipedia.org/wiki/{quote_plus(page.title)}"}
                        )
                        
        except Exception as e:
            logger.debug(f"Wikipedia coordinate fetch failed: {e}")
        
        return None
    
    def _get_coordinates_wikidata(self, city_name: str, country: str = None, 
                                 region: str = None) -> Optional[Tuple[float, float, Dict]]:
        try:
            search_url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbsearchentities',
                'search': city_name,
                'language': 'en',
                'format': 'json',
                'type': 'item'
            }
            
            if country:
                params['search'] = f"{city_name} {country}"
            
            search_data = request_handler.get_json_cached(
                search_url,
                params=params,
                cache_key=f"wikidata_search:{city_name}:{country}"
            )
            
            if search_data.get('search'):
                entity_id = search_data['search'][0]['id']
                
                entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                entity_data = request_handler.get_json_cached(
                    entity_url,
                    cache_key=f"wikidata_entity:{entity_id}"
                )
                
                entity = entity_data.get('entities', {}).get(entity_id, {})
                claims = entity.get('claims', {})
                
                if 'P625' in claims:
                    coord_claim = claims['P625'][0]['mainsnak']['datavalue']['value']
                    return (
                        coord_claim['latitude'],
                        coord_claim['longitude'],
                        {'source': 'wikidata', 'entity_id': entity_id, 'entity_url': f"https://www.wikidata.org/wiki/{entity_id}"}
                    )
                    
        except Exception as e:
            logger.debug(f"Wikidata coordinate fetch failed: {e}")
        
        return None
    
    def _get_short_description(self, city_name: str, country: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Get only 1-2 sentences for preview"""
        cache_key = f"short_desc:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached.get('description'), cached.get('title')
        
        variations = self._generate_wiki_variations(city_name, country)
        
        for variation in variations:
            try:
                page = self.wiki.page(variation)
                
                if page.exists() and page.ns == 0:
                    if self._is_city_page(page):
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
            except Exception as e:
                logger.debug(f"Short description check failed: {e}")
                continue
        
        return None, None
    
    def get_wikipedia_data_enhanced(self, city_name: str, country: str = None) -> Tuple[Optional[Dict], Optional[str]]:
        cache_key = f"wiki:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            self.stats['wiki_found'] += 1
            return cached.get('data'), cached.get('title')
        
        variations = self._generate_wiki_variations(city_name, country)
        
        for variation in variations:
            try:
                page = self.wiki.page(variation)
                
                if page.exists() and page.ns == 0:
                    if self._is_city_page(page):
                        page_data = self._extract_wiki_data(page)
                        cache.set(cache_key, {
                            'data': page_data,
                            'title': page.title
                        }, config.CACHE_TTL)
                        
                        self.stats['wiki_found'] += 1
                        logger.info(f"✅ Wikipedia page found for {city_name}: {page.title}")
                        return page_data, page.title
                        
            except Exception as e:
                logger.debug(f"Wikipedia check failed for '{variation}': {e}")
                continue
        
        self.stats['wiki_failed'] += 1
        logger.warning(f"❌ No Wikipedia page found for {city_name}")
        return None, None
    
    def _generate_wiki_variations(self, city_name: str, country: str = None) -> List[str]:
        variations = [city_name]
        
        if country:
            variations.extend([
                f"{city_name}, {country}",
                f"{city_name} ({country})",
                f"{city_name} City, {country}"
            ])
        
        variations.extend([
            f"{city_name} city",
            f"{city_name} (city)",
            f"The city of {city_name}",
            city_name.split(',')[0].strip() if ',' in city_name else city_name
        ])
        
        return list(dict.fromkeys([v for v in variations if v.strip()]))
    
    def _is_city_page(self, page) -> bool:
        try:
            text_lower = (page.summary or "").lower()
            
            city_indicators = [
                'city', 'town', 'municipality', 'capital', 'population',
                'located in', 'situated in', 'urban area'
            ]
            
            non_city_indicators = [
                'river', 'mountain', 'lake', 'island', 'species',
                'album', 'song', 'film', 'book', 'company'
            ]
            
            city_score = sum(1 for indicator in city_indicators if indicator in text_lower)
            non_city_score = sum(1 for indicator in non_city_indicators if indicator in text_lower)
            
            return city_score > non_city_score and city_score >= 1
            
        except Exception:
            return True
    
    def _extract_wiki_data(self, page) -> Dict:
        sections = []
        
        def extract_section_content(section, max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            
            title = section.title.strip()
            text = (section.text or "").strip()
            
            if title and text and title not in ["See also", "References", "External links", "Notes", "Bibliography"]:
                cleaned = clean_text(text)
                
                if cleaned:
                    sections.append({
                        "title": title,
                        "content": cleaned,
                        "length": len(cleaned)
                    })
            
            for subsection in getattr(section, 'sections', []):
                extract_section_content(subsection, max_depth, current_depth + 1)
        
        for section in getattr(page, 'sections', []):
            extract_section_content(section)
        
        full_summary = clean_text(page.summary or "")
        
        return {
            'title': page.title,
            'summary': full_summary,
            'fullurl': getattr(page, 'fullurl', f"https://en.wikipedia.org/wiki/{quote_plus(page.title)}"),
            'sections': sections,
            'pageid': getattr(page, 'pageid', None),
            'text_length': len(full_summary),
            'sections_count': len(sections)
        }
    
    def get_city_tagline_enhanced(self, city_name: str, country: str = None) -> Dict[str, str]:
        cache_key = f"tagline:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            summary, _ = self._get_short_description(city_name, country)
            if summary:
                # Use first sentence as tagline
                sentences = re.split(r'[.!?]', summary)
                if sentences and sentences[0]:
                    tagline = sentences[0].strip()
                    if tagline:
                        result = {"city": city_name, "tagline": tagline, "source": "wikipedia"}
                        cache.set(cache_key, result, config.CACHE_TTL)
                        return result
        except Exception as e:
            logger.debug(f"Tagline extraction failed: {e}")
        
        # Generic tagline
        if country:
            tagline = f"A city in {country}"
        else:
            tagline = f"Discover {city_name}"
        
        result = {"city": city_name, "tagline": tagline, "source": "generated"}
        cache.set(cache_key, result, config.CACHE_TTL)
        return result
    
    def get_city_preview_minimal(self, city_name: str, country: str = None, 
                                 region: str = None) -> Dict:
        """MINIMAL preview for /api/cities - ONLY name, 1 image, short desc, coords, region"""
        cache_key = f"minimal_preview:{city_name}:{country}:{region}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        self.stats['previews_generated'] += 1
        logger.info(f"🔄 Generating MINIMAL preview for {city_name}")
        
        # Start with absolute minimal structure
        preview = {
            "id": self._generate_city_id(city_name),
            "name": city_name,
            "display_name": city_name,
            "summary": "",  # Will be short description
            "has_details": False,
            "image": None,  # ONLY ONE image
            "images": [],   # EMPTY - no images array
            "coordinates": None,
            "static_map": None,
            "tagline": None,
            "last_updated": time.time(),
            "country": country,
            "region": region,
            "landmarks": [],  # EMPTY - no landmarks in preview
            "metadata": {
                "data_type": "minimal_preview"
            }
        }
        
        # Get coordinates
        try:
            coords_result = self.get_coordinates_enhanced(city_name, country, region)
            if coords_result:
                lat, lon, metadata = coords_result
                preview["coordinates"] = {"lat": lat, "lon": lon}
                
                # Generate static map
                preview["static_map"] = self.map_provider.generate_static_map_url(
                    {"lat": lat, "lon": lon}, width=400, height=250
                )
        except Exception as e:
            logger.warning(f"Coordinates fetch failed for {city_name}: {e}")
        
        # Get short description
        try:
            short_desc, wiki_title = self._get_short_description(city_name, country)
            if short_desc:
                preview["display_name"] = wiki_title or city_name
                preview["summary"] = short_desc
                preview["_wiki_title"] = wiki_title or city_name
            else:
                preview["summary"] = f"Discover {city_name}, a city in {country or 'the world'}"
        except Exception as e:
            logger.warning(f"Description fetch failed for {city_name}: {e}")
        
        # Get ONE image only
        try:
            wiki_title = preview.get("_wiki_title", city_name)
            image = image_fetcher.get_one_representative_image(city_name, wiki_title)
            
            if image:
                preview["image"] = image
                self.stats['images_found'] += 1
            else:
                self.stats['images_failed'] += 1
                # Add fallback image
                fallback_image = image_fetcher.generate_fallback_image(city_name)
                preview["image"] = fallback_image
                
        except Exception as e:
            logger.error(f"Image fetch failed for {city_name}: {e}")
            self.stats['images_failed'] += 1
            fallback_image = image_fetcher.generate_fallback_image(city_name)
            preview["image"] = fallback_image
        
        # Get tagline
        try:
            tagline_data = self.get_city_tagline_enhanced(city_name, country)
            preview["tagline"] = tagline_data.get("tagline")
        except Exception as e:
            logger.debug(f"Tagline fetch failed: {e}")
            preview["tagline"] = f"Discover {city_name}"
        
        # Cache the minimal preview
        cache.set(cache_key, preview, config.CACHE_TTL_PREVIEW)
        
        logger.info(f"✅ Minimal preview generated for {city_name}")
        return preview
    
    def get_city_details_enhanced(self, city_name: str, country: str = None, 
                                 region: str = None) -> Dict:
        """FULL details for individual city page"""
        cache_key = f"details:{city_name}:{country}:{region}"
        
        cached = cache.get(cache_key)
        if cached:
            self.stats['details_generated'] += 1
            return cached
        
        self.stats['details_generated'] += 1
        logger.info(f"🔄 Generating FULL details for {city_name}")
        
        # Start with minimal preview data
        preview = self.get_city_preview_minimal(city_name, country, region)
        
        # Build full details
        details = {
            **preview,
            "detailed_summary": "",
            "sections": [],
            "sources": [],
            "map_config": {},
            "additional_images": [],
            "statistics": {},
            "metadata": {
                **preview.get("metadata", {}),
                "data_type": "full_details",
                "loaded_at": time.time()
            }
        }
        
        # Get full Wikipedia data
        wiki_data, wiki_title = self.get_wikipedia_data_enhanced(city_name, country)
        if wiki_data:
            details["detailed_summary"] = wiki_data.get('summary', '')
            details["sources"].append(wiki_data.get('fullurl', ''))
            
            # Include sections
            sections_data = []
            for section in wiki_data.get('sections', []):
                if section.get('content'):
                    sections_data.append({
                        "title": section.get('title'),
                        "content": section.get('content')
                    })
            details["sections"] = sections_data
        
        # Get additional images
        try:
            additional_images = image_fetcher.get_images_for_city(
                city_name,
                wiki_title or city_name,
                limit=config.MAX_IMAGES_PER_REQUEST
            )
            
            if additional_images:
                # Update images array with all images
                details["images"] = additional_images
                # Keep first as main image
                if additional_images and not details.get("image"):
                    details["image"] = additional_images[0]
                # Set additional_images to all but first
                details["additional_images"] = additional_images[1:min(6, len(additional_images))]
        except Exception as e:
            logger.warning(f"Additional images failed for {city_name}: {e}")
        
        details["map_config"] = self.map_provider.get_map_config(
            city_name,
            details.get("coordinates")
        )
        
        # Mark that details are loaded
        details["has_details"] = True
        
        details["statistics"] = {
            "image_count": len(details.get("images", [])),
            "section_count": len(details.get("sections", [])),
            "last_updated": time.time()
        }
        
        cache.set(cache_key, details, config.CACHE_TTL)
        
        logger.info(f"✅ Details generated for {city_name}")
        return details
    
    def _generate_city_id(self, city_name: str) -> str:
        city_id = city_name.lower().strip()
        city_id = re.sub(r'[^\w\s-]', '', city_id)
        city_id = re.sub(r'[-\s]+', '-', city_id)
        return city_id
    
    def get_stats(self):
        return self.stats

# ==================== MAP PROVIDER ====================
class MapProvider:
    def __init__(self):
        self.tile_providers = {
            "openstreetmap": {
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attribution": "© OpenStreetMap contributors",
                "requires_token": False
            },
            "carto": {
                "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                "attribution": "© OpenStreetMap & CARTO",
                "requires_token": False
            },
            "opentopomap": {
                "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                "attribution": "© OpenStreetMap & OpenTopoMap",
                "requires_token": False
            }
        }
        
        if config.MAPBOX_ACCESS_TOKEN:
            self.tile_providers["mapbox"] = {
                "url": f"https://api.mapbox.com/styles/v1/mapbox/light-v10/tiles/{{z}}/{{x}}/{{y}}?access_token={config.MAPBOX_ACCESS_TOKEN}",
                "attribution": "© Mapbox & OpenStreetMap",
                "requires_token": True
            }
    
    def get_map_config(self, city_name: str, coordinates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        provider_key = config.MAP_TILE_PROVIDER
        
        if provider_key not in self.tile_providers:
            provider_key = "openstreetmap"
            logger.warning(f"Map provider {config.MAP_TILE_PROVIDER} not found, using {provider_key}")
        
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
                    "color": "#3388ff",
                    "icon": "circle"
                }
            })
        
        return map_config
    
    def generate_static_map_url(self, coordinates: Optional[Dict[str, float]], 
                               width: int = 600, height: int = 400,
                               zoom: int = 12) -> str:
        
        if not coordinates or not self._validate_coordinates(coordinates):
            return f"https://via.placeholder.com/{width}x{height}.png?text=Map+Not+Available"
        
        lat, lon = coordinates["lat"], coordinates["lon"]
        
        if config.MAPBOX_ACCESS_TOKEN:
            try:
                return f"https://api.mapbox.com/styles/v1/mapbox/light-v10/static/pin-l+3388ff({lon},{lat})/{lon},{lat},{zoom}/{width}x{height}?access_token={config.MAPBOX_ACCESS_TOKEN}"
            except Exception:
                pass
        
        try:
            return f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={width}x{height}&markers={lat},{lon},red-pushpin&scale=2"
        except Exception:
            pass
        
        return f"https://via.placeholder.com/{width}x{height}.png?text={lat:.4f}%2C{lon:.4f}"
    
    def _validate_coordinates(self, coordinates: Dict[str, float]) -> bool:
        try:
            lat = coordinates.get("lat")
            lon = coordinates.get("lon")
            
            if lat is None or lon is None:
                return False
            
            try:
                lat = float(lat)
                lon = float(lon)
            except (ValueError, TypeError):
                return False
            
            return (-90 <= lat <= 90 and -180 <= lon <= 180)
            
        except Exception:
            return False

# ==================== CITY LOADING MANAGER ====================
class CityLoadingManager:
    def __init__(self, data_provider: EnhancedCityDataProvider):
        self.data_provider = data_provider
        self.loaded_cities = {}
        self.loading_status = {
            'total': 0,
            'previews_loaded': 0,
            'details_loaded': 0,
            'failed': 0,
            'start_time': None,
        }
        self.loading_queue = []
        
    def initialize_with_world_cities(self, world_cities_data: List[Dict]):
        if not world_cities_data:
            logger.error("❌ No world cities data provided!")
            return
        
        self.loading_status['total'] = len(world_cities_data)
        self.loading_status['start_time'] = time.time()
        self.loading_status['previews_loaded'] = 0
        self.loading_status['details_loaded'] = 0
        self.loading_status['failed'] = 0
        
        self.loading_queue = []
        for city_data in world_cities_data:
            self.loading_queue.append({
                'name': city_data['name'],
                'country': city_data.get('country'),
                'region': city_data.get('region'),
                'priority': 50
            })
        
        self.loading_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"📊 Initialized loading manager with {len(self.loading_queue)} cities")
    
    def get_loading_status(self):
        return self.loading_status
    
    def get_city(self, city_name: str) -> Optional[Dict]:
        return self.loaded_cities.get(city_name)

# ==================== FLASK APP & ROUTES ====================
app = Flask(__name__)

CORS(app, 
     origins=["https://www.traveltto.com", "http://localhost:3000"],
     methods=["GET", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False,
     max_age=3600)

# Initialize providers
data_provider = EnhancedCityDataProvider()
map_provider = MapProvider()
city_loader = CityLoadingManager(data_provider)

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

REGIONS = set(["Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"])

@app.route('/')
def home():
    return jsonify({
        "name": "City Explorer API",
        "version": "2.0.0",
        "status": "operational",
        "mode": "minimal-preview-mode",
        "description": "/api/cities returns minimal data (name, 1 image, short description, coordinates, region). Full details on demand.",
        "endpoints": {
            "health": "/api/health",
            "cities": "/api/cities (minimal preview only)",
            "city_details": "/api/cities/<city_name> (full details)",
            "search": "/api/search",
            "stats": "/api/stats"
        },
        "note": "WORLD_CITIES list is currently empty. Add city data to populate the API."
    })

@app.route('/api/health')
def health():
    loader_status = city_loader.get_loading_status()
    cache_stats = cache.get_stats()
    request_stats = request_handler.get_performance_stats()
    provider_stats = data_provider.get_stats()
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "minimal-preview",
        "world_cities_count": len(WORLD_CITIES),
        "note": "WORLD_CITIES is empty. Add city data to enable city listings.",
        "city_loading": loader_status,
        "cache": cache_stats,
        "performance": request_stats,
        "provider_stats": provider_stats
    })

@app.route('/api/cities')
def get_cities():
    """
    ENDPOINT 1: Returns ONLY name, 1 image, and small description for each city
    """
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 30, type=int)
    region = request.args.get('region', type=str)
    country = request.args.get('country', type=str)
    
    # Add debug logging
    logger.info(f"📊 /api/cities called. WORLD_CITIES length: {len(WORLD_CITIES)}")
    
    if len(WORLD_CITIES) == 0:
        logger.warning("⚠️ WORLD_CITIES is empty! No cities to display.")
        return jsonify({
            "success": True,
            "data": [],
            "pagination": {
                "limit": limit,
                "next_page": None,
                "page": page,
                "pages": 1,
                "prev_page": None,
                "total": 0
            },
            "note": "WORLD_CITIES list is empty. Add city data to enable city listings.",
            "mode": "minimal-preview"
        })
    
    page = max(1, page)
    limit = min(max(1, limit), 100)
    
    # Filter cities
    filtered_cities = WORLD_CITIES
    if region:
        filtered_cities = [c for c in filtered_cities if c.get('region') == region]
    if country:
        filtered_cities = [c for c in filtered_cities if c.get('country') == country]
    
    logger.info(f"📊 Filtered cities: {len(filtered_cities)}")
    
    total_cities = len(filtered_cities)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total_cities)
    
    cities_list = []
    
    for i in range(start_idx, end_idx):
        city_info = filtered_cities[i]
        city_name = city_info['name']
        
        # Get the existing minimal preview
        try:
            city_preview = data_provider.get_city_preview_minimal(
                city_info['name'],
                city_info.get('country'),
                city_info.get('region')
            )
            
            cities_list.append({
                "id": city_preview["id"],
                "name": city_preview["name"],
                "display_name": city_preview["display_name"],
                "country": city_preview["country"],
                "region": city_preview["region"],
                "image": city_preview["image"], 
                "coordinates": city_preview["coordinates"],
                "summary": city_preview["summary"],
                "has_details": True 
            })
            
        except Exception as e:
            logger.warning(f"Failed to load {city_name}: {e}")
            cities_list.append({
                "id": city_name.lower().replace(' ', '-'),
                "name": city_name,
                "display_name": city_name,
                "country": city_info.get('country'),
                "region": city_info.get('region'),
                "image": {
                    "url": f"https://via.placeholder.com/400x300.png?text={quote_plus(city_name)}",
                    "title": city_name,
                    "description": f"Image of {city_name}",
                    "source": "placeholder"
                },
                "coordinates": None,
                "summary": f"{city_name}, {city_info.get('country', 'a city')}",
                "has_details": False
            })
    
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
        }
    })

@app.route('/api/cities/<path:city_name>')
def get_city(city_name):
    """
    ENDPOINT 2: Returns ALL images and REAL landmarks for a specific city
    """
    city_name = unquote(city_name)
    
    # Find city
    city_info = None
    for city in WORLD_CITIES:
        if city['name'].lower() == city_name.lower():
            city_info = city
            break
    
    if not city_info:
        # Even if not in WORLD_CITIES, try to fetch data for any city name
        logger.info(f"City '{city_name}' not in WORLD_CITIES list, attempting to fetch data anyway")
        try:
            # Get ALL images
            all_images = image_fetcher.get_images_for_city(
                city_name,
                city_name,  # Use city name as page title
                limit=config.MAX_IMAGES_PER_REQUEST
            )
            
            # Get Wikipedia data for landmarks
            wiki_data, wiki_title = data_provider.get_wikipedia_data_enhanced(
                city_name,
                None
            )
            
            # Extract REAL landmarks from Wikipedia sections
            landmarks = []
            if wiki_data and wiki_data.get('sections'):
                for section in wiki_data.get('sections', []):
                    section_title = section.get('title', '').lower()
                    section_content = section.get('content', '').lower()
                    
                    # Look for actual landmarks in content
                    if any(keyword in section_title for keyword in ['landmarks', 'attractions', 'architecture', 'monuments', 'tourist']):
                        # This section is about landmarks - extract specific ones
                        lines = section_content.split('.')
                        for line in lines:
                            line = line.strip()
                            if len(line) > 20 and any(word in line for word in [' is ', ' was ', ' built ', ' constructed ', ' located ', ' famous ', ' known ']):
                                # Likely a landmark description
                                landmarks.append(line[:100] + '...')
            
            # If no landmarks found in sections, use some from the summary
            if not landmarks and wiki_data and wiki_data.get('summary'):
                summary = wiki_data.get('summary', '')
                # Look for famous things mentioned
                sentences = summary.split('.')
                for sentence in sentences[:5]:
                    sentence = sentence.strip()
                    if len(sentence) > 30 and any(word in sentence.lower() for word in ['famous', 'known', 'notable', 'major', 'popular']):
                        landmarks.append(sentence[:150] + '...')
            
            # Get coordinates
            coords_result = data_provider.get_coordinates_enhanced(
                city_name,
                None,
                None
            )
            
            coordinates = None
            if coords_result:
                lat, lon, _ = coords_result
                coordinates = {"lat": lat, "lon": lon}
            
            return jsonify({
                "success": True,
                "data": {
                    "id": city_name.lower().replace(' ', '-'),
                    "name": city_name,
                    "country": None,
                    "region": None,
                    "all_images": all_images,
                    "image_count": len(all_images),
                    "landmarks": landmarks[:10],
                    "wiki_summary": wiki_data.get('summary', '') if wiki_data else "",
                    "wiki_url": wiki_data.get('fullurl', '') if wiki_data else "",
                    "coordinates": coordinates,
                    "note": "City fetched dynamically (not in WORLD_CITIES list)"
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to fetch city {city_name} dynamically: {e}")
            return jsonify({
                "success": False,
                "error": f"City '{city_name}' not found and could not be fetched dynamically"
            }), 404
    
    try:
        # Get ALL images
        all_images = image_fetcher.get_images_for_city(
            city_info['name'],
            city_info.get('name'),  # Use city name as page title
            limit=config.MAX_IMAGES_PER_REQUEST
        )
        
        # Get Wikipedia data for landmarks
        wiki_data, wiki_title = data_provider.get_wikipedia_data_enhanced(
            city_info['name'],
            city_info.get('country')
        )
        
        # Extract REAL landmarks from Wikipedia sections
        landmarks = []
        if wiki_data and wiki_data.get('sections'):
            for section in wiki_data.get('sections', []):
                section_title = section.get('title', '').lower()
                section_content = section.get('content', '').lower()
                
                # Look for actual landmarks in content
                if any(keyword in section_title for keyword in ['landmarks', 'attractions', 'architecture', 'monuments', 'tourist']):
                    # This section is about landmarks - extract specific ones
                    lines = section_content.split('.')
                    for line in lines:
                        line = line.strip()
                        if len(line) > 20 and any(word in line for word in [' is ', ' was ', ' built ', ' constructed ', ' located ', ' famous ', ' known ']):
                            # Likely a landmark description
                            landmarks.append(line[:100] + '...')
        
        # If no landmarks found in sections, use some from the summary
        if not landmarks and wiki_data and wiki_data.get('summary'):
            summary = wiki_data.get('summary', '')
            # Look for famous things mentioned
            sentences = summary.split('.')
            for sentence in sentences[:5]:
                sentence = sentence.strip()
                if len(sentence) > 30 and any(word in sentence.lower() for word in ['famous', 'known', 'notable', 'major', 'popular']):
                    landmarks.append(sentence[:150] + '...')
        
        # Get coordinates
        coords_result = data_provider.get_coordinates_enhanced(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        
        coordinates = None
        if coords_result:
            lat, lon, _ = coords_result
            coordinates = {"lat": lat, "lon": lon}
        
        return jsonify({
            "success": True,
            "data": {
                "id": city_info['name'].lower().replace(' ', '-'),
                "name": city_info['name'],
                "country": city_info.get('country'),
                "region": city_info.get('region'),
                "all_images": all_images,
                "image_count": len(all_images),
                "landmarks": landmarks[:10],
                "wiki_summary": wiki_data.get('summary', '') if wiki_data else "",
                "wiki_url": wiki_data.get('fullurl', '') if wiki_data else "",
                "coordinates": coordinates
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get city {city_name}: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch city details"
        }), 500

@app.route('/api/search')
def search_cities():
    query = request.args.get('q', '').strip()
    
    if len(query) < 2:
        return jsonify({
            "success": False,
            "error": "Search query must be at least 2 characters"
        }), 400
    
    limit = request.args.get('limit', 20, type=int)
    limit = min(max(1, limit), 50)
    
    # Check if WORLD_CITIES is empty
    if len(WORLD_CITIES) == 0:
        return jsonify({
            "success": True,
            "query": query,
            "count": 0,
            "data": [],
            "note": "WORLD_CITIES list is empty. Add city data to enable search.",
            "data_type": "minimal_preview"
        })
    
    results = []
    
    for city in WORLD_CITIES:
        if query.lower() in city['name'].lower():
            city_name = city['name']
            
            if city_name in city_loader.loaded_cities:
                results.append(city_loader.loaded_cities[city_name])
            else:
                try:
                    # Load minimal preview for search results
                    city_data = data_provider.get_city_preview_minimal(
                        city['name'],
                        city.get('country'),
                        city.get('region')
                    )
                    
                    # Ensure minimal structure
                    minimal_data = {
                        "id": city_data["id"],
                        "name": city_data["name"],
                        "display_name": city_data["display_name"],
                        "summary": city_data["summary"],
                        "has_details": False,
                        "image": city_data["image"],
                        "images": [],  # EMPTY
                        "coordinates": city_data["coordinates"],
                        "static_map": city_data["static_map"],
                        "tagline": city_data["tagline"],
                        "last_updated": city_data["last_updated"],
                        "country": city_data["country"],
                        "region": city_data["region"],
                        "landmarks": [],  # EMPTY
                        "metadata": {
                            "data_type": "minimal_preview"
                        }
                    }
                    
                    results.append(minimal_data)
                except Exception:
                    pass
            
            if len(results) >= limit:
                break
    
    return jsonify({
        "success": True,
        "query": query,
        "count": len(results),
        "data": results,
        "data_type": "minimal_preview"
    })

@app.route('/api/stats')
def get_stats():
    loader_status = city_loader.get_loading_status()
    cache_stats = cache.get_stats()
    provider_stats = data_provider.get_stats()
    request_stats = request_handler.get_performance_stats()
    
    return jsonify({
        "city_statistics": {
            "total_cities_in_list": len(WORLD_CITIES),
            "note": "WORLD_CITIES is empty. Add city data to enable city listings.",
            "previews_loaded": loader_status.get('previews_loaded', 0),
            "details_loaded": loader_status.get('details_loaded', 0),
        },
        "cache_statistics": cache_stats,
        "provider_statistics": provider_stats,
        "performance": request_stats,
        "mode": "minimal_preview_for_listings"
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

if __name__ == '__main__':
    logger.info("🚀 Starting City Explorer API with MINIMAL PREVIEW MODE")
    
    # Initialize with empty WORLD_CITIES
    if WORLD_CITIES and len(WORLD_CITIES) > 0:
        logger.info(f"📊 Total cities: {len(WORLD_CITIES)}")
    else:
        logger.info("ℹ️ WORLD_CITIES is empty. Add city data to enable city listings.")
    
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=config.FLASK_DEBUG,
        threaded=True
    )
else:
    logger.info("🔧 Running in serverless mode with MINIMAL PREVIEW")
    
    if WORLD_CITIES and len(WORLD_CITIES) > 0:
        logger.info(f"📊 Found {len(WORLD_CITIES)} cities")
    else:
        logger.info("ℹ️ WORLD_CITIES is empty. Add city data to enable city listings.")