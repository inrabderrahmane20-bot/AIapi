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
    """Configuration settings with Vercel-specific optimizations"""
    
    # API Keys & Secrets (Set in Vercel Environment Variables)
    UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")
    MAPBOX_ACCESS_TOKEN: str = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Performance & Caching
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "7200"))  # 2 hours
    CACHE_TTL_IMAGES: int = int(os.getenv("CACHE_TTL_IMAGES", "86400"))  # 24 hours for images
    CACHE_TTL_COORDS: int = int(os.getenv("CACHE_TTL_COORDS", "259200"))  # 3 days for coordinates
    
    # Worker Configuration
    MAX_IMAGE_WORKERS: int = int(os.getenv("MAX_IMAGE_WORKERS", "6"))  # Reduced for Vercel
    MAX_DETAIL_WORKERS: int = int(os.getenv("MAX_DETAIL_WORKERS", "4"))
    MAX_PRELOAD_WORKERS: int = int(os.getenv("MAX_PRELOAD_WORKERS", "3"))  # Conservative for serverless
    
    # Timeouts
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    WIKIPEDIA_TIMEOUT: int = int(os.getenv("WIKIPEDIA_TIMEOUT", "20"))
    GEOLOCATOR_TIMEOUT: int = int(os.getenv("GEOLOCATOR_TIMEOUT", "10"))
    
    # Flask & Server
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    FLASK_PORT: int = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    
    # Map Configuration
    MAP_TILE_PROVIDER: str = os.getenv("MAP_TILE_PROVIDER", "openstreetmap")
    
    # Cache Directories
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/city_explorer_cache")
    LOCAL_CACHE_FILE: str = os.getenv("LOCAL_CACHE_FILE", "/tmp/cities_data.json")
    IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "/tmp/image_cache")
    
    # Loading Configuration
    PRELOAD_TOP_CITIES: int = int(os.getenv("PRELOAD_TOP_CITIES", "8"))  # Smaller for faster cold start
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "20"))  # Cities per batch
    LAZY_LOADING: bool = os.getenv("LAZY_LOADING", "true").lower() == "true"
    
    # Image Fetching
    MAX_WIKIMEDIA_FILES_TO_SCAN: int = 80  # Increased for better coverage
    MAX_IMAGES_PER_REQUEST: int = 8  # More images for details
    WIKIMEDIA_RETRY_ATTEMPTS: int = 3
    MIN_IMAGE_WIDTH: int = 400  # Minimum acceptable image width
    MIN_IMAGE_HEIGHT: int = 300  # Minimum acceptable image height
    PREFERRED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.webp'])
    
    # Quality Thresholds
    MIN_IMAGE_QUALITY_SCORE: int = 40  # For image selection algorithm
    REQUIRED_SUCCESS_RATE: float = 0.6  # 60% of cities should have images
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Fallback Strategy
    ENABLE_FALLBACK_IMAGES: bool = os.getenv("ENABLE_FALLBACK_IMAGES", "true").lower() == "true"
    ENABLE_COORDINATE_FALLBACK: bool = os.getenv("ENABLE_COORDINATE_FALLBACK", "true").lower() == "true"
    
    # Rate Limiting
    REQUESTS_PER_MINUTE: int = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
    WIKIMEDIA_RATE_LIMIT: int = int(os.getenv("WIKIMEDIA_RATE_LIMIT", "50"))

config = Config()

# ==================== ENHANCED LOGGING ====================
class ColorFormatter(logging.Formatter):
    """Custom colored log formatter"""
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

# Setup enhanced logging
logger = logging.getLogger("CityExplorer")
logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter())
logger.addHandler(console_handler)

# File handler for Vercel logs
try:
    file_handler = logging.FileHandler('/tmp/city_explorer.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not set up file logging: {e}")

logger.info("🚀 City Explorer API Initializing...")

# ==================== ENHANCED CACHING SYSTEM ====================
class MultiLevelCache:
    """Multi-level cache with memory, disk, and fallback strategies"""
    
    def __init__(self):
        self.memory_cache = {}
        self.disk_cache = None
        self.redis_client = None
        self.hits = 0
        self.misses = 0
        
        # Initialize disk cache with error handling
        try:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            os.makedirs(config.IMAGE_CACHE_DIR, exist_ok=True)
            self.disk_cache = diskcache.Cache(config.CACHE_DIR)
            logger.info(f"✅ Disk cache initialized at: {config.CACHE_DIR}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize disk cache: {e}")
            self.disk_cache = diskcache.Cache()  # Memory-only fallback
        
        # Try Redis if available
        if os.getenv("REDIS_URL"):
            try:
                self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis cache unavailable: {e}")
    
    def get(self, key: str, default=None):
        """Multi-level cache retrieval"""
        # Try memory first
        if key in self.memory_cache:
            item = self.memory_cache.get(key)
            if item and time.time() - item.get('timestamp', 0) < config.CACHE_TTL:
                self.hits += 1
                return item.get('value')
        
        # Try Redis if available
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"city:{key}")
                if cached:
                    self.hits += 1
                    data = json.loads(cached)
                    # Store in memory for faster access
                    self.memory_cache[key] = {
                        'value': data,
                        'timestamp': time.time()
                    }
                    return data
            except Exception:
                pass
        
        # Try disk cache
        if self.disk_cache:
            try:
                cached = self.disk_cache.get(key)
                if cached and time.time() - cached.get('timestamp', 0) < config.CACHE_TTL:
                    self.hits += 1
                    # Store in memory
                    self.memory_cache[key] = cached
                    return cached.get('value')
            except Exception:
                pass
        
        self.misses += 1
        return default
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in all cache levels"""
        cache_item = {
            'value': value,
            'timestamp': time.time()
        }
        
        # Memory cache
        self.memory_cache[key] = cache_item
        
        # Redis cache
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
        
        # Disk cache
        if self.disk_cache:
            try:
                self.disk_cache.set(key, cache_item, expire=ttl or config.CACHE_TTL)
            except Exception:
                pass
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'memory_items': len(self.memory_cache),
            'disk_size': len(self.disk_cache) if self.disk_cache else 0
        }

# Initialize cache
cache = MultiLevelCache()

# ==================== ENHANCED REQUEST HANDLER ====================
class SmartRequestHandler:
    """Intelligent request handler with retry logic and fallbacks"""
    
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
        """Get request with intelligent retry logic"""
        start_time = time.time()
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or config.REQUEST_TIMEOUT
            )
            
            # Track performance
            duration = time.time() - start_time
            self.request_times.append(duration)
            if len(self.request_times) > 100:
                self.request_times.pop(0)
            
            # Log slow requests
            if duration > 5:
                logger.warning(f"Slow request: {url} took {duration:.2f}s")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            self.failure_count[url] += 1
            logger.error(f"Request failed for {url}: {e}")
            
            # Circuit breaker - if too many failures, skip for a while
            if self.failure_count[url] > 5:
                logger.warning(f"Circuit breaker triggered for {url}")
                raise
            
            raise
    
    def get_json_cached(self, url: str, params: dict = None, headers: dict = None, 
                        cache_key: str = None, ttl: int = None) -> Any:
        """Get JSON with intelligent caching"""
        if not cache_key:
            cache_key = hashlib.md5(
                f"{url}{json.dumps(params or {}, sort_keys=True)}".encode()
            ).hexdigest()
        
        # Check cache first
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {url}")
            return cached
        
        # Fetch fresh data
        try:
            response = self.get_with_retry(url, params, headers)
            data = response.json()
            
            # Store in cache
            cache.set(cache_key, data, ttl or config.CACHE_TTL)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            
            # Try to return stale cache if available (even if expired)
            if cached is not None:
                logger.info(f"Using stale cache for {url}")
                return cached
            
            raise
    
    def get_performance_stats(self):
        """Get request performance statistics"""
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
    """Advanced image fetcher with multiple fallback strategies"""
    
    def __init__(self):
        self.wikimedia_api = "https://commons.wikimedia.org/w/api.php"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.image_quality_scores = {}
        self.alternative_sources = [
            self.fetch_from_wikimedia,
            self.fetch_from_wikipedia,
            self.fetch_from_flickr,
            self.generate_fallback_image
        ]
    
    def calculate_image_quality(self, image_info: dict) -> int:
        """Calculate quality score for an image (0-100)"""
        score = 50  # Base score
        
        # Size bonus
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        if width >= 1200 and height >= 800:
            score += 30
        elif width >= 800 and height >= 600:
            score += 20
        elif width >= 400 and height >= 300:
            score += 10
        
        # Format bonus
        url = image_info.get('url', '').lower()
        if any(fmt in url for fmt in ['.jpg', '.jpeg']):
            score += 5  # JPG usually good quality
        
        # Aspect ratio penalty for extremes
        if width > 0 and height > 0:
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                score -= 10
        
        # Source bonus
        if image_info.get('source') == 'wikimedia':
            score += 5
        
        return min(100, max(0, score))
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def fetch_from_wikimedia(self, query: str, limit: int = 8) -> List[Dict]:
        """Fetch images from Wikimedia Commons with enhanced filtering"""
        images = []
        
        try:
            params = {
                'action': 'query',
                'generator': 'search',
                'gsrsearch': f'{query} city skyline landscape',
                'gsrnamespace': '6',  # File namespace
                'gsrlimit': 30,
                'prop': 'imageinfo',
                'iiprop': 'url|size|mime|extmetadata',
                'iiurlwidth': 800,
                'format': 'json'
            }
            
            data = request_handler.get_json_cached(
                self.wikimedia_api,
                params=params,
                cache_key=f"wikimedia:{query}",
                ttl=config.CACHE_TTL_IMAGES
            )
            
            for page in data.get('query', {}).get('pages', {}).values():
                if 'imageinfo' in page:
                    info = page['imageinfo'][0]
                    
                    # Enhanced filtering
                    if self._is_high_quality_image(info):
                        image_data = {
                            'url': info.get('thumburl') or info.get('url'),
                            'title': page.get('title', '').replace('File:', ''),
                            'description': self._extract_description(info),
                            'source': 'wikimedia',
                            'width': info.get('width'),
                            'height': info.get('height'),
                            'quality_score': self.calculate_image_quality(info)
                        }
                        
                        if image_data['url'] and image_data['quality_score'] >= config.MIN_IMAGE_QUALITY_SCORE:
                            images.append(image_data)
                            
                            if len(images) >= limit:
                                break
            
            # Sort by quality score
            images.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
        except Exception as e:
            logger.warning(f"Wikimedia fetch failed for {query}: {e}")
        
        return images[:limit]
    
    def fetch_from_wikipedia(self, page_title: str, limit: int = 6) -> List[Dict]:
        """Fetch images from Wikipedia article"""
        images = []
        
        try:
            # Get page images
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
                # Add thumbnail if available
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
                            'quality_score': 80  # Thumbnails are usually good
                        })
                
                # Collect file titles for detailed info
                for img in page.get('images', []):
                    title = img.get('title', '')
                    if title.startswith('File:'):
                        # Filter out unwanted files
                        lower_title = title.lower()
                        if not any(x in lower_title for x in ['.svg', '.ogg', '.webm', '.tif', 'map_']):
                            file_titles.append(title)
            
            # Get detailed info for collected files in batches
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
                                'quality_score': self.calculate_image_quality(info)
                            }
                            
                            if image_data['quality_score'] >= config.MIN_IMAGE_QUALITY_SCORE:
                                images.append(image_data)
                
                if len(images) >= limit:
                    break
            
            # Sort by quality and remove duplicates
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
    
    def fetch_from_flickr(self, query: str, limit: int = 4) -> List[Dict]:
        """Fetch images from Flickr (requires API key)"""
        # Implementation would go here if you add Flickr API key
        return []
    
    def generate_fallback_image(self, city_name: str) -> Dict:
        """Generate a fallback image URL using placeholder services"""
        # Use Unsplash if available
        if config.UNSPLASH_ACCESS_KEY:
            try:
                params = {
                    'query': f'{city_name} city',
                    'orientation': 'landscape',
                    'per_page': 1,
                    'client_id': config.UNSPLASH_ACCESS_KEY
                }
                
                response = request_handler.get_with_retry(
                    'https://api.unsplash.com/search/photos',
                    params=params
                )
                
                data = response.json()
                if data.get('results'):
                    photo = data['results'][0]
                    return {
                        'url': photo['urls']['regular'],
                        'title': f'{city_name} photo',
                        'description': f'Photo of {city_name} from Unsplash',
                        'source': 'unsplash',
                        'width': photo['width'],
                        'height': photo['height'],
                        'quality_score': 70
                    }
            except Exception as e:
                logger.debug(f"Unsplash fallback failed: {e}")
        
        # Use placeholder service
        encoded_city = quote_plus(city_name)
        return {
            'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80&txt={encoded_city}&txt-size=40&txt-color=white&txt-align=middle,center',
            'title': f'{city_name}',
            'description': f'Representation of {city_name}',
            'source': 'placeholder',
            'width': 800,
            'height': 600,
            'quality_score': 30
        }
    
    def get_images_for_city(self, city_name: str, page_title: str = None, 
                           limit: int = None) -> List[Dict]:
        """Main method to get images with multiple fallback strategies"""
        limit = limit or config.MAX_IMAGES_PER_REQUEST
        images = []
        
        # Try multiple strategies
        strategies = [
            (self.fetch_from_wikimedia, [city_name]),
            (self.fetch_from_wikipedia, [page_title or city_name])
        ]
        
        for strategy, args in strategies:
            if len(images) >= limit:
                break
            
            try:
                strategy_images = strategy(*args, limit=limit - len(images))
                # Filter out low quality images
                strategy_images = [img for img in strategy_images 
                                 if img.get('quality_score', 0) >= config.MIN_IMAGE_QUALITY_SCORE]
                images.extend(strategy_images)
            except Exception as e:
                logger.debug(f"Image strategy failed: {e}")
        
        # If still no images, use fallback
        if not images and config.ENABLE_FALLBACK_IMAGES:
            fallback = self.generate_fallback_image(city_name)
            images.append(fallback)
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_images = []
        for img in images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        return unique_images[:limit]
    
    def _is_high_quality_image(self, image_info: dict) -> bool:
        """Determine if image meets quality criteria"""
        mime = image_info.get('mime', '')
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        url = image_info.get('url', '').lower()
        
        # Must be an image
        if not mime.startswith('image/'):
            return False
        
        # Check minimum dimensions
        if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
            return False
        
        # Check format
        if not any(fmt in url for fmt in config.PREFERRED_IMAGE_FORMATS):
            return False
        
        # Check aspect ratio (avoid extreme panoramas or portraits)
        if width > 0 and height > 0:
            ratio = width / height
            if ratio < 0.3 or ratio > 3.0:
                return False
        
        return True
    
    def _extract_description(self, image_info: dict) -> str:
        """Extract description from image metadata"""
        extmetadata = image_info.get('extmetadata', {})
        
        for field in ['ImageDescription', 'ObjectName', 'Caption']:
            if field in extmetadata:
                value = extmetadata[field].get('value', '')
                if isinstance(value, str) and value.strip():
                    # Clean HTML tags
                    clean_value = re.sub(r'<[^>]+>', '', value)
                    return clean_value[:200]  # Limit length
        
        return ""

image_fetcher = IntelligentImageFetcher()

# ==================== ENHANCED CITY DATA PROVIDER ====================
class EnhancedCityDataProvider:
    """Advanced city data provider with improved accuracy and fallbacks"""
    
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
        self.city_coordinates_cache = {}
        self.city_wiki_cache = {}
        self.stats = {
            'coordinates_found': 0,
            'coordinates_failed': 0,
            'wiki_found': 0,
            'wiki_failed': 0,
            'images_found': 0,
            'images_failed': 0
        }
    
    def get_coordinates_enhanced(self, city_name: str, country: str = None, 
                                region: str = None) -> Optional[Tuple[float, float, Dict]]:
        """Get coordinates with multiple fallback strategies"""
        cache_key = f"coords:{city_name}:{country}"
        
        # Check cache first
        cached = cache.get(cache_key)
        if cached:
            self.stats['coordinates_found'] += 1
            return cached
        
        strategies = [
            self._get_coordinates_nominatim,
            self._get_coordinates_wikipedia,
            self._get_coordinates_wikidata,
            self._get_coordinates_from_world_cities
        ]
        
        for strategy in strategies:
            try:
                result = strategy(city_name, country, region)
                if result:
                    lat, lon, metadata = result
                    
                    # Validate coordinates
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
        """Get coordinates using Nominatim with intelligent queries"""
        queries = []
        
        # Build multiple query variations
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
        """Get coordinates from Wikipedia API"""
        try:
            # Try to find the Wikipedia page first
            page = self.wiki.page(city_name)
            if not page.exists():
                # Try with country
                if country:
                    page = self.wiki.page(f"{city_name}, {country}")
            
            if page.exists():
                # Extract coordinates from page
                # Note: wikipedia-api doesn't directly expose coordinates
                # We'll use MediaWiki API instead
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
                            {'source': 'wikipedia', 'page_title': page.title}
                        )
                        
        except Exception as e:
            logger.debug(f"Wikipedia coordinate fetch failed: {e}")
        
        return None
    
    def _get_coordinates_wikidata(self, city_name: str, country: str = None, 
                                 region: str = None) -> Optional[Tuple[float, float, Dict]]:
        """Get coordinates from Wikidata"""
        try:
            # Search for entity
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
                
                # Get entity data with coordinates
                entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                entity_data = request_handler.get_json_cached(
                    entity_url,
                    cache_key=f"wikidata_entity:{entity_id}"
                )
                
                # Extract coordinates (P625 property)
                entity = entity_data.get('entities', {}).get(entity_id, {})
                claims = entity.get('claims', {})
                
                if 'P625' in claims:
                    coord_claim = claims['P625'][0]['mainsnak']['datavalue']['value']
                    return (
                        coord_claim['latitude'],
                        coord_claim['longitude'],
                        {'source': 'wikidata', 'entity_id': entity_id}
                    )
                    
        except Exception as e:
            logger.debug(f"Wikidata coordinate fetch failed: {e}")
        
        return None
    
    def _get_coordinates_from_world_cities(self, city_name: str, country: str = None, 
                                          region: str = None) -> Optional[Tuple[float, float, Dict]]:
        """Fallback to pre-defined world cities data"""
        # This would use your WORLD_CITIES data when you insert it
        # For now, returns None - you'll populate this with your data
        return None
    
    def get_wikipedia_data_enhanced(self, city_name: str, country: str = None) -> Tuple[Optional[Dict], Optional[str]]:
        """Get Wikipedia data with improved matching"""
        cache_key = f"wiki:{city_name}:{country}"
        
        # Check cache
        cached = cache.get(cache_key)
        if cached:
            self.stats['wiki_found'] += 1
            return cached.get('data'), cached.get('title')
        
        # Try multiple page title variations
        variations = self._generate_wiki_variations(city_name, country)
        
        for variation in variations:
            try:
                page = self.wiki.page(variation)
                
                if page.exists() and page.ns == 0:
                    # Check if page is actually about a city
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
        """Generate variations for Wikipedia search"""
        variations = [city_name]
        
        if country:
            variations.extend([
                f"{city_name}, {country}",
                f"{city_name} ({country})",
                f"{city_name} City, {country}"
            ])
        
        # Common variations
        variations.extend([
            f"{city_name} city",
            f"{city_name} (city)",
            f"The city of {city_name}",
            city_name.split(',')[0].strip() if ',' in city_name else city_name
        ])
        
        # Remove duplicates and empty strings
        return list(dict.fromkeys([v for v in variations if v.strip()]))
    
    def _is_city_page(self, page) -> bool:
        """Check if a Wikipedia page is about a city"""
        try:
            text_lower = (page.summary or "").lower()
            
            # Positive indicators
            city_indicators = [
                'city', 'town', 'municipality', 'capital', 'population',
                'located in', 'situated in', 'urban area'
            ]
            
            # Negative indicators (not a city)
            non_city_indicators = [
                'river', 'mountain', 'lake', 'island', 'species',
                'album', 'song', 'film', 'book', 'company'
            ]
            
            city_score = sum(1 for indicator in city_indicators if indicator in text_lower)
            non_city_score = sum(1 for indicator in non_city_indicators if indicator in text_lower)
            
            return city_score > non_city_score and city_score >= 1
            
        except Exception:
            return True  # Default to true if we can't determine
    
    def _extract_wiki_data(self, page) -> Dict:
        """Extract structured data from Wikipedia page"""
        sections = {}
        
        def extract_section_content(section, max_depth=2, current_depth=0):
            if current_depth >= max_depth:
                return
            
            title = section.title.strip()
            text = (section.text or "").strip()
            
            if title and text and title not in ["See also", "References", "External links", "Notes"]:
                # Clean text
                cleaned = re.sub(r'\[\d+\]', '', text)
                cleaned = re.sub(r'\{\{.*?\}\}', '', cleaned)  # Remove templates
                
                if cleaned:
                    sections[title] = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
            
            # Recurse into subsections
            for subsection in getattr(section, 'sections', []):
                extract_section_content(subsection, max_depth, current_depth + 1)
        
        # Extract top-level sections
        for section in getattr(page, 'sections', []):
            extract_section_content(section)
        
        return {
            'title': page.title,
            'summary': (page.summary or "")[:1000],
            'fullurl': getattr(page, 'fullurl', f"https://en.wikipedia.org/wiki/{quote_plus(page.title)}"),
            'sections': sections,
            'pageid': getattr(page, 'pageid', None)
        }
    
    def get_city_tagline_enhanced(self, city_name: str, country: str = None) -> Dict[str, str]:
        """Generate enhanced taglines for cities"""
        cache_key = f"tagline:{city_name}:{country}"
        
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # 1. Check hardcoded famous taglines
        tagline = self._get_hardcoded_tagline(city_name)
        if tagline:
            result = {"city": city_name, "tagline": tagline, "source": "known"}
            cache.set(cache_key, result, config.CACHE_TTL)
            return result
        
        # 2. Try to extract from Wikipedia summary
        try:
            page_data, _ = self.get_wikipedia_data_enhanced(city_name, country)
            if page_data and page_data.get('summary'):
                tagline = self._generate_tagline_from_summary(page_data['summary'])
                if tagline:
                    result = {"city": city_name, "tagline": tagline, "source": "wikipedia"}
                    cache.set(cache_key, result, config.CACHE_TTL)
                    return result
        except Exception as e:
            logger.debug(f"Tagline extraction failed: {e}")
        
        # 3. Generate based on country/region
        tagline = self._generate_contextual_tagline(city_name, country)
        result = {"city": city_name, "tagline": tagline, "source": "generated"}
        cache.set(cache_key, result, config.CACHE_TTL)
        return result
    
    def _get_hardcoded_tagline(self, city_name: str) -> Optional[str]:
        """Get pre-defined tagline for famous cities"""
        taglines = {
            "Paris": "The City of Light",
            "New York": "The Big Apple",
            "London": "The Old Smoke",
            "Tokyo": "The Eastern Capital",
            "Rome": "The Eternal City",
            "Dubai": "The City of Gold",
            "Venice": "The Floating City",
            "Barcelona": "The City of Gaudí",
            "Istanbul": "Where East Meets West",
            "Amsterdam": "The Venice of the North",
            "Prague": "The City of a Hundred Spires",
            "Vienna": "The City of Music",
            "Rio de Janeiro": "The Marvelous City",
            "San Francisco": "The Golden City",
            "Las Vegas": "The Entertainment Capital of the World",
            "Bangkok": "The City of Angels",
            "Singapore": "The Lion City",
            "Jerusalem": "The Holy City",
            "Moscow": "The Third Rome",
            "Beijing": "The Forbidden City"
        }
        
        # Try exact match first, then case-insensitive
        if city_name in taglines:
            return taglines[city_name]
        
        for key, value in taglines.items():
            if city_name.lower() == key.lower():
                return value
        
        return None
    
    def _generate_tagline_from_summary(self, summary: str) -> Optional[str]:
        """Generate tagline from Wikipedia summary"""
        if not summary:
            return None
        
        # Clean the summary
        clean = re.sub(r'\[\d+\]', '', summary)
        clean = re.sub(r'\([^)]*\)', '', clean)  # Remove parentheses
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Take first sentence or first 12 words
        sentences = re.split(r'[.!?]', clean)
        if sentences and sentences[0]:
            first_sentence = sentences[0].strip()
            words = first_sentence.split()
            
            if len(words) > 15:
                # Take key phrase
                return " ".join(words[:12]) + "..."
            else:
                return first_sentence
        
        return None
    
    def _generate_contextual_tagline(self, city_name: str, country: str = None) -> str:
        """Generate a contextual tagline based on location"""
        if country:
            # Country-specific taglines
            country_taglines = {
                "Italy": f"A beautiful Italian city",
                "France": f"A charming French city",
                "Japan": f"A vibrant Japanese city",
                "USA": f"An American city full of opportunities",
                "Spain": f"A sunny Spanish destination",
                "Germany": f"A historic German city",
                "UK": f"A classic British destination",
                "Greece": f"A city with ancient Greek heritage",
                "Thailand": f"A exotic Thai destination",
                "Australia": f"A down under adventure"
            }
            
            if country in country_taglines:
                return country_taglines[country]
        
        # Generic fallback
        descriptors = [
            "A beautiful destination worth exploring",
            "A city full of charm and character",
            "An unforgettable travel experience",
            "A place of rich culture and history",
            "A hidden gem waiting to be discovered",
            "A vibrant urban adventure",
            "A journey through time and tradition"
        ]
        
        import random
        return random.choice(descriptors)
    
    def get_city_preview_enhanced(self, city_name: str, country: str = None, 
                                 region: str = None) -> Dict:
        """Get enhanced city preview with guaranteed image and coordinates"""
        # Generate cache key
        cache_key = f"preview:{city_name}:{country}:{region}"
        
        # Check cache first
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        logger.info(f"🔄 Generating preview for {city_name}")
        
        # Start building preview
        preview = {
            "id": self._generate_city_id(city_name),
            "name": city_name,
            "display_name": city_name,
            "summary": "Loading city information...",
            "has_details": True,  # Assume true, will be updated
            "image": None,
            "images": [],
            "coordinates": None,
            "static_map": None,
            "tagline": None,
            "tagline_source": "loading",
            "last_updated": time.time(),
            "country": country,
            "region": region,
            "metadata": {
                "image_quality": "unknown",
                "coordinate_accuracy": "unknown",
                "data_completeness": 0
            }
        }
        
        # Step 1: Get coordinates (parallelizable with other tasks)
        coordinates_future = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit coordinates task
            coordinates_future = executor.submit(
                self.get_coordinates_enhanced, city_name, country, region
            )
            
            # Step 2: Get Wikipedia data in parallel
            wiki_future = executor.submit(
                self.get_wikipedia_data_enhanced, city_name, country
            )
            
            # Get results
            try:
                coords_result = coordinates_future.result(timeout=10)
                if coords_result:
                    lat, lon, metadata = coords_result
                    preview["coordinates"] = {"lat": lat, "lon": lon}
                    preview["metadata"]["coordinate_accuracy"] = metadata.get('source', 'unknown')
                    
                    # Generate static map
                    preview["static_map"] = self.map_provider.generate_static_map_url(
                        {"lat": lat, "lon": lon}, width=400, height=250
                    )
            except Exception as e:
                logger.warning(f"Coordinates fetch failed for {city_name}: {e}")
            
            try:
                wiki_data, wiki_title = wiki_future.result(timeout=10)
                if wiki_data:
                    preview["display_name"] = wiki_data.get('title', city_name)
                    preview["summary"] = (wiki_data.get('summary', '')[:150] + '...') if wiki_data.get('summary') else "A fascinating city to explore"
                    preview["has_details"] = True
                    
                    # Store wiki title for image fetching
                    preview["_wiki_title"] = wiki_title or city_name
                else:
                    preview["has_details"] = False
            except Exception as e:
                logger.warning(f"Wikipedia fetch failed for {city_name}: {e}")
                preview["has_details"] = False
        
        # Step 3: Get images (most important for user experience)
        try:
            wiki_title = preview.get("_wiki_title", city_name)
            images = image_fetcher.get_images_for_city(
                city_name, 
                wiki_title, 
                limit=min(3, config.MAX_IMAGES_PER_REQUEST)
            )
            
            if images:
                preview["images"] = images
                preview["image"] = images[0]  # Best image first
                
                # Calculate image quality score
                quality_scores = [img.get('quality_score', 0) for img in images]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                
                if avg_quality >= 70:
                    preview["metadata"]["image_quality"] = "excellent"
                elif avg_quality >= 50:
                    preview["metadata"]["image_quality"] = "good"
                elif avg_quality >= 30:
                    preview["metadata"]["image_quality"] = "fair"
                else:
                    preview["metadata"]["image_quality"] = "basic"
                
                self.stats['images_found'] += 1
            else:
                logger.warning(f"No images found for {city_name}")
                self.stats['images_failed'] += 1
                
        except Exception as e:
            logger.error(f"Image fetch failed for {city_name}: {e}")
            self.stats['images_failed'] += 1
        
        # Step 4: Get tagline
        try:
            tagline_data = self.get_city_tagline_enhanced(city_name, country)
            preview["tagline"] = tagline_data.get("tagline")
            preview["tagline_source"] = tagline_data.get("source")
        except Exception as e:
            logger.debug(f"Tagline fetch failed: {e}")
            preview["tagline"] = "A beautiful destination worth exploring"
            preview["tagline_source"] = "default"
        
        # Step 5: Calculate data completeness
        completeness_score = 0
        if preview.get("coordinates"): completeness_score += 30
        if preview.get("image"): completeness_score += 40
        if preview.get("has_details"): completeness_score += 20
        if preview.get("tagline"): completeness_score += 10
        
        preview["metadata"]["data_completeness"] = completeness_score
        
        # Step 6: Ensure we have at least a placeholder image
        if not preview.get("image"):
            fallback_image = image_fetcher.generate_fallback_image(city_name)
            preview["image"] = fallback_image
            preview["images"] = [fallback_image]
        
        # Step 7: Ensure we have a static map (even if no coordinates)
        if not preview.get("static_map"):
            preview["static_map"] = self.map_provider.generate_static_map_url(
                preview.get("coordinates"), width=400, height=250
            )
        
        # Cache the result
        cache.set(cache_key, preview, config.CACHE_TTL)
        
        logger.info(f"✅ Preview generated for {city_name} (completeness: {completeness_score}%)")
        return preview
    
    def get_city_details_enhanced(self, city_name: str, country: str = None, 
                                 region: str = None) -> Dict:
        """Get comprehensive city details"""
        cache_key = f"details:{city_name}:{country}:{region}"
        
        # Check cache
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        logger.info(f"🔄 Generating detailed data for {city_name}")
        
        # Start with preview data
        preview = self.get_city_preview_enhanced(city_name, country, region)
        
        # Build detailed response
        details = {
            **preview,
            "detailed_summary": "",
            "sections": [],
            "landmarks": [],
            "culture": {},
            "transportation": {},
            "best_time_to_visit": "",
            "sources": [],
            "map_config": {},
            "additional_images": [],
            "statistics": {},
            "nearby_cities": []
        }
        
        # Get Wikipedia data for details
        wiki_data, wiki_title = self.get_wikipedia_data_enhanced(city_name, country)
        if wiki_data:
            details["detailed_summary"] = wiki_data.get('summary', '')[:2000]
            details["sources"].append(wiki_data.get('fullurl', ''))
            
            # Structure sections
            sections_data = []
            for section_title, content in wiki_data.get('sections', {}).items():
                if content.strip():
                    sections_data.append({
                        "title": section_title,
                        "content": content[:800] + ("..." if len(content) > 800 else "")
                    })
            details["sections"] = sections_data[:6]  # Limit to 6 sections
        
        # Get additional images
        try:
            additional_images = image_fetcher.get_images_for_city(
                city_name,
                wiki_title or city_name,
                limit=config.MAX_IMAGES_PER_REQUEST
            )
            
            if additional_images:
                # Use first image as main, rest as additional
                if additional_images and not details.get("image"):
                    details["image"] = additional_images[0]
                
                details["additional_images"] = additional_images[1:6]  # Next 5 images
                details["images"] = additional_images  # All images
        except Exception as e:
            logger.warning(f"Additional images failed for {city_name}: {e}")
        
        # Generate map configuration
        details["map_config"] = self.map_provider.get_map_config(
            city_name,
            details.get("coordinates")
        )
        
        # Add statistics
        details["statistics"] = {
            "image_count": len(details.get("images", [])),
            "section_count": len(details.get("sections", [])),
            "data_quality": preview.get("metadata", {}).get("data_completeness", 0),
            "last_updated": time.time()
        }
        
        # Cache the result
        cache.set(cache_key, details, config.CACHE_TTL)
        
        logger.info(f"✅ Details generated for {city_name}")
        return details
    
    def _generate_city_id(self, city_name: str) -> str:
        """Generate a URL-safe city ID"""
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        city_id = city_name.lower().strip()
        city_id = re.sub(r'[^\w\s-]', '', city_id)  # Remove special characters
        city_id = re.sub(r'[-\s]+', '-', city_id)   # Replace spaces and multiple hyphens
        return city_id
    
    def get_stats(self):
        """Get provider statistics"""
        return self.stats

# ==================== MAP PROVIDER (UPDATED) ====================
class MapProvider:
    """Enhanced map provider with better fallbacks"""
    
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
        
        # Add Mapbox if token is available
        if config.MAPBOX_ACCESS_TOKEN:
            self.tile_providers["mapbox"] = {
                "url": f"https://api.mapbox.com/styles/v1/mapbox/light-v10/tiles/{{z}}/{{x}}/{{y}}?access_token={config.MAPBOX_ACCESS_TOKEN}",
                "attribution": "© Mapbox & OpenStreetMap",
                "requires_token": True
            }
    
    def get_map_config(self, city_name: str, coordinates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get map configuration with fallbacks"""
        provider_key = config.MAP_TILE_PROVIDER
        
        # Validate provider or fallback to openstreetmap
        if provider_key not in self.tile_providers:
            provider_key = "openstreetmap"
            logger.warning(f"Map provider {config.MAP_TILE_PROVIDER} not found, using {provider_key}")
        
        provider_config = self.tile_providers[provider_key]
        
        # Base configuration
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
        
        # Add coordinates if valid
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
        """Generate static map URL with multiple fallback options"""
        
        # If no coordinates, return a generic world map
        if not coordinates or not self._validate_coordinates(coordinates):
            return "https://via.placeholder.com/{width}x{height}.png?text=Map+Not+Available".format(
                width=width, height=height
            )
        
        lat, lon = coordinates["lat"], coordinates["lon"]
        
        # Try Mapbox first if available
        if config.MAPBOX_ACCESS_TOKEN:
            try:
                return f"https://api.mapbox.com/styles/v1/mapbox/light-v10/static/pin-l+3388ff({lon},{lat})/{lon},{lat},{zoom}/{width}x{height}?access_token={config.MAPBOX_ACCESS_TOKEN}"
            except Exception:
                pass
        
        # Try OpenStreetMap static
        try:
            # StaticMap from OSM
            return f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={width}x{height}&markers={lat},{lon},red-pushpin&scale=2"
        except Exception:
            pass
        
        # Fallback to placeholder with coordinates
        return f"https://via.placeholder.com/{width}x{height}.png?text={lat:.4f}%2C{lon:.4f}"
    
    def _validate_coordinates(self, coordinates: Dict[str, float]) -> bool:
        """Robust coordinate validation"""
        try:
            lat = coordinates.get("lat")
            lon = coordinates.get("lon")
            
            if lat is None or lon is None:
                return False
            
            # Convert to float
            try:
                lat = float(lat)
                lon = float(lon)
            except (ValueError, TypeError):
                return False
            
            # Check valid ranges
            return (-90 <= lat <= 90 and -180 <= lon <= 180)
            
        except Exception:
            return False

# ==================== CITY LOADING MANAGER ====================
class CityLoadingManager:
    """Manages loading of 1500+ cities with optimized performance"""
    
    def __init__(self, data_provider: EnhancedCityDataProvider):
        self.data_provider = data_provider
        self.loaded_cities = {}
        self.loading_status = {
            'total': 0,
            'loaded': 0,
            'failed': 0,
            'with_images': 0,
            'with_coordinates': 0,
            'start_time': None,
            'estimated_completion': None
        }
        self.loading_queue = []
        self.is_loading = False
        self.loading_thread = None
        
    def initialize_with_world_cities(self, world_cities_data: List[Dict]):
        """Initialize with your world cities data"""
        if not world_cities_data:
            logger.error("❌ No world cities data provided!")
            return
        
        self.loading_status['total'] = len(world_cities_data)
        self.loading_status['start_time'] = time.time()
        self.loading_status['loaded'] = 0
        self.loading_status['failed'] = 0
        self.loading_status['with_images'] = 0
        self.loading_status['with_coordinates'] = 0
        
        # Create loading queue
        self.loading_queue = []
        for city_data in world_cities_data:
            self.loading_queue.append({
                'name': city_data['name'],
                'country': city_data.get('country'),
                'region': city_data.get('region'),
                'priority': self._calculate_priority(city_data)
            })
        
        # Sort by priority (popular cities first)
        self.loading_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"📊 Initialized loading manager with {len(self.loading_queue)} cities")
        logger.info(f"📊 First 5 cities to load: {[c['name'] for c in self.loading_queue[:5]]}")
    
    def _calculate_priority(self, city_data: Dict) -> int:
        """Calculate loading priority for a city"""
        priority = 50  # Base priority
        
        # Boost priority for major cities
        major_cities = ['Paris', 'London', 'New York', 'Tokyo', 'Rome', 
                       'Dubai', 'Sydney', 'Barcelona', 'Amsterdam', 'Berlin']
        
        if city_data['name'] in major_cities:
            priority += 50
        
        # Boost for capitals
        if 'capital' in city_data.get('name', '').lower():
            priority += 30
        
        # Boost for cities with known country
        if city_data.get('country'):
            priority += 10
        
        return priority
    
    def start_loading(self, batch_size: int = None):
        """Start loading cities in batches"""
        if self.is_loading:
            return
        
        self.is_loading = True
        batch_size = batch_size or config.BATCH_SIZE
        
        def load_task():
            logger.info(f"🚀 Starting city loading (batch size: {batch_size})")
            
            total_batches = (len(self.loading_queue) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                if not self.is_loading:
                    break
                
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(self.loading_queue))
                batch = self.loading_queue[start_idx:end_idx]
                
                logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} cities)")
                
                # Load batch in parallel
                with ThreadPoolExecutor(max_workers=min(config.MAX_PRELOAD_WORKERS, len(batch))) as executor:
                    futures = {}
                    
                    for city_info in batch:
                        future = executor.submit(
                            self.data_provider.get_city_preview_enhanced,
                            city_info['name'],
                            city_info.get('country'),
                            city_info.get('region')
                        )
                        futures[future] = city_info
                    
                    # Process results
                    for future in as_completed(futures):
                        city_info = futures[future]
                        
                        try:
                            city_preview = future.result(timeout=30)
                            city_name = city_info['name']
                            
                            # Store in loaded cities
                            self.loaded_cities[city_name] = city_preview
                            self.loading_status['loaded'] += 1
                            
                            # Update statistics
                            if city_preview.get('image') and city_preview['image'].get('url'):
                                if 'placeholder.com' not in city_preview['image']['url']:
                                    self.loading_status['with_images'] += 1
                            
                            if city_preview.get('coordinates'):
                                self.loading_status['with_coordinates'] += 1
                            
                            # Calculate success rate
                            success_rate = self.loading_status['loaded'] / self.loading_status['total']
                            
                            # Log progress every 50 cities
                            if self.loading_status['loaded'] % 50 == 0:
                                elapsed = time.time() - self.loading_status['start_time']
                                cities_per_second = self.loading_status['loaded'] / elapsed if elapsed > 0 else 0
                                
                                logger.info(
                                    f"📊 Progress: {self.loading_status['loaded']}/{self.loading_status['total']} "
                                    f"({success_rate:.1%}) | "
                                    f"Images: {self.loading_status['with_images']} | "
                                    f"Coords: {self.loading_status['with_coordinates']} | "
                                    f"Speed: {cities_per_second:.2f} cities/sec"
                                )
                            
                        except Exception as e:
                            self.loading_status['failed'] += 1
                            logger.warning(f"Failed to load {city_info['name']}: {e}")
                
                # Small delay between batches to be nice to APIs
                if batch_num < total_batches - 1:
                    time.sleep(2)
            
            # Loading complete
            self.is_loading = False
            total_time = time.time() - self.loading_status['start_time']
            
            logger.info(f"✅ City loading completed!")
            logger.info(f"   Total cities: {self.loading_status['loaded']}")
            logger.info(f"   With images: {self.loading_status['with_images']} ({(self.loading_status['with_images']/self.loading_status['loaded']*100):.1f}%)")
            logger.info(f"   With coordinates: {self.loading_status['with_coordinates']} ({(self.loading_status['with_coordinates']/self.loading_status['loaded']*100):.1f}%)")
            logger.info(f"   Failed: {self.loading_status['failed']}")
            logger.info(f"   Total time: {total_time:.2f} seconds")
            logger.info(f"   Average: {self.loading_status['loaded']/total_time:.2f} cities/second")
            
            # Check if we met quality thresholds
            image_success_rate = self.loading_status['with_images'] / self.loading_status['loaded']
            if image_success_rate < config.REQUIRED_SUCCESS_RATE:
                logger.warning(f"⚠️ Image success rate ({image_success_rate:.1%}) below target ({config.REQUIRED_SUCCESS_RATE:.0%})")
        
        # Start loading in background thread
        self.loading_thread = threading.Thread(target=load_task, daemon=True)
        self.loading_thread.start()
    
    def get_loading_status(self):
        """Get current loading status"""
        if self.loading_status['start_time']:
            elapsed = time.time() - self.loading_status['start_time']
            if self.loading_status['loaded'] > 0:
                estimated_total = elapsed * self.loading_status['total'] / self.loading_status['loaded']
                remaining = max(0, estimated_total - elapsed)
                self.loading_status['estimated_completion'] = remaining
            else:
                self.loading_status['estimated_completion'] = None
        
        return self.loading_status
    
    def get_city(self, city_name: str) -> Optional[Dict]:
        """Get a city from loaded data"""
        return self.loaded_cities.get(city_name)
    
    def search_cities(self, query: str, limit: int = 20) -> List[Dict]:
        """Search through loaded cities"""
        if not query or len(query) < 2:
            return []
        
        query_lower = query.lower()
        results = []
        
        for city_name, city_data in self.loaded_cities.items():
            if query_lower in city_name.lower():
                results.append(city_data)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_cities_by_region(self, region: str, limit: int = 50) -> List[Dict]:
        """Get cities by region"""
        results = []
        
        for city_name, city_data in self.loaded_cities.items():
            if city_data.get('region') == region:
                results.append(city_data)
            
            if len(results) >= limit:
                break
        
        return results

# ==================== FLASK APP & ROUTES ====================
app = Flask(__name__)

# Configure CORS for your frontend
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

# Global variables
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
REGIONS = set(["Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"])

# ==================== API ROUTES ====================
@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "name": "City Explorer API",
        "version": "2.0.0",
        "status": "operational",
        "frontend": "https://www.traveltto.com",
        "endpoints": {
            "health": "/api/health",
            "cities": "/api/cities",
            "city_details": "/api/cities/<city_name>",
            "search": "/api/search",
            "regions": "/api/regions",
            "stats": "/api/stats",
            "cache": "/api/cache",
            "reload": "/api/reload"
        },
        "documentation": "https://github.com/yourusername/city-explorer-api"
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    loader_status = city_loader.get_loading_status()
    cache_stats = cache.get_stats()
    request_stats = request_handler.get_performance_stats()
    provider_stats = data_provider.get_stats()
    
    # Calculate memory usage
    def get_memory_usage():
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": "production",
        "deployed_on": "vercel",
        
        "city_loading": {
            "total_cities": loader_status.get('total', 0),
            "loaded": loader_status.get('loaded', 0),
            "loading_in_progress": city_loader.is_loading,
            "with_images": loader_status.get('with_images', 0),
            "with_coordinates": loader_status.get('with_coordinates', 0),
            "success_rate": f"{(loader_status.get('loaded', 0) / max(loader_status.get('total', 1), 1) * 100):.1f}%",
            "estimated_completion_seconds": loader_status.get('estimated_completion')
        },
        
        "cache": cache_stats,
        
        "performance": {
            **request_stats,
            "active_threads": threading.active_count(),
            "memory_usage_mb": get_memory_usage()
        },
        
        "provider_stats": provider_stats,
        
        "configuration": {
            "cache_ttl": config.CACHE_TTL,
            "image_workers": config.MAX_IMAGE_WORKERS,
            "preload_cities": config.PRELOAD_TOP_CITIES,
            "lazy_loading": config.LAZY_LOADING,
            "min_image_quality": config.MIN_IMAGE_QUALITY_SCORE
        }
    })

@app.route('/api/cities')
def get_cities():
    """Get all cities with pagination"""
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    region = request.args.get('region', type=str)
    country = request.args.get('country', type=str)
    
    # Validate pagination
    page = max(1, page)
    limit = min(max(1, limit), 100)  # Cap at 100 per page
    
    # FIX: If loader hasn't been initialized, initialize it now
    if city_loader.loading_status['total'] == 0:
        try:
            # Extract regions
            global REGIONS
            REGIONS.clear()
            for city in WORLD_CITIES:
                if 'region' in city:
                    REGIONS.add(city['region'])
            
            # Ensure basic regions
            basic_regions = {"Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"}
            REGIONS.update(basic_regions)
            
            city_loader.initialize_with_world_cities(WORLD_CITIES)
            logger.info(f"📊 Lazy initialization: Loaded {len(WORLD_CITIES)} cities")
            
            # Start loading a small batch
            if not city_loader.is_loading:
                city_loader.start_loading(batch_size=min(10, config.PRELOAD_TOP_CITIES))
        except Exception as e:
            logger.error(f"❌ Failed to initialize city loader: {e}")
            return jsonify({
                "success": False,
                "error": "City data not initialized",
                "total_cities_in_dataset": len(WORLD_CITIES)
            }), 500
    
    # Start loading if not started and we have data
    if not city_loader.is_loading and city_loader.loading_status['total'] > 0:
        if city_loader.loading_status['loaded'] == 0:
            # Start with a small batch
            city_loader.start_loading(batch_size=min(10, config.PRELOAD_TOP_CITIES))
    
    # FIX: Always use WORLD_CITIES count as total
    total_cities_in_dataset = len(WORLD_CITIES)
    loaded_cities_count = city_loader.loading_status['loaded']
    
    # Get cities based on filters
    if region or country:
        filtered_cities = []
        loaded_cities = city_loader.loaded_cities
        
        for city_name, city_data in loaded_cities.items():
            if region and city_data.get('region') != region:
                continue
            if country and city_data.get('country') != country:
                continue
            filtered_cities.append(city_data)
        
        cities_list = filtered_cities
        filtered_total = len(filtered_cities)
    else:
        # Return loaded cities
        cities_list = list(city_loader.loaded_cities.values())
        filtered_total = loaded_cities_count
    
    # Apply pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_cities = cities_list[start_idx:end_idx]
    
    # Calculate pagination metadata
    # Use the actual total from WORLD_CITIES for pagination info
    total_pages = max(1, (total_cities_in_dataset + limit - 1) // limit)
    
    # Update loading stats
    loading_status = city_loader.get_loading_status()
    # Make sure total shows the real number
    loading_status['total'] = total_cities_in_dataset
    
    return jsonify({
        "success": True,
        "data": paginated_cities,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_cities_in_dataset,  # Show actual total from WORLD_CITIES
            "pages": total_pages,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None
        },
        "loading": {
            "complete": not city_loader.is_loading,
            "loaded": loaded_cities_count,
            "total": total_cities_in_dataset,  # Show actual total
            "progress": f"{(loaded_cities_count / max(total_cities_in_dataset, 1) * 100):.1f}%"
        },
        "info": {
            "total_cities_in_dataset": total_cities_in_dataset,
            "cities_loaded_so_far": loaded_cities_count,
            "loading_in_progress": city_loader.is_loading,
            "regions_available": list(REGIONS)
        }
    })
@app.route('/api/debug')
def debug():
    """Debug endpoint to check initialization"""
    return jsonify({
        "success": True,
        "world_cities_count": len(WORLD_CITIES),
        "first_5_cities": WORLD_CITIES[:5],
        "loader_initialized": city_loader.loading_status['total'] > 0,
        "loader_total": city_loader.loading_status['total'],
        "loader_loaded": city_loader.loading_status['loaded'],
        "loaded_cities_count": len(city_loader.loaded_cities),
        "is_loading": city_loader.is_loading,
        "regions": list(REGIONS)
    })

@app.route('/api/simple-cities')
def simple_cities():
    """Simple endpoint that returns city names without loading"""
    return jsonify({
        "success": True,
        "total": len(WORLD_CITIES),
        "data": [{"name": city["name"], "country": city.get("country"), "region": city.get("region")} for city in WORLD_CITIES[:50]],
        "message": f"Showing first 50 of {len(WORLD_CITIES)} cities. Use /api/cities for full data."
    })

@app.route('/api/cities/<path:city_name>')
def get_city(city_name):
    """Get detailed information for a specific city"""
    city_name = unquote(city_name)
    
    # Find the city in your data
    city_info = None
    for city in WORLD_CITIES:
        if city['name'].lower() == city_name.lower():
            city_info = city
            break
    
    if not city_info:
        return jsonify({
            "success": False,
            "error": "City not found",
            "suggestions": city_loader.search_cities(city_name, limit=5) if hasattr(city_loader, 'search_cities') else []
        }), 404
    
    # Get city details
    try:
        details = data_provider.get_city_details_enhanced(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        
        return jsonify({
            "success": True,
            "data": details
        })
        
    except Exception as e:
        logger.error(f"Failed to get details for {city_name}: {e}")
        
        # Try to return at least preview data
        preview = city_loader.get_city(city_info['name']) if hasattr(city_loader, 'get_city') else None
        if preview:
            return jsonify({
                "success": True,
                "data": preview,
                "warning": "Full details unavailable, showing preview only"
            })
        
        return jsonify({
            "success": False,
            "error": "Failed to fetch city details"
        }), 500

@app.route('/api/search')
def search_cities():
    """Search for cities"""
    query = request.args.get('q', '').strip()
    
    if len(query) < 2:
        return jsonify({
            "success": False,
            "error": "Search query must be at least 2 characters"
        }), 400
    
    limit = request.args.get('limit', 20, type=int)
    limit = min(max(1, limit), 50)
    
    # Search in loaded cities
    results = []
    if hasattr(city_loader, 'search_cities'):
        results = city_loader.search_cities(query, limit)
    
    # If we have few results and loading is complete, search in world cities directly
    if len(results) < 5 and not city_loader.is_loading:
        for city in WORLD_CITIES:
            if query.lower() in city['name'].lower():
                # Load this city on demand
                try:
                    city_data = data_provider.get_city_preview_enhanced(
                        city['name'],
                        city.get('country'),
                        city.get('region')
                    )
                    results.append(city_data)
                except Exception as e:
                    logger.debug(f"Failed to load city {city['name']}: {e}")
                    pass
                
                if len(results) >= limit:
                    break
    
    return jsonify({
        "success": True,
        "query": query,
        "count": len(results),
        "data": results
    })

@app.route('/api/regions')
def get_regions():
    """Get all available regions"""
    regions = list(REGIONS)
    regions.sort()
    
    # Count cities per region
    region_stats = {}
    for city in WORLD_CITIES:
        region = city.get('region')
        if region:
            region_stats[region] = region_stats.get(region, 0) + 1
    
    return jsonify({
        "success": True,
        "count": len(regions),
        "data": regions,
        "stats": region_stats
    })

@app.route('/api/stats')
def get_stats():
    """Get API statistics"""
    loader_status = city_loader.get_loading_status()
    cache_stats = cache.get_stats()
    provider_stats = data_provider.get_stats()
    request_stats = request_handler.get_performance_stats()
    
    # Calculate image success rate
    total_cities = max(loader_status.get('loaded', 1), 1)
    image_success_rate = loader_status.get('with_images', 0) / total_cities
    
    def get_improvement_suggestions(success_rate: float) -> List[str]:
        """Get suggestions for improving data quality"""
        suggestions = []
        
        if success_rate < config.REQUIRED_SUCCESS_RATE:
            suggestions.append("Consider adding Unsplash API key for fallback images")
            suggestions.append("Increase WIKIMEDIA_RETRY_ATTEMPTS for better image fetching")
            suggestions.append("Lower MIN_IMAGE_QUALITY_SCORE to accept more images")
        
        coord_rate = loader_status.get('with_coordinates', 0) / total_cities
        if coord_rate < 0.8:
            suggestions.append("Add more coordinate sources or improve geocoding queries")
        
        return suggestions
    
    return jsonify({
        "city_statistics": {
            "total_cities": len(WORLD_CITIES),
            "loaded_cities": loader_status.get('loaded', 0),
            "loading_in_progress": city_loader.is_loading,
            "cities_with_images": loader_status.get('with_images', 0),
            "image_success_rate": f"{image_success_rate:.1%}",
            "cities_with_coordinates": loader_status.get('with_coordinates', 0),
            "coordinate_success_rate": f"{loader_status.get('with_coordinates', 0) / total_cities:.1%}",
            "estimated_loading_time_seconds": loader_status.get('estimated_completion')
        },
        "cache_statistics": cache_stats,
        "provider_statistics": provider_stats,
        "performance": request_stats,
        "quality_metrics": {
            "target_image_success_rate": f"{config.REQUIRED_SUCCESS_RATE:.0%}",
            "minimum_image_quality_score": config.MIN_IMAGE_QUALITY_SCORE,
            "current_image_quality": "good" if image_success_rate >= config.REQUIRED_SUCCESS_RATE else "needs_improvement",
            "suggestions": get_improvement_suggestions(image_success_rate)
        }
    })

@app.route('/api/cache')
def cache_info():
    """Get cache information and management"""
    action = request.args.get('action', 'info')
    
    if action == 'clear':
        # Clear specific cache or all
        city_name = request.args.get('city')
        
        if city_name:
            # Clear cache for specific city
            keys_to_clear = [
                f"preview:{city_name}",
                f"details:{city_name}",
                f"coords:{city_name}",
                f"wiki:{city_name}",
                f"tagline:{city_name}"
            ]
            
            for key in keys_to_clear:
                # Implementation depends on your cache system
                # For diskcache, you can delete by key
                try:
                    if hasattr(cache, 'delete'):
                        cache.delete(key)
                    elif hasattr(cache, 'set'):
                        # Set to None with immediate expiry
                        cache.set(key, None, expire=0)
                except Exception as e:
                    logger.debug(f"Failed to clear cache key {key}: {e}")
            
            return jsonify({
                "success": True,
                "message": f"Cache cleared for {city_name}"
            })
        else:
            # Clear all cache
            try:
                if hasattr(cache, 'clear'):
                    cache.clear()
                elif hasattr(cache.disk_cache, 'clear'):
                    cache.disk_cache.clear()
                # Clear memory cache
                if hasattr(cache, 'memory_cache'):
                    cache.memory_cache.clear()
            except Exception as e:
                logger.error(f"Failed to clear all cache: {e}")
            
            return jsonify({
                "success": True,
                "message": "All cache cleared"
            })
    
    else:
        # Return cache info
        cache_stats = cache.get_stats()
        
        return jsonify({
            "success": True,
            "cache": cache_stats,
            "configuration": {
                "cache_ttl_seconds": config.CACHE_TTL,
                "image_cache_ttl_seconds": config.CACHE_TTL_IMAGES,
                "coordinate_cache_ttl_seconds": config.CACHE_TTL_COORDS,
                "cache_directory": config.CACHE_DIR
            }
        })

@app.route('/api/reload')
def reload_city():
    """Force reload of a specific city"""
    city_name = request.args.get('city', '').strip()
    
    if not city_name:
        return jsonify({
            "success": False,
            "error": "City name is required"
        }), 400
    
    # Find city info
    city_info = None
    for city in WORLD_CITIES:
        if city['name'].lower() == city_name.lower():
            city_info = city
            break
    
    if not city_info:
        return jsonify({
            "success": False,
            "error": "City not found"
        }), 404
    
    try:
        # Clear cache for this city
        cache_keys = [
            f"preview:{city_name}",
            f"details:{city_name}",
            f"coords:{city_name}:{city_info.get('country')}",
            f"wiki:{city_name}:{city_info.get('country')}",
            f"tagline:{city_name}:{city_info.get('country')}"
        ]
        
        for key in cache_keys:
            try:
                if hasattr(cache, 'delete'):
                    cache.delete(key)
                elif hasattr(cache, 'set'):
                    cache.set(key, None, expire=0)
            except Exception:
                pass
        
        # Reload city data
        details = data_provider.get_city_details_enhanced(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        
        # Update in loader if exists
        if hasattr(city_loader, 'loaded_cities') and city_name in city_loader.loaded_cities:
            city_loader.loaded_cities[city_name] = data_provider.get_city_preview_enhanced(
                city_info['name'],
                city_info.get('country'),
                city_info.get('region')
            )
        
        return jsonify({
            "success": True,
            "message": f"City {city_name} reloaded successfully",
            "data": details
        })
        
    except Exception as e:
        logger.error(f"Failed to reload city {city_name}: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to reload city: {str(e)}"
        }), 500

# ==================== HELPER FUNCTIONS ====================
def initialize_city_data():
    """Initialize the city data (call this with your WORLD_CITIES)"""
    global REGIONS
    
    # Clear and extract unique regions from WORLD_CITIES
    REGIONS.clear()
    for city in WORLD_CITIES:
        if 'region' in city:
            REGIONS.add(city['region'])
    
    # Ensure we have the basic regions even if no cities for them
    basic_regions = {"Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"}
    REGIONS.update(basic_regions)
    
    # Initialize city loader
    if len(WORLD_CITIES) > 0:
        try:
            city_loader.initialize_with_world_cities(WORLD_CITIES)
            logger.info(f"✅ Initialized with {len(WORLD_CITIES)} cities and {len(REGIONS)} regions")
            
            # Start loading immediately for Vercel (no background threads)
            if not city_loader.is_loading:
                # Load a small batch synchronously
                city_loader.start_loading(batch_size=min(config.PRELOAD_TOP_CITIES, 10))
                logger.info(f"Started loading {config.PRELOAD_TOP_CITIES} cities")
        except Exception as e:
            logger.error(f"❌ Failed to initialize city loader: {e}")
    else:
        logger.error("❌ WORLD_CITIES is empty!")

# ==================== ERROR HANDLERS ====================
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
        "error": "Internal server error",
        "request_id": request.headers.get('X-Request-ID', 'unknown')
    }), 500

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded. Please try again later."
    }), 429

# ==================== VERCEL COMPATIBILITY ====================
# For Vercel serverless, we need to export the Flask app
# Remove the AWS Lambda handler as it's not compatible with Vercel Python

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    """Local development server"""
    
    logger.info("🚀 Starting City Explorer API in LOCAL mode")
    
    if WORLD_CITIES and len(WORLD_CITIES) > 0:
        initialize_city_data()
    else:
        logger.error("❌ WORLD_CITIES is empty! Add your 1500+ cities data")
    
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"📊 Total cities: {len(WORLD_CITIES)}")
    logger.info(f"🌍 Regions: {len(REGIONS)}")
    logger.info(f"⚙️ Configuration: CACHE_TTL={config.CACHE_TTL}s, IMAGE_WORKERS={config.MAX_IMAGE_WORKERS}")
    
    # Run Flask development server
    app.run(
        host='0.0.0.0',
        port=port,
        debug=config.FLASK_DEBUG,
        threaded=True
    )
else:
    """Vercel serverless deployment"""
    logger.info("🔧 Running in VERCEL serverless mode")
    
    # Initialize SYNCHRONOUSLY for Vercel (no background threads)
    if WORLD_CITIES and len(WORLD_CITIES) > 0:
        logger.info(f"📊 Found {len(WORLD_CITIES)} cities")
        
        try:
            # Extract regions
            REGIONS.clear()
            for city in WORLD_CITIES:
                if 'region' in city:
                    REGIONS.add(city['region'])
            
            # Ensure basic regions
            basic_regions = {"Europe", "North America", "Asia", "Oceania", "Middle East", "South America", "Africa"}
            REGIONS.update(basic_regions)
            
            # Initialize city loader immediately
            city_loader.initialize_with_world_cities(WORLD_CITIES)
            logger.info(f"✅ City loader initialized with {len(WORLD_CITIES)} cities")
            
            # Start loading a small batch synchronously
            city_loader.start_loading(batch_size=min(10, config.PRELOAD_TOP_CITIES))
            logger.info(f"Started loading initial batch of cities")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize city data: {e}")
    else:
        logger.error("❌ WORLD_CITIES is empty in Vercel environment")