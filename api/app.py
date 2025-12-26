import os
import re
import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import quote_plus, unquote
import requests
import wikipediaapi
import diskcache
from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request
from flask_cors import CORS
from dataclasses import dataclass, field

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """Configuration settings"""
    
    # API Keys & Secrets
    UNSPLASH_ACCESS_KEY: str = os.getenv("UNSPLASH_ACCESS_KEY", "")
    MAPBOX_ACCESS_TOKEN: str = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    
    # Caching
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "7200"))  # 2 hours
    CACHE_TTL_IMAGES: int = int(os.getenv("CACHE_TTL_IMAGES", "86400"))  # 24 hours
    
    # Workers
    MAX_IMAGE_WORKERS: int = int(os.getenv("MAX_IMAGE_WORKERS", "4"))
    MAX_DETAIL_WORKERS: int = int(os.getenv("MAX_DETAIL_WORKERS", "3"))
    
    # Timeouts
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))
    WIKIPEDIA_TIMEOUT: int = int(os.getenv("WIKIPEDIA_TIMEOUT", "15"))
    GEOLOCATOR_TIMEOUT: int = int(os.getenv("GEOLOCATOR_TIMEOUT", "10"))
    
    # Flask
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    FLASK_PORT: int = int(os.getenv("PORT", "5000"))
    
    # Image settings
    MAX_IMAGES_PER_REQUEST: int = 6
    MIN_IMAGE_WIDTH: int = 400
    MIN_IMAGE_HEIGHT: int = 300
    MIN_IMAGE_QUALITY_SCORE: int = 30
    
    # Loading
    PRELOAD_TOP_CITIES: int = int(os.getenv("PRELOAD_TOP_CITIES", "10"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# ==================== LOGGING ====================
logger = logging.getLogger("CityExplorerAPI")
logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

logger.info("🚀 City Explorer API Starting...")

# ==================== CACHE SYSTEM ====================
class CacheManager:
    """Simple cache manager"""
    
    def __init__(self):
        self.cache_dir = "/tmp/city_explorer_cache"
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(self.cache_dir)
            logger.info(f"✅ Cache initialized at: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Disk cache failed: {e}")
            self.cache = diskcache.Cache()  # Memory-only
    
    def get(self, key: str):
        """Get item from cache"""
        try:
            item = self.cache.get(key)
            if item and time.time() - item.get('timestamp', 0) < config.CACHE_TTL:
                return item.get('value')
        except Exception:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set item in cache"""
        try:
            self.cache.set(key, {
                'value': value,
                'timestamp': time.time()
            }, expire=ttl or config.CACHE_TTL)
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def delete(self, key: str):
        """Delete item from cache"""
        try:
            self.cache.delete(key)
        except Exception:
            pass

cache_manager = CacheManager()

# ==================== REQUEST HANDLER ====================
class RequestHandler:
    """Handles HTTP requests with retry logic"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CityExplorer/2.0)',
            'Accept': 'application/json'
        })
    
    def get_json(self, url: str, params: dict = None, headers: dict = None):
        """Get JSON from URL with retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Request failed for {url}: {e}")
                    raise
                time.sleep(1 * (attempt + 1))
    
    def get_json_cached(self, url: str, params: dict = None, cache_key: str = None, ttl: int = None):
        """Get JSON with caching"""
        if not cache_key:
            cache_key = f"req:{hashlib.md5(f'{url}{json.dumps(params or {})}'.encode()).hexdigest()}"
        
        cached = cache_manager.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            data = self.get_json(url, params)
            cache_manager.set(cache_key, data, ttl or config.CACHE_TTL)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

request_handler = RequestHandler()

# ==================== IMAGE FETCHER ====================
class ImageFetcher:
    """Fetches images for cities"""
    
    def __init__(self):
        self.wikimedia_api = "https://commons.wikimedia.org/w/api.php"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
    
    def fetch_from_wikimedia(self, query: str, limit: int = 6):
        """Fetch images from Wikimedia Commons"""
        try:
            params = {
                'action': 'query',
                'generator': 'search',
                'gsrsearch': f'{query} city skyline',
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
                cache_key=f"wikimedia:{query}",
                ttl=config.CACHE_TTL_IMAGES
            )
            
            images = []
            if data and 'query' in data and 'pages' in data['query']:
                for page in data['query']['pages'].values():
                    if 'imageinfo' in page:
                        info = page['imageinfo'][0]
                        mime = info.get('mime', '')
                        
                        if mime.startswith('image/'):
                            image_data = {
                                'url': info.get('thumburl') or info.get('url'),
                                'title': page.get('title', '').replace('File:', ''),
                                'description': self._extract_description(info),
                                'source': 'wikimedia',
                                'width': info.get('width'),
                                'height': info.get('height'),
                                'quality_score': self._calculate_quality_score(info)
                            }
                            
                            if (image_data['width'] or 0) >= config.MIN_IMAGE_WIDTH and \
                               (image_data['height'] or 0) >= config.MIN_IMAGE_HEIGHT:
                                images.append(image_data)
                    
                    if len(images) >= limit:
                        break
            
            return sorted(images, key=lambda x: x.get('quality_score', 0), reverse=True)[:limit]
            
        except Exception as e:
            logger.debug(f"Wikimedia fetch failed for {query}: {e}")
            return []
    
    def fetch_from_wikipedia(self, page_title: str, limit: int = 4):
        """Fetch images from Wikipedia article"""
        try:
            # Get page images
            params = {
                'action': 'query',
                'titles': page_title,
                'prop': 'images|pageimages',
                'pithumbsize': 800,
                'imlimit': 20,
                'format': 'json'
            }
            
            data = request_handler.get_json_cached(
                self.wikipedia_api,
                params=params,
                cache_key=f"wikipedia_images:{page_title}",
                ttl=config.CACHE_TTL_IMAGES
            )
            
            images = []
            pages = data.get('query', {}).get('pages', {})
            
            for page in pages.values():
                # Add thumbnail if available
                if 'thumbnail' in page:
                    thumb = page['thumbnail']
                    images.append({
                        'url': thumb['source'],
                        'title': f'Main image of {page_title}',
                        'description': 'Featured image from Wikipedia',
                        'source': 'wikipedia',
                        'width': thumb.get('width'),
                        'height': thumb.get('height'),
                        'quality_score': 70
                    })
            
            return images[:limit]
            
        except Exception as e:
            logger.debug(f"Wikipedia image fetch failed: {e}")
            return []
    
    def get_fallback_image(self, city_name: str):
        """Get fallback image for a city"""
        encoded_city = quote_plus(city_name)
        return {
            'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80&txt={encoded_city}&txt-size=40&txt-color=white&txt-align=middle,center',
            'title': city_name,
            'description': f'Image of {city_name}',
            'source': 'placeholder',
            'width': 800,
            'height': 600,
            'quality_score': 20
        }
    
    def get_images_for_city(self, city_name: str, page_title: str = None, limit: int = None):
        """Get images for a city using multiple sources"""
        limit = limit or 3
        images = []
        
        # Try Wikimedia first
        images.extend(self.fetch_from_wikimedia(city_name, limit))
        
        # Try Wikipedia if we have page title
        if page_title and len(images) < limit:
            images.extend(self.fetch_from_wikipedia(page_title, limit - len(images)))
        
        # Remove duplicates
        unique_images = []
        seen_urls = set()
        for img in images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        # Ensure we have at least one image
        if not unique_images:
            unique_images.append(self.get_fallback_image(city_name))
        
        return unique_images[:limit]
    
    def _extract_description(self, image_info: dict):
        """Extract description from image metadata"""
        extmetadata = image_info.get('extmetadata', {})
        
        for field in ['ImageDescription', 'ObjectName', 'Caption']:
            if field in extmetadata:
                value = extmetadata[field].get('value', '')
                if value and isinstance(value, str):
                    clean = re.sub(r'<[^>]+>', '', value)
                    return clean[:200]
        
        return ""
    
    def _calculate_quality_score(self, image_info: dict):
        """Calculate quality score for an image"""
        score = 50
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        
        if width >= 1200 and height >= 800:
            score += 30
        elif width >= 800 and height >= 600:
            score += 20
        
        return min(100, max(0, score))

image_fetcher = ImageFetcher()

# ==================== CITY DATA PROVIDER ====================
class CityDataProvider:
    """Provides city data with coordinates and Wikipedia info"""
    
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="CityExplorer/2.0",
            timeout=config.GEOLOCATOR_TIMEOUT
        )
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='CityExplorer/2.0'
        )
    
    def get_coordinates(self, city_name: str, country: str = None):
        """Get coordinates for a city"""
        cache_key = f"coords:{city_name}:{country}"
        
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
        
        queries = []
        if country:
            queries.append(f"{city_name}, {country}")
            queries.append(f"{city_name} city, {country}")
        
        queries.append(city_name)
        queries.append(f"{city_name} city")
        
        for query in queries:
            try:
                location = self.geolocator.geocode(
                    query,
                    exactly_one=True,
                    timeout=config.GEOLOCATOR_TIMEOUT
                )
                
                if location:
                    result = {
                        "lat": location.latitude,
                        "lon": location.longitude
                    }
                    cache_manager.set(cache_key, result, 86400)  # 24 hours
                    return result
                    
            except Exception as e:
                logger.debug(f"Geocode failed for '{query}': {e}")
                continue
        
        return None
    
    def get_wikipedia_data(self, city_name: str, country: str = None):
        """Get Wikipedia data for a city"""
        cache_key = f"wiki:{city_name}:{country}"
        
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
        
        variations = [city_name]
        if country:
            variations.extend([
                f"{city_name}, {country}",
                f"{city_name} ({country})"
            ])
        
        for variation in variations:
            try:
                page = self.wiki.page(variation)
                
                if page.exists() and page.ns == 0:
                    # Check if it's a city page
                    if self._is_city_page(page):
                        result = {
                            'title': page.title,
                            'summary': (page.summary or "")[:500],
                            'fullurl': page.fullurl,
                            'exists': True
                        }
                        cache_manager.set(cache_key, result)
                        return result
                        
            except Exception as e:
                logger.debug(f"Wikipedia check failed: {e}")
                continue
        
        # No Wikipedia page found
        result = {
            'title': city_name,
            'summary': f"{city_name} is a city worth exploring.",
            'fullurl': f"https://en.wikipedia.org/wiki/{quote_plus(city_name)}",
            'exists': False
        }
        cache_manager.set(cache_key, result)
        return result
    
    def _is_city_page(self, page):
        """Check if Wikipedia page is about a city"""
        try:
            text = (page.summary or "").lower()
            city_indicators = ['city', 'town', 'municipality', 'capital', 'population']
            return any(indicator in text for indicator in city_indicators)
        except Exception:
            return True
    
    def get_city_tagline(self, city_name: str, country: str = None):
        """Get tagline for a city"""
        cache_key = f"tagline:{city_name}:{country}"
        
        cached = cache_manager.get(cache_key)
        if cached:
            return cached
        
        # Hardcoded taglines for famous cities
        taglines = {
            "Paris": "The City of Light",
            "London": "The Old Smoke",
            "Rome": "The Eternal City",
            "Barcelona": "The City of Gaudí",
            "Amsterdam": "The Venice of the North",
            "New York": "The Big Apple",
            "Tokyo": "The Eastern Capital",
            "Dubai": "The City of Gold",
            "Venice": "The Floating City",
            "Prague": "The City of a Hundred Spires",
            "Vienna": "The City of Music",
            "Rio de Janeiro": "The Marvelous City",
            "San Francisco": "The Golden City",
            "Las Vegas": "The Entertainment Capital of the World",
            "Singapore": "The Lion City"
        }
        
        if city_name in taglines:
            result = {"tagline": taglines[city_name], "source": "known"}
        else:
            # Generate from country
            country_taglines = {
                "Italy": "A beautiful Italian city",
                "France": "A charming French city",
                "Japan": "A vibrant Japanese city",
                "USA": "An American city full of opportunities",
                "Spain": "A sunny Spanish destination",
                "Germany": "A historic German city",
                "UK": "A classic British destination",
                "Greece": "A city with ancient Greek heritage",
                "Thailand": "An exotic Thai destination"
            }
            
            if country and country in country_taglines:
                result = {"tagline": country_taglines[country], "source": "country"}
            else:
                result = {"tagline": f"Discover {city_name}", "source": "generated"}
        
        cache_manager.set(cache_key, result)
        return result

city_provider = CityDataProvider()

# ==================== CITY LOADER ====================
class CityLoader:
    """Manages loading of cities"""
    
    def __init__(self):
        self.loaded_cities = {}
        self.stats = {
            'loaded': 0,
            'with_images': 0,
            'with_coordinates': 0,
            'total': 0
        }
    
    def load_city(self, city_info: Dict, force_refresh: bool = False):
        """Load a single city with all data"""
        city_name = city_info['name']
        
        # Check cache first
        cache_key = f"city:{city_name}:{city_info.get('country')}"
        if not force_refresh:
            cached = cache_manager.get(cache_key)
            if cached:
                self.loaded_cities[city_name] = cached
                return cached
        
        logger.info(f"🔄 Loading city: {city_name}")
        
        try:
            # Get Wikipedia data
            wiki_data = city_provider.get_wikipedia_data(city_name, city_info.get('country'))
            
            # Get coordinates
            coordinates = city_provider.get_coordinates(city_name, city_info.get('country'))
            
            # Get images
            images = image_fetcher.get_images_for_city(
                city_name, 
                wiki_data['title'] if wiki_data.get('exists') else None,
                limit=3
            )
            
            # Get tagline
            tagline_data = city_provider.get_city_tagline(city_name, city_info.get('country'))
            
            # Build city data
            city_data = {
                "_wiki_title": wiki_data.get('title'),
                "coordinates": coordinates,
                "country": city_info.get('country', 'Unknown'),
                "display_name": wiki_data.get('title', city_name),
                "has_details": wiki_data.get('exists', False),
                "id": city_name.lower().replace(' ', '-').replace(',', ''),
                "image": images[0] if images else image_fetcher.get_fallback_image(city_name),
                "images": images,
                "last_updated": time.time(),
                "metadata": {
                    "coordinate_accuracy": "nominatim" if coordinates else "unknown",
                    "data_completeness": 100 if coordinates and images and wiki_data.get('exists') else 50,
                    "image_quality": "good" if images and images[0].get('quality_score', 0) > 50 else "basic"
                },
                "name": city_name,
                "region": city_info.get('region', 'Unknown'),
                "static_map": self._generate_static_map(coordinates),
                "summary": wiki_data.get('summary', f"{city_name} is a city waiting to be explored.")[:150] + '...',
                "tagline": tagline_data['tagline'],
                "tagline_source": tagline_data['source']
            }
            
            # Update stats
            self.stats['loaded'] += 1
            if images and images[0].get('quality_score', 0) > 30:
                self.stats['with_images'] += 1
            if coordinates:
                self.stats['with_coordinates'] += 1
            
            # Cache and store
            cache_manager.set(cache_key, city_data)
            self.loaded_cities[city_name] = city_data
            
            logger.info(f"✅ Loaded: {city_name}")
            return city_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load {city_name}: {e}")
            # Return basic data
            return self._get_basic_city_data(city_info)
    
    def _generate_static_map(self, coordinates: Dict):
        """Generate static map URL"""
        if coordinates:
            lat, lon = coordinates["lat"], coordinates["lon"]
            return f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom=12&size=400x250&markers={lat},{lon},red-pushpin&scale=2"
        return "https://via.placeholder.com/400x250.png?text=Map+Not+Available"
    
    def _get_basic_city_data(self, city_info: Dict):
        """Get basic city data when loading fails"""
        city_name = city_info['name']
        return {
            "id": city_name.lower().replace(' ', '-').replace(',', ''),
            "name": city_name,
            "display_name": city_name,
            "summary": f"Loading information for {city_name}...",
            "has_details": False,
            "image": image_fetcher.get_fallback_image(city_name),
            "images": [],
            "coordinates": None,
            "static_map": "https://via.placeholder.com/400x250.png?text=Loading+Map",
            "tagline": f"Discover {city_name}",
            "tagline_source": "basic",
            "last_updated": time.time(),
            "country": city_info.get('country', 'Unknown'),
            "region": city_info.get('region', 'Unknown'),
            "metadata": {
                "image_quality": "basic",
                "coordinate_accuracy": "unknown",
                "data_completeness": 10
            }
        }
    
    def get_city(self, city_name: str):
        """Get a loaded city by name"""
        return self.loaded_cities.get(city_name)
    
    def search_cities(self, query: str, limit: int = 20):
        """Search loaded cities"""
        if not query or len(query) < 2:
            return []
        
        query_lower = query.lower()
        results = []
        
        for city_data in self.loaded_cities.values():
            if (query_lower in city_data['name'].lower() or 
                query_lower in city_data.get('display_name', '').lower() or
                query_lower in city_data.get('country', '').lower()):
                results.append(city_data)
            
            if len(results) >= limit:
                break
        
        return results

city_loader = CityLoader()

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)

# Initialize with your data
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

def initialize_city_data():
    """Initialize with your world cities data"""
    global WORLD_CITIES, REGIONS
    
    if WORLD_CITIES:
        # Extract unique regions
        REGIONS.clear()
        for city in WORLD_CITIES:
            if 'region' in city:
                REGIONS.add(city['region'])
        
        city_loader.stats['total'] = len(WORLD_CITIES)
        
        # Preload popular cities
        popular_cities = [
            "Paris", "London", "Rome", "Barcelona", "Amsterdam",
            "New York", "Tokyo", "Dubai", "Berlin", "Sydney"
        ]
        
        logger.info(f"📊 Initializing with {len(WORLD_CITIES)} cities")
        
        # Load popular cities
        with ThreadPoolExecutor(max_workers=config.MAX_DETAIL_WORKERS) as executor:
            futures = []
            for city_name in popular_cities:
                # Find city in WORLD_CITIES
                for city_info in WORLD_CITIES:
                    if city_info['name'] == city_name:
                        futures.append(executor.submit(city_loader.load_city, city_info))
                        break
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.warning(f"Preload failed: {e}")

# ==================== API ROUTES ====================
@app.route('/')
def home():
    return jsonify({
        "name": "City Explorer API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "cities": "/api/cities",
            "city_details": "/api/cities/<city_name>",
            "search": "/api/search",
            "regions": "/api/regions",
            "stats": "/api/stats",
            "health": "/api/health"
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "city_loading": {
            "loaded": city_loader.stats['loaded'],
            "total": city_loader.stats['total'],
            "with_images": city_loader.stats['with_images'],
            "with_coordinates": city_loader.stats['with_coordinates']
        }
    })

@app.route('/api/cities')
def get_cities():
    """Get paginated cities - this is the main endpoint your Blog.jsx uses"""
    try:
        # Get query parameters
        page = max(1, int(request.args.get('page', 1)))
        limit = min(100, max(1, int(request.args.get('limit', 20))))
        region = request.args.get('region')
        
        logger.info(f"📄 API Request: page={page}, limit={limit}, region={region}")
        
        if not WORLD_CITIES:
            return jsonify({
                "success": False,
                "error": "No cities data loaded",
                "data": []
            }), 200
        
        # Filter by region if specified
        cities_to_process = WORLD_CITIES
        if region and region != "All":
            cities_to_process = [c for c in WORLD_CITIES if c.get('region') == region]
        
        # Calculate pagination
        total_cities = len(cities_to_process)
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total_cities)
        total_pages = max(1, (total_cities + limit - 1) // limit)
        
        # Get cities for this page
        page_cities = cities_to_process[start_idx:end_idx]
        
        # Load cities in parallel
        loaded_cities_data = []
        with ThreadPoolExecutor(max_workers=config.MAX_DETAIL_WORKERS) as executor:
            futures = {executor.submit(city_loader.load_city, city_info): city_info for city_info in page_cities}
            
            for future in as_completed(futures):
                try:
                    city_data = future.result(timeout=15)
                    loaded_cities_data.append(city_data)
                except Exception as e:
                    city_info = futures[future]
                    logger.warning(f"Failed to load {city_info['name']}: {e}")
                    # Add basic data
                    loaded_cities_data.append(city_loader._get_basic_city_data(city_info))
        
        # Sort by name for consistent ordering
        loaded_cities_data.sort(key=lambda x: x['name'])
        
        # Build response matching your Blog.jsx expectations
        response = {
            "success": True,
            "data": loaded_cities_data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_cities,
                "pages": total_pages,
                "next_page": page + 1 if page < total_pages else None,
                "prev_page": page - 1 if page > 1 else None
            },
            "loading": {
                "complete": city_loader.stats['loaded'] >= city_loader.stats['total'],
                "loaded": city_loader.stats['loaded'],
                "total": city_loader.stats['total'],
                "progress": f"{(city_loader.stats['loaded'] / max(city_loader.stats['total'], 1) * 100):.1f}%",
                "message": f"{city_loader.stats['loaded']} cities fully loaded. Others will load when accessed."
            },
            "info": {
                "total_cities": city_loader.stats['total'],
                "fully_loaded_cities": city_loader.stats['loaded'],
                "cities_with_images": city_loader.stats['with_images'],
                "cities_with_coordinates": city_loader.stats['with_coordinates'],
                "loading_strategy": "on-demand"
            }
        }
        
        logger.info(f"✅ API Response: {len(loaded_cities_data)} cities on page {page}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "data": []
        }), 500

@app.route('/api/cities/<path:city_name>')
def get_city_details(city_name):
    """Get detailed information for a specific city"""
    city_name = unquote(city_name)
    
    # Find the city
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
        # Load city with more images for details page
        city_data = city_loader.load_city(city_info)
        
        # Get more images for details page
        wiki_data = city_provider.get_wikipedia_data(city_info['name'], city_info.get('country'))
        more_images = image_fetcher.get_images_for_city(
            city_info['name'],
            wiki_data.get('title'),
            limit=6
        )
        
        # Enhance with additional data for details page
        city_data['detailed_summary'] = wiki_data.get('summary', '')
        city_data['wikipedia_url'] = wiki_data.get('fullurl')
        city_data['all_images'] = more_images
        city_data['additional_info'] = {
            "coordinates_source": "OpenStreetMap",
            "images_source": "Wikimedia Commons",
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return jsonify({
            "success": True,
            "data": city_data
        })
        
    except Exception as e:
        logger.error(f"Failed to get details for {city_name}: {e}")
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
            "success": True,
            "query": query,
            "count": 0,
            "data": []
        })
    
    limit = min(50, max(1, int(request.args.get('limit', 20))))
    
    # Search in loaded cities first
    results = city_loader.search_cities(query, limit)
    
    # If not enough results, search in all cities and load them
    if len(results) < limit:
        for city_info in WORLD_CITIES:
            if (query.lower() in city_info['name'].lower() or 
                query.lower() in city_info.get('country', '').lower()):
                
                # Check if already loaded
                if city_info['name'] not in city_loader.loaded_cities:
                    # Load this city
                    try:
                        city_data = city_loader.load_city(city_info)
                        results.append(city_data)
                    except Exception as e:
                        logger.debug(f"Failed to load city during search: {e}")
                
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
    return jsonify({
        "success": True,
        "stats": {
            "total_cities": len(WORLD_CITIES),
            "loaded_cities": city_loader.stats['loaded'],
            "cities_with_images": city_loader.stats['with_images'],
            "cities_with_coordinates": city_loader.stats['with_coordinates'],
            "cache_hits": "N/A",
            "cache_misses": "N/A"
        }
    })

@app.route('/api/reload')
def reload_city():
    """Reload a specific city"""
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
            f"city:{city_name}:{city_info.get('country')}",
            f"coords:{city_name}:{city_info.get('country')}",
            f"wiki:{city_name}:{city_info.get('country')}",
            f"tagline:{city_name}:{city_info.get('country')}"
        ]
        
        for key in cache_keys:
            cache_manager.delete(key)
        
        # Reload city
        city_data = city_loader.load_city(city_info, force_refresh=True)
        
        return jsonify({
            "success": True,
            "message": f"City {city_name} reloaded",
            "data": city_data
        })
        
    except Exception as e:
        logger.error(f"Failed to reload city {city_name}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# Initialize when module loads
if __name__ == '__main__':
    # Run development server
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=config.FLASK_DEBUG)
else:
    # For production (Vercel)
    logger.info("🔧 Running in production mode")
    # initialize_city_data()  # Call this after you populate WORLD_CITIES