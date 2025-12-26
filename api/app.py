import os
import re
import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import quote_plus, unquote
import requests
import wikipediaapi
import diskcache
from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request
from flask_cors import CORS
from dataclasses import dataclass

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    CACHE_TTL: int = 7200  # 2 hours
    CACHE_TTL_IMAGES: int = 86400  # 24 hours
    REQUEST_TIMEOUT: int = 10
    GEOLOCATOR_TIMEOUT: int = 5
    MAX_WORKERS: int = 3  # Reduced for Vercel limits

config = Config()

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CityExplorerAPI")

# ==================== CACHE ====================
class CacheManager:
    def __init__(self):
        try:
            os.makedirs("/tmp/city_cache", exist_ok=True)
            self.cache = diskcache.Cache("/tmp/city_cache")
        except:
            self.cache = {}
    
    def get(self, key: str):
        try:
            if isinstance(self.cache, dict):
                item = self.cache.get(key)
                if item and time.time() - item.get('ts', 0) < config.CACHE_TTL:
                    return item.get('data')
            else:
                item = self.cache.get(key)
                if item and time.time() - item.get('ts', 0) < config.CACHE_TTL:
                    return item.get('data')
        except:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        try:
            item = {'data': value, 'ts': time.time()}
            if isinstance(self.cache, dict):
                self.cache[key] = item
            else:
                self.cache.set(key, item, expire=ttl or config.CACHE_TTL)
        except:
            pass

cache = CacheManager()

# ==================== REQUEST HANDLER ====================
class RequestHandler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CityExplorer/1.0',
            'Accept': 'application/json'
        })
    
    def get_json(self, url: str, params: dict = None):
        try:
            resp = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Request failed: {url} - {e}")
            return None

req_handler = RequestHandler()

# ==================== IMAGE FETCHER ====================
class ImageFetcher:
    def get_images(self, city_name: str, limit: int = 3):
        cache_key = f"img:{city_name}:{limit}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Try Wikimedia Commons
            params = {
                'action': 'query',
                'generator': 'search',
                'gsrsearch': f'{city_name} city',
                'gsrnamespace': '6',
                'gsrlimit': 10,
                'prop': 'imageinfo',
                'iiprop': 'url|size',
                'iiurlwidth': 800,
                'format': 'json'
            }
            
            data = req_handler.get_json("https://commons.wikimedia.org/w/api.php", params)
            images = []
            
            if data and 'query' in data:
                for page in data['query'].get('pages', {}).values():
                    if 'imageinfo' in page:
                        info = page['imageinfo'][0]
                        if info.get('width', 0) >= 400 and info.get('height', 0) >= 300:
                            images.append({
                                'url': info.get('thumburl') or info.get('url'),
                                'title': page.get('title', '').replace('File:', ''),
                                'description': f'Image of {city_name}',
                                'source': 'wikimedia',
                                'width': info.get('width'),
                                'height': info.get('height'),
                                'quality_score': 70
                            })
                    
                    if len(images) >= limit:
                        break
            
            if not images:
                # Fallback to placeholder
                encoded = quote_plus(city_name)
                images.append({
                    'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&fit=crop&w=800&q=80&txt={encoded}&txt-size=40',
                    'title': city_name,
                    'description': f'Image of {city_name}',
                    'source': 'placeholder',
                    'width': 800,
                    'height': 600,
                    'quality_score': 30
                })
            
            cache.set(cache_key, images, config.CACHE_TTL_IMAGES)
            return images
            
        except Exception as e:
            logger.warning(f"Image fetch failed for {city_name}: {e}")
            encoded = quote_plus(city_name)
            return [{
                'url': f'https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&fit=crop&w=800&q=80&txt={encoded}&txt-size=40',
                'title': city_name,
                'description': f'Image of {city_name}',
                'source': 'placeholder',
                'width': 800,
                'height': 600,
                'quality_score': 30
            }]

image_fetcher = ImageFetcher()

# ==================== CITY PROVIDER ====================
class CityProvider:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="city_explorer",
            timeout=config.GEOLOCATOR_TIMEOUT
        )
        self.wiki = wikipediaapi.Wikipedia('en')
    
    def get_coordinates(self, city_name: str, country: str = None):
        cache_key = f"coord:{city_name}:{country}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            query = f"{city_name}, {country}" if country else city_name
            location = self.geolocator.geocode(query, exactly_one=True, timeout=5)
            if location:
                coords = {"lat": location.latitude, "lon": location.longitude}
                cache.set(cache_key, coords, 86400)
                return coords
        except Exception as e:
            logger.warning(f"Geocode failed for {city_name}: {e}")
        
        return None
    
    def get_wiki_data(self, city_name: str, country: str = None):
        cache_key = f"wiki:{city_name}:{country}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            page_title = f"{city_name}, {country}" if country else city_name
            page = self.wiki.page(page_title)
            
            if page.exists():
                data = {
                    'title': page.title,
                    'summary': (page.summary or '')[:300] + '...',
                    'exists': True,
                    'url': page.fullurl
                }
            else:
                # Try without country
                page = self.wiki.page(city_name)
                if page.exists():
                    data = {
                        'title': page.title,
                        'summary': (page.summary or '')[:300] + '...',
                        'exists': True,
                        'url': page.fullurl
                    }
                else:
                    data = {
                        'title': city_name,
                        'summary': f'{city_name} is a city waiting to be explored.',
                        'exists': False,
                        'url': f'https://en.wikipedia.org/wiki/{quote_plus(city_name)}'
                    }
            
            cache.set(cache_key, data)
            return data
            
        except Exception as e:
            logger.warning(f"Wiki failed for {city_name}: {e}")
            return {
                'title': city_name,
                'summary': f'{city_name} is a city waiting to be explored.',
                'exists': False,
                'url': f'https://en.wikipedia.org/wiki/{quote_plus(city_name)}'
            }

city_provider = CityProvider()

# ==================== TAGLINES ====================
TAGLINES = {
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

def get_tagline(city_name: str):
    if city_name in TAGLINES:
        return TAGLINES[city_name], "known"
    return f"Discover {city_name}", "generated"

# ==================== WORLD CITIES DATA ====================
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

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)

# Stats tracking
stats = {
    'total': len(WORLD_CITIES),
    'loaded': 0,
    'with_images': 0,
    'with_coords': 0
}

def process_city(city_info: Dict) -> Dict:
    """Process a single city to get all data"""
    city_name = city_info['name']
    country = city_info.get('country')
    
    cache_key = f"fullcity:{city_name}:{country}"
    cached = cache.get(cache_key)
    if cached:
        stats['loaded'] += 1
        if cached.get('image'):
            stats['with_images'] += 1
        if cached.get('coordinates'):
            stats['with_coords'] += 1
        return cached
    
    logger.info(f"Processing: {city_name}")
    
    # Get data in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        coords_future = executor.submit(city_provider.get_coordinates, city_name, country)
        wiki_future = executor.submit(city_provider.get_wiki_data, city_name, country)
        images_future = executor.submit(image_fetcher.get_images, city_name, 3)
        
        coordinates = coords_future.result(timeout=8)
        wiki_data = wiki_future.result(timeout=8)
        images = images_future.result(timeout=8)
    
    # Generate static map URL
    static_map = None
    if coordinates:
        lat, lon = coordinates["lat"], coordinates["lon"]
        static_map = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom=12&size=400x250&markers={lat},{lon},red-pushpin&scale=2"
    
    tagline, tagline_source = get_tagline(city_name)
    
    city_data = {
        "_wiki_title": wiki_data.get('title'),
        "coordinates": coordinates,
        "country": city_info.get('country', 'Unknown'),
        "display_name": wiki_data.get('title', city_name),
        "has_details": wiki_data.get('exists', False),
        "id": re.sub(r'[^a-z0-9-]', '-', city_name.lower()),
        "image": images[0] if images else None,
        "images": images,
        "last_updated": time.time(),
        "metadata": {
            "coordinate_accuracy": "nominatim" if coordinates else "unknown",
            "data_completeness": 100 if coordinates and images else 50,
            "image_quality": "good" if images else "basic"
        },
        "name": city_name,
        "region": city_info.get('region', 'Unknown'),
        "static_map": static_map or "https://via.placeholder.com/400x250.png?text=Map+Not+Available",
        "summary": wiki_data.get('summary', f"{city_name} is a city waiting to be explored.")[:150] + '...',
        "tagline": tagline,
        "tagline_source": tagline_source
    }
    
    # Update stats
    stats['loaded'] += 1
    if images:
        stats['with_images'] += 1
    if coordinates:
        stats['with_coords'] += 1
    
    cache.set(cache_key, city_data)
    return city_data

# ==================== API ENDPOINTS ====================
@app.route('/')
def home():
    return jsonify({
        "name": "City Explorer API",
        "version": "2.0",
        "endpoints": ["/api/cities", "/api/search", "/api/regions", "/api/stats", "/api/health"]
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cities": {
            "total": stats['total'],
            "loaded": stats['loaded'],
            "cached_percentage": f"{(stats['loaded'] / max(stats['total'], 1) * 100):.1f}%"
        }
    })

@app.route('/api/cities')
def get_cities():
    """Main endpoint for Blog.jsx"""
    try:
        page = max(1, int(request.args.get('page', 1)))
        limit = min(50, max(1, int(request.args.get('limit', 20))))
        region = request.args.get('region')
        
        logger.info(f"GET /api/cities - page={page}, limit={limit}, region={region}")
        
        # Filter by region if specified
        cities_to_process = WORLD_CITIES
        if region and region != "All":
            cities_to_process = [c for c in WORLD_CITIES if c.get('region') == region]
        
        # Pagination
        total = len(cities_to_process)
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, total)
        total_pages = max(1, (total + limit - 1) // limit)
        
        # Get cities for this page
        page_cities = cities_to_process[start_idx:end_idx]
        
        # Process cities in parallel (with limit for Vercel)
        max_workers = min(config.MAX_WORKERS, len(page_cities))
        city_data_list = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_city, city_info) for city_info in page_cities]
            
            for future in futures:
                try:
                    city_data = future.result(timeout=10)
                    city_data_list.append(city_data)
                except Exception as e:
                    logger.error(f"Failed to process city: {e}")
                    # Add placeholder
                    city_info = page_cities[futures.index(future)]
                    city_data_list.append({
                        "id": re.sub(r'[^a-z0-9-]', '-', city_info['name'].lower()),
                        "name": city_info['name'],
                        "display_name": city_info['name'],
                        "summary": f"Loading {city_info['name']}...",
                        "has_details": False,
                        "image": {
                            "url": "https://images.unsplash.com/photo-1519681393784-d120267933ba?ixlib=rb-4.0.3&fit=crop&w=800&q=80",
                            "title": city_info['name'],
                            "description": f"Image of {city_info['name']}",
                            "source": "placeholder",
                            "width": 800,
                            "height": 600,
                            "quality_score": 10
                        },
                        "images": [],
                        "coordinates": None,
                        "static_map": "https://via.placeholder.com/400x250.png?text=Loading",
                        "tagline": f"Discover {city_info['name']}",
                        "tagline_source": "loading",
                        "country": city_info.get('country', 'Unknown'),
                        "region": city_info.get('region', 'Unknown'),
                        "metadata": {
                            "image_quality": "loading",
                            "coordinate_accuracy": "loading",
                            "data_completeness": 10
                        }
                    })
        
        # Build response matching Blog.jsx expectations
        response = {
            "success": True,
            "data": city_data_list,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": total_pages,
                "next_page": page + 1 if page < total_pages else None,
                "prev_page": page - 1 if page > 1 else None
            },
            "loading": {
                "complete": stats['loaded'] >= stats['total'],
                "loaded": stats['loaded'],
                "total": stats['total'],
                "progress": f"{(stats['loaded'] / max(stats['total'], 1) * 100):.1f}%",
                "message": f"{stats['loaded']} cities loaded. Others load on access."
            },
            "info": {
                "total_cities": stats['total'],
                "fully_loaded_cities": stats['loaded'],
                "cities_with_images": stats['with_images'],
                "cities_with_coordinates": stats['with_coords'],
                "loading_strategy": "on-demand"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in /api/cities: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "data": []
        }), 500

@app.route('/api/search')
def search():
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify({"success": True, "data": []})
    
    limit = min(20, max(1, int(request.args.get('limit', 10))))
    
    # Simple search in city names
    results = []
    for city_info in WORLD_CITIES[:50]:  # Limit search scope
        if query.lower() in city_info['name'].lower():
            city_data = process_city(city_info)
            results.append(city_data)
        
        if len(results) >= limit:
            break
    
    return jsonify({
        "success": True,
        "query": query,
        "data": results
    })

@app.route('/api/regions')
def get_regions():
    return jsonify({
        "success": True,
        "data": REGIONS,
        "count": len(REGIONS)
    })

@app.route('/api/stats')
def get_stats():
    return jsonify({
        "success": True,
        "stats": {
            "total_cities": stats['total'],
            "loaded_cities": stats['loaded'],
            "percentage_loaded": f"{(stats['loaded'] / max(stats['total'], 1) * 100):.1f}%",
            "cities_with_images": stats['with_images'],
            "cities_with_coordinates": stats['with_coords']
        }
    })

@app.route('/api/cities/<path:city_name>')
def get_city_details(city_name):
    city_name = unquote(city_name)
    
    # Find the city
    city_info = None
    for city in WORLD_CITIES:
        if city['name'].lower() == city_name.lower():
            city_info = city
            break
    
    if not city_info:
        return jsonify({"success": False, "error": "City not found"}), 404
    
    try:
        city_data = process_city(city_info)
        return jsonify({"success": True, "data": city_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==================== VERCEL HANDLER ====================
# This is crucial for Vercel serverless
# DO NOT use if __name__ == '__main__' in Vercel

# The Flask app instance
if __name__ == '__main__':
    # Only for local testing
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)