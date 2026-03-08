import time
import requests
from typing import Dict, Optional
from urllib.parse import quote_plus
from geopy.geocoders import Nominatim
from api.config import config
from api.utils.logger import logger
from api.utils.cache import cache
from api.utils.wikipedia_provider import wikipedia_provider

class CoordinatesProvider:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="CityExplorer/2.0 (https://traveltto.com)",
            timeout=config.GEOLOCATOR_TIMEOUT
        )
    
    def get_coordinates(self, city_name: str, country: str = None, refresh: bool = False) -> Optional[Dict]:
        cache_key = f"coords:{city_name}:{country}"
        
        if not refresh:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        # 1. Try Mapbox first (Faster, higher limits)
        if config.MAPBOX_ACCESS_TOKEN:
            try:
                query = f"{city_name}, {country}" if country else city_name
                url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{quote_plus(query)}.json"
                params = {
                    "access_token": config.MAPBOX_ACCESS_TOKEN,
                    "types": "place,locality",
                    "limit": 1,
                    "language": "en"
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    features = data.get("features", [])
                    if features:
                        # Mapbox returns [lon, lat]
                        center = features[0].get("center")
                        if center and len(center) == 2:
                            coords = {
                                "lat": center[1],
                                "lon": center[0]
                            }
                            # Longer TTL for coordinates as they rarely change
                            cache.set(cache_key, coords, config.CACHE_TTL_COORDS)
                            return coords
            except Exception as e:
                logger.warning(f"Mapbox geocoding failed for {city_name}: {e}")
        
        # 2. Try Wikipedia (Reliable, often has coords)
        # We try this before Nominatim because it's less rate-limited and often more accurate for "City" entities
        try:
            wiki_coords = wikipedia_provider.get_city_coordinates(city_name, refresh=refresh)
            if wiki_coords:
                logger.info(f"Found coordinates for {city_name} via Wikipedia: {wiki_coords}")
                cache.set(cache_key, wiki_coords, config.CACHE_TTL_COORDS)
                return wiki_coords
        except Exception as e:
            logger.debug(f"Wikipedia coords fallback failed: {e}")

        # 3. Fallback to Nominatim (Slower, rate limited)
        queries = []
        
        if country:
            queries.append(f"{city_name}, {country}")
        
        queries.append(f"{city_name} city")
        queries.append(f"{city_name}")
        
        if country:
             queries.append(f"{city_name}, {country}, city")
        
        for query in queries:
            try:
                # Add a small random delay to mitigate rate limiting in parallel execution
                # (Simple jitter)
                if not config.MAPBOX_ACCESS_TOKEN:
                    time.sleep(0.1 + (hash(query) % 10) / 100.0)
                
                location = self.geolocator.geocode(
                    query,
                    exactly_one=True,
                    addressdetails=True,
                    language="en",
                    timeout=config.GEOLOCATOR_TIMEOUT + 5  # Increased timeout
                )
                
                if location and hasattr(location, 'latitude'):
                    coords = {
                        "lat": location.latitude,
                        "lon": location.longitude
                    }
                    cache.set(cache_key, coords, config.CACHE_TTL_COORDS)
                    return coords
                    
            except Exception as e:
                logger.debug(f"Geolocation failed for '{query}': {e}")
                continue
        
        return None

coordinates_provider = CoordinatesProvider()
