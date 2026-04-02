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
    
    def get_coordinates_batch(self, cities: list, refresh: bool = False) -> Dict[str, Dict]:
        """
        Batch fetch coordinates for up to 50 cities using Wikipedia API to minimize external calls.
        Returns a mapping: { city_name: {lat, lon} }
        """
        if not cities:
            return {}
        
        from api.utils.request_handler import request_handler
        
        # Prepare titles with country disambiguation
        title_to_city = {}
        titles = []
        for c in cities:
            name = c.get('name')
            country = c.get('country')
            if not name:
                continue
            if country:
                t1 = f"{name}, {country}"
                t2 = f"{name} ({country})"
                titles.extend([t1, t2])
                title_to_city[t1] = name
                title_to_city[t2] = name
            titles.append(name)
            title_to_city[name] = name
        
        # Wikipedia allows up to ~50 titles per request; chunk if necessary
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]
        
        results = {}
        for batch in chunks(titles, 50):
            params = {
                "action": "query",
                "prop": "coordinates",
                "titles": "|".join(batch),
                "format": "json",
                "redirects": 1
            }
            try:
                data = request_handler.get_json_cached(
                    "https://en.wikipedia.org/w/api.php",
                    params=params,
                    cache_key=f"wiki_batch_coords:{hash('|'.join(batch))}",
                    ttl=config.CACHE_TTL_COORDS,
                    refresh=refresh
                )
                q = data.get("query", {}) if data else {}
                pages = q.get("pages", {})

                # Build a resolver from normalized/redirects 'to' -> original city
                resolved_to_city = {}
                for norm in q.get("normalized", []) or []:
                    src = norm.get("from")
                    dst = norm.get("to")
                    if src in title_to_city and dst:
                        resolved_to_city[dst] = title_to_city[src]
                for red in q.get("redirects", []) or []:
                    src = red.get("from")
                    dst = red.get("to")
                    if src in title_to_city and dst:
                        resolved_to_city[dst] = title_to_city[src]

                for page in pages.values():
                    title = page.get("title")
                    coords_list = page.get("coordinates") or []
                    if title and coords_list:
                        coords = coords_list[0]
                        city = title_to_city.get(title) or resolved_to_city.get(title)
                        if city and city not in results:
                            results[city] = {"lat": coords.get("lat"), "lon": coords.get("lon")}
            except Exception as e:
                logger.debug(f"Batch Wikipedia coords fetch failed: {e}")
                continue
        
        return results
    
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
            wiki_coords = wikipedia_provider.get_city_coordinates(city_name, country=country, refresh=refresh)
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
