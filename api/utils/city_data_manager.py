import concurrent.futures
from typing import List, Optional, Dict, Any
from api.config import config
from api.utils.logger import logger
from api.utils.cache import cache
from api.utils.image_fetcher import image_fetcher
from api.utils.wikipedia_provider import wikipedia_provider
from api.utils.coordinates_provider import coordinates_provider

def fetch_single_city_data(city_info: dict, requested_fields: List[str] = None, refresh: bool = False) -> dict:
    """Fetch data for a single city with optional field filtering"""
    city_name = city_info['name']
    
    # Generate cache key based on city and fields
    fields_key = "all" if not requested_fields else ",".join(sorted(requested_fields))
    cache_key = f"city_data:{city_name}:{fields_key}"
    
    # Check cache if not refreshing
    if not refresh:
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data

    # Default behavior: fetch everything if no fields specified
    fetch_all = requested_fields is None or len(requested_fields) == 0
    
    # Determine what to fetch based on requested fields
    should_fetch_summary = fetch_all or 'summary' in requested_fields
    should_fetch_image = fetch_all or 'image' in requested_fields
    should_fetch_coords = fetch_all or 'coordinates' in requested_fields or 'location' in requested_fields
    
    # Construct search query with country to avoid ambiguity (e.g. Rabat Malta vs Rabat Morocco)
    search_query = f"{city_name} {city_info.get('country', '')}".strip()
    
    result = {
        "id": city_name.lower().replace(' ', '-'),
        "name": city_name,
        "country": city_info.get('country'),
        "region": city_info.get('region'),
        "has_details": True
    }
    
    try:
        # 1. Get summary (Wikipedia)
        if should_fetch_summary:
            summary, _ = wikipedia_provider.get_city_summary(city_name, refresh=refresh)
            result["summary"] = summary or f"{city_name}, {city_info.get('country', 'a city')}"
        
        # 2. Get image (Wikimedia)
        if should_fetch_image:
            image = image_fetcher.get_one_representative_image(search_query, refresh=refresh)
            result["image"] = image
        
        # 3. Get coordinates (Nominatim)
        if should_fetch_coords:
            coordinates = coordinates_provider.get_coordinates(city_name, city_info.get('country'), refresh=refresh)
            result["coordinates"] = coordinates
        
        # Cache successful result (TTL: 24 hours)
        if result.get("has_details"):
            cache.set(cache_key, result, 86400)
            
    except Exception as e:
        logger.warning(f"Failed to load data for {city_name}: {e}")
        # If critical failure, mark has_details as False but still return basic info
        result["has_details"] = False
        if should_fetch_summary and "summary" not in result:
             result["summary"] = f"{city_name}, {city_info.get('country', 'a city')}"
    
    return result

def fetch_cities_parallel(cities: List[dict], requested_fields: List[str] = None, max_workers: int = 50, refresh: bool = False) -> List[dict]:
    """Fetch data for multiple cities in parallel"""
    if not cities:
        return []
        
    logger.info(f"Fetching data for {len(cities)} cities in parallel (fields={requested_fields}, workers={max_workers})")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Using map ensures results are returned in the same order as input
        results = list(executor.map(lambda c: fetch_single_city_data(c, requested_fields, refresh), cities))
        
    return results

def fetch_city_details(city_info: dict, refresh: bool = False) -> Dict[str, Any]:
    """Fetch comprehensive details for a single city (for detail view)"""
    city_name = city_info['name']
    cache_key = f"city_details:{city_name}"
    
    logger.info(f"DEBUG: fetch_city_details called for {city_name}, refresh={refresh}")

    # Check cache if not refreshing
    if not refresh:
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.info(f"DEBUG: Returning cached data for {city_name}")
            return cached_data
            
    logger.info(f"Fetching full details for {city_name} (refresh={refresh})")
    
    # Construct search query with country to avoid ambiguity
    search_query = f"{city_name} {city_info.get('country', '')}".strip()
    
    result = {
        "id": city_name.lower().replace(' ', '-'),
        "name": city_name,
        "country": city_info.get('country'),
        "region": city_info.get('region'),
        "population": city_info.get('population', 'Unknown'),
        "currency": city_info.get('currency', 'Unknown'),
        "language": city_info.get('language', 'Unknown'),
        "best_time_to_visit": city_info.get('best_time_to_visit', 'Spring/Autumn'),
        "details_loaded": True
    }
    
    try:
        # Execute independent fetches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 1. Summary & Title
            future_summary = executor.submit(wikipedia_provider.get_city_summary, city_name, refresh=refresh)
            
            # 2. Coordinates
            future_coords = executor.submit(coordinates_provider.get_coordinates, city_name, city_info.get('country'), refresh=refresh)
            
            # 3. Landmarks
            future_landmarks = executor.submit(wikipedia_provider.extract_landmarks, city_name, refresh=refresh)
            
            # 4. Images (Gallery)
            future_images = executor.submit(image_fetcher.get_images_for_city, search_query, limit=10, refresh=refresh)

            # 5. Main Image (Panorama preferred)
            # Use full search query (City + Country) to avoid ambiguity (e.g. Rabat Malta vs Rabat Morocco)
            future_main_image = executor.submit(image_fetcher.get_one_representative_image, search_query, refresh=refresh)
            
            # Collect results
            summary, title = future_summary.result()
            coordinates = future_coords.result()
            landmarks = future_landmarks.result()
            images = future_images.result()
            main_image = future_main_image.result()
            
            result["description"] = summary or f"Discover the beauty of {city_name}."
            result["title"] = title or city_name
            result["coordinates"] = coordinates
            result["landmarks"] = landmarks
            result["gallery"] = images
            
            # Set main image (prioritize dedicated panorama fetch)
            if main_image:
                result["image"] = main_image
            elif images:
                result["image"] = images[0]
            else:
                result["image"] = None
        
        # Cache successful result
        logger.info(f"DEBUG: Caching result for {city_name}")
        cache.set(cache_key, result, config.CACHE_TTL)
        
    except Exception as e:
        logger.error(f"Failed to fetch details for {city_name}: {e}")
        result["details_loaded"] = False
        result["error"] = str(e)
        
    return result
