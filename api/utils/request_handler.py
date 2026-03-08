import requests
import json
import hashlib
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from api.config import config
from api.utils.logger import logger
from api.utils.cache import cache

class SmartRequestHandler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CityExplorer/2.0; +https://traveltto.com)',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.Timeout, 
                                       requests.exceptions.ConnectionError))
    )
    def get_with_retry(self, url: str, params: dict = None, headers: dict = None, 
                       timeout: int = None) -> requests.Response:
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def get_json_cached(self, url: str, params: dict = None, headers: dict = None, 
                        cache_key: str = None, ttl: int = None, refresh: bool = False) -> Any:
        if not cache_key:
            cache_key = hashlib.md5(
                f"{url}{json.dumps(params or {}, sort_keys=True)}".encode()
            ).hexdigest()
        
        cached = None
        if not refresh:
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
            
            # Try to recover from cache if refresh failed
            if refresh and cached is None:
                cached = cache.get(cache_key)
            
            if cached is not None:
                logger.info(f"Using stale cache for {url}")
                return cached
            
            raise

request_handler = SmartRequestHandler()
