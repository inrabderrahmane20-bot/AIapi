import time
import diskcache
import os
from typing import Any
from api.config import config
from api.utils.logger import logger

class MultiLevelCache:
    def __init__(self):
        self.memory_cache = {}
        self.disk_cache = None
        
        try:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            self.disk_cache = diskcache.Cache(config.CACHE_DIR)
            logger.info(f"✅ Disk cache initialized at: {config.CACHE_DIR}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize disk cache: {e}")
            self.disk_cache = diskcache.Cache()
    
    def get(self, key: str, default=None):
        if key in self.memory_cache:
            item = self.memory_cache.get(key)
            if item and time.time() - item.get('timestamp', 0) < config.CACHE_TTL:
                return item.get('value')
        
        if self.disk_cache:
            try:
                cached = self.disk_cache.get(key)
                if cached and time.time() - cached.get('timestamp', 0) < config.CACHE_TTL:
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
        
        self.memory_cache[key] = cache_item
        
        if self.disk_cache:
            try:
                self.disk_cache.set(key, cache_item, expire=ttl or config.CACHE_TTL)
            except Exception:
                pass
    
    def delete(self, key: str):
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if self.disk_cache:
            try:
                del self.disk_cache[key]
            except Exception:
                pass

cache = MultiLevelCache()
