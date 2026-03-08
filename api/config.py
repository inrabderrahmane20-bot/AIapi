import os
import logging
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "7200"))
    CACHE_TTL_IMAGES: int = int(os.getenv("CACHE_TTL_IMAGES", "86400"))
    CACHE_TTL_COORDS: int = int(os.getenv("CACHE_TTL_COORDS", "259200"))
    
    MAPBOX_ACCESS_TOKEN: str = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    
    MAX_IMAGES_PER_REQUEST: int = int(os.getenv("MAX_IMAGES_PER_REQUEST", "50"))
    MAX_IMAGES_PREVIEW: int = 1
    
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    WIKIPEDIA_TIMEOUT: int = int(os.getenv("WIKIPEDIA_TIMEOUT", "20"))
    GEOLOCATOR_TIMEOUT: int = int(os.getenv("GEOLOCATOR_TIMEOUT", "10"))
    
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    FLASK_PORT: int = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/city_explorer_cache")
    
    MAX_BATCH_SIZE: int = 100
    MIN_IMAGE_WIDTH: int = 400
    MIN_IMAGE_HEIGHT: int = 300
    PREFERRED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.webp'])
    
    MIN_IMAGE_QUALITY_SCORE: int = 40
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()

ALLOWED_ORIGINS = ["https://www.traveltto.com", "https://traveltto.vercel.app"]
