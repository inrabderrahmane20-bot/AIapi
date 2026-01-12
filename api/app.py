import os
import re
import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import quote_plus, unquote
import requests
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from dataclasses import dataclass, field

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    # Your two allowed domains
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: [
        "https://www.traveltto.com",
        "https://traveltto.vercel.app",
        "http://localhost:3000"
    ])
    
    # Cache TTLs
    CACHE_TTL: int = 7200  # 2 hours
    CACHE_TTL_IMAGES: int = 86400  # 24 hours
    CACHE_TTL_PREVIEW: int = 3600  # 1 hour
    
    # Performance settings
    REQUEST_TIMEOUT: int = 10
    
    # Pagination settings
    CITIES_PER_PAGE: int = 50
    
    # Client-side caching
    ENABLE_CLIENT_CACHE: bool = True
    CLIENT_CACHE_TTL: int = 86400  # 24 hours
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CityExplorer")

# ==================== CACHING SYSTEM ====================
class Cache:
    def __init__(self):
        self.memory_cache = {}
    
    def get(self, key: str):
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if time.time() - item.get('timestamp', 0) < config.CACHE_TTL:
                return item.get('value')
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        item = {
            'value': value,
            'timestamp': time.time()
        }
        self.memory_cache[key] = item
    
    def delete(self, key: str):
        self.memory_cache.pop(key, None)

cache = Cache()

# ==================== IMAGE FETCHER ====================
class ImageFetcher:
    def __init__(self):
        self.wikimedia_api = "https://commons.wikimedia.org/w/api.php"
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
    
    def get_city_image(self, city_name: str, country: str = None) -> Dict:
        """Get one representative image for a city"""
        cache_key = f"city_image:{city_name}:{country}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Always use placeholder for speed
        image = self._get_placeholder_image(city_name)
        cache.set(cache_key, image, 3600)
        return image
    
    def get_landmark_images(self, landmark_name: str, city_name: str = None) -> List[Dict]:
        """Get images for a specific landmark"""
        cache_key = f"landmark_images:{landmark_name}:{city_name}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Use placeholder for speed
        images = [self._get_placeholder_image(landmark_name)]
        cache.set(cache_key, images, config.CACHE_TTL_IMAGES)
        return images
    
    def _get_placeholder_image(self, name: str) -> Dict:
        """Generate a placeholder image URL"""
        encoded_name = quote_plus(name[:30])
        colors = ['3388ff', 'ff5733', '33ff57', 'ff33a1', 'a133ff']
        color = colors[hash(name) % len(colors)]
        
        return {
            'url': f'https://via.placeholder.com/800x600/{color}/ffffff?text={encoded_name}',
            'title': name,
            'description': f'Image of {name}',
            'source': 'placeholder',
            'width': 800,
            'height': 600,
            'is_placeholder': True
        }

image_fetcher = ImageFetcher()

# ==================== CITY DATA PROVIDER ====================
class CityDataProvider:
    def __init__(self):
        self.city_descriptions = self._load_city_descriptions()
        self.city_coordinates = self._load_city_coordinates()
    
    def _load_city_coordinates(self) -> Dict:
        """Coordinates for Moroccan cities"""
        return {
            "Marrakech": {"lat": 31.6295, "lon": -7.9811},
            "Casablanca": {"lat": 33.5731, "lon": -7.5898},
            "Fez": {"lat": 34.0181, "lon": -5.0078},
            "Tangier": {"lat": 35.7595, "lon": -5.8340},
            "Rabat": {"lat": 34.0209, "lon": -6.8416},
            "Agadir": {"lat": 30.4278, "lon": -9.5981},
            "Meknes": {"lat": 33.8935, "lon": -5.5473},
            "Oujda": {"lat": 34.6819, "lon": -1.9086},
            "Kenitra": {"lat": 34.2541, "lon": -6.5890},
            "Tetouan": {"lat": 35.5762, "lon": -5.3684},
            "Safi": {"lat": 32.2833, "lon": -9.2333},
            "El Jadida": {"lat": 33.2568, "lon": -8.5088},
            "Nador": {"lat": 35.1686, "lon": -2.9333},
            "Settat": {"lat": 33.0010, "lon": -7.6166},
            "Beni Mellal": {"lat": 32.3373, "lon": -6.3498},
            "Khourigba": {"lat": 32.8800, "lon": -6.9060},
            "Al Hoceima": {"lat": 35.2510, "lon": -3.9373},
            "Larache": {"lat": 35.1918, "lon": -6.1557},
            "Taza": {"lat": 34.2234, "lon": -4.0066},
            "Errachidia": {"lat": 31.9319, "lon": -4.4246},
            "Taroudant": {"lat": 30.4720, "lon": -8.8749},
            "Essaouira": {"lat": 31.5085, "lon": -9.7595},
            "Chefchaouen": {"lat": 35.1714, "lon": -5.2699},
            "Ouarzazate": {"lat": 30.9335, "lon": -6.9370},
            "Ifrane": {"lat": 33.5260, "lon": -5.1106},
            "Azrou": {"lat": 33.4342, "lon": -5.2213},
            "Midelt": {"lat": 32.6805, "lon": -4.7361},
            "Guelmim": {"lat": 28.9884, "lon": -10.0528},
            "Dakhla": {"lat": 23.7141, "lon": -15.9368},
            "Laayoune": {"lat": 27.1253, "lon": -13.1625},
            "Asilah": {"lat": 35.4662, "lon": -6.0404},
            "Sidi Ifni": {"lat": 29.3792, "lon": -10.1750},
            "Tiznit": {"lat": 29.6974, "lon": -9.7316},
            "Khenifra": {"lat": 32.9395, "lon": -5.6675},
            "Berkane": {"lat": 34.9172, "lon": -2.3197},
            "Taourirt": {"lat": 34.4073, "lon": -2.8978},
            "Figuig": {"lat": 32.1097, "lon": -1.2284},
            "Sidi Kacem": {"lat": 34.2266, "lon": -5.7066},
            "Skhirat": {"lat": 33.8524, "lon": -7.0317},
            "Temara": {"lat": 33.9274, "lon": -6.9160},
            "Mohammedia": {"lat": 33.6879, "lon": -7.3829},
            "Ben Guerir": {"lat": 32.2314, "lon": -7.9526},
            "Tifelt": {"lat": 33.8948, "lon": -6.3094},
            "Bouznika": {"lat": 33.7894, "lon": -7.1596},
            "M'diq": {"lat": 35.6855, "lon": -5.3206},
            "Fnideq": {"lat": 35.8486, "lon": -5.3575},
            "Martil": {"lat": 35.6098, "lon": -5.2755},
            "Berrechid": {"lat": 33.2602, "lon": -7.5826},
            "Sidi Bennour": {"lat": 32.6523, "lon": -8.4277},
            "Youssoufia": {"lat": 32.2463, "lon": -8.5290},
            "Jerada": {"lat": 34.3102, "lon": -2.1630},
            "Oulad Teima": {"lat": 30.3950, "lon": -9.2086},
            "Benslimane": {"lat": 33.6115, "lon": -7.1216},
            "Ait Melloul": {"lat": 30.3342, "lon": -9.4972},
            "Sidi Slimane": {"lat": 34.2642, "lon": -5.9254},
            "Tan-Tan": {"lat": 28.4371, "lon": -11.1034},
            "Ouezzane": {"lat": 34.7956, "lon": -5.5780},
            "Sefrou": {"lat": 33.8304, "lon": -4.8351},
            "Boulemane": {"lat": 33.3625, "lon": -4.7303},
            "Taounate": {"lat": 34.5361, "lon": -4.6400},
            "Goulmima": {"lat": 31.6826, "lon": -4.9548},
            "Midar": {"lat": 34.9392, "lon": -3.5325},
            "Zagora": {"lat": 30.3316, "lon": -5.8376},
            "Sidi Yahya Zaer": {"lat": 33.6667, "lon": -6.7167},
        }
    
    def _load_city_descriptions(self) -> Dict:
        """Descriptions for all cities starting with 'About the city:'"""
        return {
            # Moroccan Cities
            "Marrakech": "About the city: Known as the 'Red City' for its distinctive red sandstone buildings, Marrakech is a vibrant cultural hub famous for its historic medina, bustling souks, and the iconic Jemaa el-Fnaa square.",
            "Casablanca": "About the city: Morocco's largest city and economic capital, Casablanca is a modern metropolis famous for its stunning Hassan II Mosque, art deco architecture, and vibrant seaside corniche.",
            "Fez": "About the city: The spiritual and cultural capital of Morocco, Fez is home to the world's oldest university and a UNESCO-listed medina that preserves centuries of Islamic architecture and traditional craftsmanship.",
            "Tangier": "About the city: A historic port city straddling Africa and Europe, Tangier has long attracted artists and writers with its unique blend of Moroccan, European, and African influences.",
            "Rabat": "About the city: Morocco's political and administrative capital, Rabat features historic sites like the Hassan Tower and Kasbah of the Udayas alongside modern government buildings.",
            "Agadir": "About the city: A modern beach resort city on Morocco's southern Atlantic coast, Agadir is famous for its beautiful beaches, modern marina, and year-round sunshine.",
            "Meknes": "About the city: Once the imperial capital of Morocco under Sultan Moulay Ismail, Meknes is known for its grand gates, extensive royal stables, and well-preserved historic monuments.",
            "Essaouira": "About the city: A charming coastal town known for its fortified medina, fresh seafood, and strong winds that make it a paradise for windsurfers and kitesurfers.",
            "Chefchaouen": "About the city: The famous 'Blue City' nestled in the Rif Mountains, Chefchaouen is renowned for its striking blue-washed buildings, relaxed atmosphere, and stunning mountain scenery.",
            "Ouarzazate": "About the city: Known as the 'Door of the Desert', Ouarzazate serves as a gateway to the Sahara and is famous for its kasbahs and as a filming location for Hollywood movies.",
            "Oujda": "About the city: The capital of eastern Morocco, Oujda is known for its Andalusian-influenced architecture and strategic location near the Algerian border.",
            "Kenitra": "About the city: A major port city on the Sebou River, Kenitra is an important industrial and agricultural center with modern infrastructure.",
            "Tetouan": "About the city: Known for its distinctive white architecture and strong Spanish influence, Tetouan features a UNESCO-listed medina with well-preserved Andalusian heritage.",
            "Safi": "About the city: A historic port city famous for its pottery and ceramics, Safi has been a center for craftsmanship since the 11th century.",
            "El Jadida": "About the city: A coastal city known for its Portuguese fortifications and the unique underground cistern, part of a UNESCO World Heritage site.",
            "Nador": "About the city: A port city on the Mediterranean coast near the Spanish enclave of Melilla, known for its beautiful lagoon and beaches.",
            "Settat": "About the city: An important agricultural and commercial center located between Casablanca and Marrakech.",
            "Beni Mellal": "About the city: The economic capital of the Tadla-Azilal region, known for its fertile plains and the nearby Ain Asserdoun springs.",
            "Khourigba": "About the city: The center of Morocco's phosphate mining industry, located in the phosphate plateau region.",
            "Al Hoceima": "About the city: A picturesque Mediterranean port city in the Rif Mountains, known for its beautiful beaches and Spanish colonial architecture.",
            "Larache": "About the city: A historic Atlantic port city with Spanish influences and important archaeological sites nearby.",
            "Taza": "About the city: A strategic mountain city guarding the pass between the Rif and Middle Atlas mountains.",
            "Errachidia": "About the city: The capital of the Draa-Tafilalet region, serving as gateway to the Sahara Desert and its stunning oases.",
            "Taroudant": "About the city: Known as 'Little Marrakech', this walled city features impressive ramparts and a traditional souk atmosphere.",
            "Ifrane": "About the city: A unique mountain resort town often called 'Little Switzerland' for its alpine architecture and clean streets.",
            "Azrou": "About the city: A Berber town in the Middle Atlas known for its cedar forests, handicrafts, and weekly markets.",
            "Midelt": "About the city: A market town in the Atlas Mountains known as the 'apple capital' of Morocco and gateway to the desert.",
            "Guelmim": "About the city: Known as the 'Gateway to the Sahara', famous for its camel market and desert landscapes.",
            "Dakhla": "About the city: A coastal city in Western Sahara known for its excellent kitesurfing conditions and oyster farming.",
            "Laayoune": "About the city: The largest city in Western Sahara, known for its modern architecture and desert surroundings.",
            "Asilah": "About the city: A charming coastal town famous for its annual arts festival, whitewashed buildings, and historic ramparts.",
            "Sidi Ifni": "About the city: A former Spanish enclave known for its art deco architecture and beautiful beaches along the Atlantic coast.",
            "Tiznit": "About the city: A historic walled city famous for its silver jewelry, traditional crafts, and annual festival.",
            "Khenifra": "About the city: A city in the Middle Atlas known for its Berber heritage, beautiful lakes, and traditional carpet weaving.",
            "Berkane": "About the city: Known as the 'Orange Capital' of Morocco, famous for its citrus production and agricultural wealth.",
            "Taourirt": "About the city: A city in eastern Morocco known for its historic kasbah and strategic location on trade routes.",
            "Figuig": "About the city: An oasis city on the Algerian border, famous for its date palms and traditional mud-brick architecture.",
            "Sidi Kacem": "About the city: An important railway junction and agricultural center in northwestern Morocco.",
            "Skhirat": "About the city: A coastal town near Rabat known for its royal palace and beautiful beaches.",
            "Temara": "About the city: A suburb of Rabat known for its beaches, modern developments, and recreational facilities.",
            "Mohammedia": "About the city: A port city and industrial center with beautiful beaches and the largest oil refinery in Morocco.",
            "Ben Guerir": "About the city: Known for its phosphate mines and recent development as a university and technology hub.",
            "Tifelt": "About the city: A town in the Rabat-Salé-Kénitra region known for its agricultural production and traditional markets.",
            "Bouznika": "About the city: A coastal resort town between Rabat and Casablanca, popular for its beaches and summer festivals.",
            "M'diq": "About the city: A Mediterranean port town known for its fishing harbor and tourist facilities.",
            "Fnideq": "About the city: A border town near Ceuta, known for its shopping markets and cross-border trade.",
            "Martil": "About the city: A popular beach resort town near Tetouan with beautiful sandy beaches and waterfront promenade.",
            "Berrechid": "About the city: An industrial and agricultural city known for its food processing industries and agricultural markets.",
            "Sidi Bennour": "About the city: An agricultural town in the Casablanca-Settat region known for its sugar production and farming.",
            "Youssoufia": "About the city: A mining town in central Morocco known for its phosphate extraction and industrial facilities.",
            "Jerada": "About the city: A former mining town in northeastern Morocco, now focusing on renewable energy projects.",
            "Oulad Teima": "About the city: An agricultural town in the Souss-Massa region known for its argan oil production and markets.",
            "Benslimane": "About the city: A town known for its eucalyptus forests, agricultural production, and pleasant climate.",
            "Ait Melloul": "About the city: A suburb of Agadir known for its agricultural markets and proximity to the airport.",
            "Sidi Slimane": "About the city: An agricultural town in the Gharb region known for its sugar refinery and farming.",
            "Tan-Tan": "About the city: A city in southern Morocco known for its annual moussem (festival) and desert culture.",
            "Ouezzane": "About the city: A spiritual city in northwestern Morocco known for its religious heritage and olive production.",
            "Sefrou": "About the city: Known as the 'City of Cherries', famous for its annual cherry festival and historic medina.",
            "Boulemane": "About the city: A town in the Middle Atlas known for its pastoral landscapes and traditional Berber culture.",
            "Taounate": "About the city: A town in the Fès-Meknès region known for its olive oil production and agricultural terraces.",
            "Goulmima": "About the city: An oasis town in southeastern Morocco known for its ksar (fortified village) and date production.",
            "Midar": "About the city: A town in the Rif Mountains known for its agricultural markets and traditional crafts.",
            "Zagora": "About the city: A desert town famous as the starting point for camel treks into the Sahara Desert.",
            "Sidi Yahya Zaer": "About the city: A town in the Rabat region known for its agricultural production and rural landscapes.",
        }
    
    def get_city_preview(self, city_name: str, country: str = None, region: str = None) -> Dict:
        """Get minimal preview data for city listing"""
        cache_key = f"preview:{city_name}:{country}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Generate city ID
        city_id = re.sub(r'[^\w\s-]', '', city_name.lower())
        city_id = re.sub(r'[-\s]+', '-', city_id)
        
        # Get coordinates
        coordinates = self.city_coordinates.get(city_name, None)
        
        # Get image
        image = image_fetcher.get_city_image(city_name, country)
        
        # Get description
        description = self.city_descriptions.get(
            city_name, 
            f"About the city: {city_name} is a significant city in {country or 'Morocco'}, known for its unique culture and historical importance."
        )
        
        preview = {
            "id": city_id,
            "name": city_name,
            "country": country,
            "region": region,
            "coordinates": coordinates,
            "image": image,
            "summary": description,
            "has_details": False
        }
        
        cache.set(cache_key, preview, config.CACHE_TTL_PREVIEW)
        return preview
    
    def get_city_details(self, city_name: str, country: str = None, region: str = None) -> Dict:
        """Get full details for a city"""
        cache_key = f"details:{city_name}:{country}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Start with preview
        preview = self.get_city_preview(city_name, country, region)
        
        # Get landmarks
        landmarks = self._get_landmarks(city_name)
        
        # Build full details
        details = {
            **preview,
            "description": preview["summary"],  # Use the same description for now
            "landmarks": landmarks,
            "additional_images": [],
            "has_details": True,
            "last_updated": time.time()
        }
        
        cache.set(cache_key, details, config.CACHE_TTL)
        return details
    
    def _get_landmarks(self, city_name: str) -> List[Dict]:
        """Get landmarks for a city"""
        landmarks_data = {
            "Marrakech": [
                {"name": "Jemaa el-Fnaa", "description": "The main square and market place in Marrakech's medina."},
                {"name": "Bahia Palace", "description": "A masterpiece of Moroccan architecture from the 19th century."},
                {"name": "Koutoubia Mosque", "description": "The largest mosque in Marrakech with a 77-meter minaret."},
                {"name": "Saadian Tombs", "description": "Historical burial ground of the Saadian dynasty."},
                {"name": "Majorelle Garden", "description": "A botanical garden created by French painter Jacques Majorelle."}
            ],
            "Casablanca": [
                {"name": "Hassan II Mosque", "description": "The largest mosque in Africa with a minaret 210 meters high."},
                {"name": "Old Medina", "description": "The historic heart of Casablanca with traditional markets."},
                {"name": "Corniche", "description": "A seaside promenade with beaches, restaurants, and clubs."},
                {"name": "Mohammed V Square", "description": "The main square surrounded by important public buildings."},
                {"name": "Sacred Heart Cathedral", "description": "A former Catholic cathedral in art deco style."}
            ],
            "Fez": [
                {"name": "Fes el Bali", "description": "The oldest walled part of Fez, a UNESCO World Heritage site."},
                {"name": "Al-Qarawiyyin University", "description": "Founded in 859, it's the oldest continuously operating university in the world."},
                {"name": "Bou Inania Madrasa", "description": "A 14th-century Islamic school and mosque."},
                {"name": "Chouara Tannery", "description": "One of the oldest tanneries in the world, operating since the 11th century."},
                {"name": "Royal Palace", "description": "The ceremonial palace of the King of Morocco in Fez."}
            ],
            "Rabat": [
                {"name": "Hassan Tower", "description": "An incomplete minaret of what was intended to be the world's largest mosque."},
                {"name": "Kasbah of the Udayas", "description": "A fortified city with Andalusian gardens and blue-white walls."},
                {"name": "Chellah", "description": "A medieval fortified Muslim necropolis with Roman ruins."},
                {"name": "Royal Palace", "description": "The primary and official residence of the king of Morocco."},
                {"name": "Mohammed V Mausoleum", "description": "The final resting place of Moroccan kings."}
            ]
        }
        
        if city_name in landmarks_data:
            return landmarks_data[city_name]
        
        # Generic landmarks for other cities
        generic_landmarks = [
            {"name": "Historic Medina", "description": f"The traditional old town of {city_name} with narrow streets and markets."},
            {"name": "Main Square", "description": f"The central gathering place and social hub of {city_name}."},
            {"name": "Local Museum", "description": f"A museum showcasing the history and culture of {city_name}."},
            {"name": "Traditional Market", "description": f"A bustling market offering local products and crafts."},
            {"name": "City Park", "description": f"A green space for relaxation and recreation in {city_name}."}
        ]
        
        return generic_landmarks[:3]

city_provider = CityDataProvider()

# ==================== DATA ====================
# Moroccan Cities (70+ cities)
MOROCCAN_CITIES = [
    {"name": "Marrakech", "country": "Morocco", "region": "Africa"},
    {"name": "Casablanca", "country": "Morocco", "region": "Africa"},
    {"name": "Fez", "country": "Morocco", "region": "Africa"},
    {"name": "Tangier", "country": "Morocco", "region": "Africa"},
    {"name": "Rabat", "country": "Morocco", "region": "Africa"},
    {"name": "Agadir", "country": "Morocco", "region": "Africa"},
    {"name": "Meknes", "country": "Morocco", "region": "Africa"},
    {"name": "Oujda", "country": "Morocco", "region": "Africa"},
    {"name": "Kenitra", "country": "Morocco", "region": "Africa"},
    {"name": "Tetouan", "country": "Morocco", "region": "Africa"},
    {"name": "Safi", "country": "Morocco", "region": "Africa"},
    {"name": "El Jadida", "country": "Morocco", "region": "Africa"},
    {"name": "Nador", "country": "Morocco", "region": "Africa"},
    {"name": "Settat", "country": "Morocco", "region": "Africa"},
    {"name": "Beni Mellal", "country": "Morocco", "region": "Africa"},
    {"name": "Khourigba", "country": "Morocco", "region": "Africa"},
    {"name": "Al Hoceima", "country": "Morocco", "region": "Africa"},
    {"name": "Larache", "country": "Morocco", "region": "Africa"},
    {"name": "Taza", "country": "Morocco", "region": "Africa"},
    {"name": "Errachidia", "country": "Morocco", "region": "Africa"},
    {"name": "Taroudant", "country": "Morocco", "region": "Africa"},
    {"name": "Essaouira", "country": "Morocco", "region": "Africa"},
    {"name": "Chefchaouen", "country": "Morocco", "region": "Africa"},
    {"name": "Ouarzazate", "country": "Morocco", "region": "Africa"},
    {"name": "Ifrane", "country": "Morocco", "region": "Africa"},
    {"name": "Azrou", "country": "Morocco", "region": "Africa"},
    {"name": "Midelt", "country": "Morocco", "region": "Africa"},
    {"name": "Guelmim", "country": "Morocco", "region": "Africa"},
    {"name": "Dakhla", "country": "Morocco", "region": "Africa"},
    {"name": "Laayoune", "country": "Morocco", "region": "Africa"},
    {"name": "Asilah", "country": "Morocco", "region": "Africa"},
    {"name": "Sidi Ifni", "country": "Morocco", "region": "Africa"},
    {"name": "Tiznit", "country": "Morocco", "region": "Africa"},
    {"name": "Khenifra", "country": "Morocco", "region": "Africa"},
    {"name": "Berkane", "country": "Morocco", "region": "Africa"},
    {"name": "Taourirt", "country": "Morocco", "region": "Africa"},
    {"name": "Figuig", "country": "Morocco", "region": "Africa"},
    {"name": "Sidi Kacem", "country": "Morocco", "region": "Africa"},
    {"name": "Skhirat", "country": "Morocco", "region": "Africa"},
    {"name": "Temara", "country": "Morocco", "region": "Africa"},
    {"name": "Mohammedia", "country": "Morocco", "region": "Africa"},
    {"name": "Ben Guerir", "country": "Morocco", "region": "Africa"},
    {"name": "Tifelt", "country": "Morocco", "region": "Africa"},
    {"name": "Bouznika", "country": "Morocco", "region": "Africa"},
    {"name": "M'diq", "country": "Morocco", "region": "Africa"},
    {"name": "Fnideq", "country": "Morocco", "region": "Africa"},
    {"name": "Martil", "country": "Morocco", "region": "Africa"},
    {"name": "Berrechid", "country": "Morocco", "region": "Africa"},
    {"name": "Sidi Bennour", "country": "Morocco", "region": "Africa"},
    {"name": "Youssoufia", "country": "Morocco", "region": "Africa"},
    {"name": "Jerada", "country": "Morocco", "region": "Africa"},
    {"name": "Oulad Teima", "country": "Morocco", "region": "Africa"},
    {"name": "Benslimane", "country": "Morocco", "region": "Africa"},
    {"name": "Ait Melloul", "country": "Morocco", "region": "Africa"},
    {"name": "Sidi Slimane", "country": "Morocco", "region": "Africa"},
    {"name": "Tan-Tan", "country": "Morocco", "region": "Africa"},
    {"name": "Ouezzane", "country": "Morocco", "region": "Africa"},
    {"name": "Sefrou", "country": "Morocco", "region": "Africa"},
    {"name": "Boulemane", "country": "Morocco", "region": "Africa"},
    {"name": "Taounate", "country": "Morocco", "region": "Africa"},
    {"name": "Goulmima", "country": "Morocco", "region": "Africa"},
    {"name": "Midar", "country": "Morocco", "region": "Africa"},
    {"name": "Zagora", "country": "Morocco", "region": "Africa"},
    {"name": "Sidi Yahya Zaer", "country": "Morocco", "region": "Africa"},
]

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

# ==================== FLASK APP ====================
app = Flask(__name__)

# CORS configuration
CORS(app, 
     origins=config.ALLOWED_ORIGINS,
     methods=["GET", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "If-None-Match"],
     supports_credentials=True,
     max_age=3600)

# Helper function for cache headers
def add_cache_headers(response, ttl=None):
    """Add cache control headers to response"""
    if config.ENABLE_CLIENT_CACHE:
        ttl = ttl or config.CLIENT_CACHE_TTL
        response.headers['Cache-Control'] = f'public, max-age={ttl}'
        response.headers['Expires'] = (datetime.utcnow() + timedelta(seconds=ttl)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    return response

# ==================== ROUTES ====================

@app.route('/')
def home():
    return jsonify({
        "name": "City Explorer API",
        "version": "2.0",
        "status": "online",
        "description": "API for exploring Moroccan cities with detailed information",
        "endpoints": {
            "health": "/api/health",
            "cities": "/api/cities",
            "city_details": "/api/cities/<city_name>",
            "morocco": "/api/morocco",
            "morocco_city": "/api/morocco/<city_name>",
            "search": "/api/search",
            "regions": "/api/regions"
        },
        "statistics": {
            "total_cities": len(WORLD_CITIES),
            "moroccan_cities": len(MOROCCAN_CITIES)
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "statistics": {
            "total_cities": len(WORLD_CITIES),
            "moroccan_cities": len(MOROCCAN_CITIES)
        },
        "cache_enabled": True,
        "performance_mode": "optimized"
    })

@app.route('/api/regions')
def get_regions():
    """Get all available regions from city data"""
    regions = sorted(set(city.get('region') for city in WORLD_CITIES if city.get('region')))
    
    # Count cities per region
    region_counts = {}
    for city in WORLD_CITIES:
        region = city.get('region')
        if region:
            region_counts[region] = region_counts.get(region, 0) + 1
    
    return jsonify({
        "success": True,
        "regions": regions,
        "counts": region_counts,
        "total_regions": len(regions)
    })

@app.route('/api/cities')
def get_cities():
    """
    Get paginated list of cities with minimal data
    Returns 50 cities per page by default
    """
    # Parse query parameters
    page = max(1, int(request.args.get('page', 1)))
    limit = min(50, max(1, int(request.args.get('limit', config.CITIES_PER_PAGE))))
    region = request.args.get('region')
    country = request.args.get('country')
    
    # Cache key for this specific request
    cache_key = f"cities:{page}:{limit}:{region}:{country}"
    cached = cache.get(cache_key)
    
    if cached:
        response = jsonify(cached)
        return add_cache_headers(response, 300)
    
    # Filter cities
    filtered = WORLD_CITIES
    if region:
        filtered = [c for c in filtered if c.get('region') == region]
    if country:
        filtered = [c for c in filtered if c.get('country') == country]
    
    # Pagination
    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    page_cities = filtered[start:end]
    
    # Fetch previews
    cities_list = []
    for city_info in page_cities:
        preview = city_provider.get_city_preview(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        cities_list.append(preview)
    
    # Build response
    response_data = {
        "success": True,
        "data": cities_list,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": max(1, (total + limit - 1) // limit),
            "next_page": page + 1 if end < total else None,
            "prev_page": page - 1 if page > 1 else None
        },
        "cache_info": {
            "client_cache_ttl": 300,
            "images_are_optimized": True,
            "descriptions_available": True
        }
    }
    
    # Cache the response
    cache.set(cache_key, response_data, 300)
    
    response = jsonify(response_data)
    return add_cache_headers(response, 300)

@app.route('/api/cities/<path:city_name>')
def get_city_details(city_name):
    """Get full details for a specific city"""
    city_name = unquote(city_name)
    
    # Check ETag for client cache
    etag = hashlib.md5(f"city:{city_name}".encode()).hexdigest()
    if_none_match = request.headers.get('If-None-Match')
    if if_none_match == etag:
        return Response(status=304)
    
    # Find city
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
        details = city_provider.get_city_details(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        
        # Add ETag for caching
        details['_cache'] = {
            "etag": etag,
            "last_updated": time.time()
        }
        
        response = jsonify({
            "success": True,
            "data": details
        })
        
        # Add cache headers
        response.headers['ETag'] = etag
        response.headers['Cache-Control'] = f'public, max-age={config.CLIENT_CACHE_TTL}'
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching city details for {city_name}: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch city details"
        }), 500

@app.route('/api/cities/<path:city_name>/landmarks/<path:landmark_name>/images')
def get_landmark_images(city_name, landmark_name):
    """Get images for a specific landmark"""
    city_name = unquote(city_name)
    landmark_name = unquote(landmark_name)
    
    try:
        images = image_fetcher.get_landmark_images(landmark_name, city_name)
        
        return jsonify({
            "success": True,
            "data": {
                "city": city_name,
                "landmark": landmark_name,
                "images": images
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching landmark images: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch landmark images"
        }), 500

@app.route('/api/morocco')
def get_moroccan_cities():
    """Get only Moroccan cities"""
    # Same logic as /api/cities but filtered
    page = max(1, int(request.args.get('page', 1)))
    limit = min(50, max(1, int(request.args.get('limit', config.CITIES_PER_PAGE))))
    
    cache_key = f"morocco:{page}:{limit}"
    cached = cache.get(cache_key)
    
    if cached:
        response = jsonify(cached)
        return add_cache_headers(response, 300)
    
    total = len(MOROCCAN_CITIES)
    start = (page - 1) * limit
    end = start + limit
    page_cities = MOROCCAN_CITIES[start:end]
    
    cities_list = []
    for city_info in page_cities:
        preview = city_provider.get_city_preview(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        cities_list.append(preview)
    
    response_data = {
        "success": True,
        "data": cities_list,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": max(1, (total + limit - 1) // limit),
            "next_page": page + 1 if end < total else None,
            "prev_page": page - 1 if page > 1 else None
        },
        "country_info": {
            "name": "Morocco",
            "total_cities": total,
            "description": "Kingdom of Morocco - A North African country with diverse landscapes and rich history."
        }
    }
    
    cache.set(cache_key, response_data, 300)
    
    response = jsonify(response_data)
    return add_cache_headers(response, 300)

@app.route('/api/morocco/<path:city_name>')
def get_moroccan_city_details(city_name):
    """Get details for a Moroccan city"""
    city_name = unquote(city_name)
    
    # Find Moroccan city
    city_info = None
    for city in MOROCCAN_CITIES:
        if city['name'].lower() == city_name.lower():
            city_info = city
            break
    
    if not city_info:
        return jsonify({
            "success": False,
            "error": "Moroccan city not found"
        }), 404
    
    try:
        details = city_provider.get_city_details(
            city_info['name'],
            city_info.get('country'),
            city_info.get('region')
        )
        
        etag = hashlib.md5(f"morocco:{city_name}".encode()).hexdigest()
        details['_cache'] = {"etag": etag}
        
        response = jsonify({
            "success": True,
            "data": details
        })
        
        response.headers['ETag'] = etag
        response.headers['Cache-Control'] = f'public, max-age={config.CLIENT_CACHE_TTL}'
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching Moroccan city details: {e}")
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
            "error": "Query must be at least 2 characters"
        }), 400
    
    limit = min(20, max(1, int(request.args.get('limit', 10))))
    
    # Search in city names
    results = []
    for city in WORLD_CITIES:
        if query.lower() in city['name'].lower():
            preview = city_provider.get_city_preview(
                city['name'],
                city.get('country'),
                city.get('region')
            )
            results.append(preview)
            
            if len(results) >= limit:
                break
    
    return jsonify({
        "success": True,
        "query": query,
        "results": results,
        "count": len(results)
    })

# ==================== MIDDLEWARE ====================
@app.before_request
def before_request():
    """Log requests and check origin"""
    origin = request.headers.get('Origin')
    if origin and origin not in config.ALLOWED_ORIGINS and not request.path.startswith('/api/health'):
        logger.warning(f"Blocked request from unauthorized origin: {origin}")
        return jsonify({
            "success": False,
            "error": "Unauthorized origin"
        }), 403

@app.after_request
def after_request(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# ==================== VERCEL ENTRY POINT ====================
# This is important for Vercel serverless deployment
def handler(event, context):
    """Vercel serverless handler"""
    return app(event, context)

# ==================== MAIN ====================
if __name__ == '__main__':
    logger.info("Starting City Explorer API")
    logger.info(f"Total cities: {len(WORLD_CITIES)}")
    logger.info(f"Moroccan cities: {len(MOROCCAN_CITIES)}")
    
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False  # Set to False for production
    )