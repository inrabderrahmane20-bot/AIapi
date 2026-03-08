import os
import time
from urllib.parse import unquote
from flask import Flask, jsonify, request
from flask_cors import CORS
from api.config import config
from api.utils.logger import logger
from api.utils.cache import cache
from api.data.cities import WORLD_CITIES, INTERLEAVED_WORLD_CITIES, MOROCCO_CITIES, REGIONS
from api.utils.city_data_manager import fetch_cities_parallel, fetch_city_details
import sys
import inspect

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ==================== CORS ORIGIN CHECK ====================
ALLOWED_ORIGINS = ["https://www.traveltto.com", "https://traveltto.vercel.app"]

def check_origin():
    """Middleware to check origin"""
    origin = request.headers.get('Origin')
    if origin and origin not in ALLOWED_ORIGINS:
        logger.warning(f"Blocked request from unauthorized origin: {origin}")
        return jsonify({"error": "Unauthorized origin"}), 403
    return None

# ==================== ROUTES ====================

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "City Explorer API is running",
        "version": "2.0",
        "documentation": "/api/docs"
    })

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "cache_stats": {
            "memory_items": len(cache.memory_cache),
            "disk_enabled": cache.disk_cache is not None
        }
    })

@app.route('/api/cities')
def get_cities():
    """
    Get paginated list of cities with minimal data (100 per page)
    Returns: name, region, small description, and one image
    """
    # Check origin
    origin_check = check_origin()
    if origin_check:
        return origin_check
    
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 100, type=int)
    region = request.args.get('region', type=str)
    country = request.args.get('country', type=str)
    fields_param = request.args.get('fields', type=str)
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    requested_fields = None
    if fields_param:
        requested_fields = [f.strip().lower() for f in fields_param.split(',')]
    
    # Filter cities
    # If no filters, use interleaved list for better region distribution
    if not region and not country:
        filtered_cities = INTERLEAVED_WORLD_CITIES
    else:
        filtered_cities = WORLD_CITIES
        if region:
            filtered_cities = [c for c in filtered_cities if c.get('region') == region]
        if country:
            filtered_cities = [c for c in filtered_cities if c.get('country') == country]
    
    total_cities = len(filtered_cities)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total_cities)
    
    # Slice the list for the current page
    cities_page = filtered_cities[start_idx:end_idx]
    
    # Fetch data in parallel
    cities_list = fetch_cities_parallel(cities_page, requested_fields, refresh=refresh)
    
    return jsonify({
        "success": True,
        "data": cities_list,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_cities,
            "pages": max(1, (total_cities + limit - 1) // limit),
            "next_page": page + 1 if end_idx < total_cities else None,
            "prev_page": page - 1 if page > 1 else None
        }
    })

@app.route('/api/cities/<city_id>')
def get_city_details(city_id):
    """
    Get comprehensive details for a specific city
    """
    # Check origin
    origin_check = check_origin()
    if origin_check:
        return origin_check
        
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    # Find the city in our database
    # Handle slug-like IDs (new-york) or names (New York)
    city_name_query = city_id.replace('-', ' ').lower()
    
    city_info = next((c for c in WORLD_CITIES if c['name'].lower() == city_name_query), None)
    
    if not city_info:
        return jsonify({"error": "City not found"}), 404
    
    # Fetch full details
    logger.info(f"Calling fetch_city_details for {city_info['name']} with refresh={refresh}")
    details = fetch_city_details(city_info, refresh=refresh)
    
    return jsonify({
        "success": True,
        "data": details
    })

@app.route('/api/morocco')
def get_morocco_cities():
    """
    Get paginated list of Moroccan cities only (100 per page)
    """
    # Check origin
    origin_check = check_origin()
    if origin_check:
        return origin_check
    
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 100, type=int)
    fields_param = request.args.get('fields', type=str)
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    requested_fields = None
    if fields_param:
        requested_fields = [f.strip().lower() for f in fields_param.split(',')]
    
    total_cities = len(MOROCCO_CITIES)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total_cities)
    
    # Slice list
    cities_page = MOROCCO_CITIES[start_idx:end_idx]
    
    # Fetch in parallel
    cities_list = fetch_cities_parallel(cities_page, requested_fields, refresh=refresh)
    
    return jsonify({
        "success": True,
        "country": "Morocco",
        "data": cities_list,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_cities,
            "pages": max(1, (total_cities + limit - 1) // limit),
            "next_page": page + 1 if end_idx < total_cities else None,
            "prev_page": page - 1 if page > 1 else None
        }
    })

@app.route('/api/morocco/<path:city_name>')
def get_morocco_city_details(city_name):
    """
    Get full details for a specific Moroccan city
    """
    # Check origin
    origin_check = check_origin()
    if origin_check:
        return origin_check
    
    city_name = unquote(city_name)
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    # Find city in MOROCCO_CITIES
    city_info = next((c for c in MOROCCO_CITIES if c['name'].lower() == city_name.lower()), None)
    
    if not city_info:
        return jsonify({
            "success": False,
            "error": f"Moroccan city '{city_name}' not found in database"
        }), 404
    
    # Fetch full details
    details = fetch_city_details(city_info, refresh=refresh)
    
    return jsonify({
        "success": True,
        "data": details
    })

@app.route('/api/regions')
def get_regions():
    return jsonify({
        "success": True,
        "data": REGIONS
    })

# Main Execution
if __name__ == '__main__':
    print(f"DEBUG: sys.path: {sys.path}")
    import api.utils.city_data_manager
    print(f"DEBUG: city_data_manager file: {api.utils.city_data_manager.__file__}")
    
    logger.info(f"Starting server on port {config.FLASK_PORT}")
    app.run(
        debug=config.FLASK_DEBUG, 
        host='0.0.0.0', 
        port=config.FLASK_PORT,
        threaded=True
    )
