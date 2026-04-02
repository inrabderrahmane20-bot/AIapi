import re
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from api.config import config
from api.utils.logger import logger
from api.utils.cache import cache
from api.utils.request_handler import request_handler

class IntelligentImageFetcher:
    def __init__(self):
        self.wikimedia_api = "https://commons.wikimedia.org/w/api.php"
        
    def calculate_image_quality(self, image_info: dict, is_panorama: bool = False) -> int:
        score = 50
        
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        title = image_info.get('title', '').lower()
        
        # Resolution bonus (favor higher res)
        if width >= 3840: # 4K
            score += 50
        elif width >= 1920: # 1080p
            score += 40
        elif width >= 1200:
            score += 30
        elif width >= 800:
            score += 20
            
        # Aspect ratio bonus
        if height > 0:
            aspect_ratio = width / height
            
            if is_panorama:
                # Strict panorama scoring
                if aspect_ratio >= 2.0:
                    score += 50 # Massive bonus for true panoramas
                elif aspect_ratio >= 1.6:
                    score += 20
                else:
                    score -= 30 # Penalize non-panoramas heavily if requested
            else:
                # Standard scoring
                if 1.3 <= aspect_ratio <= 2.0: # Standard landscape
                    score += 20
                elif aspect_ratio > 2.0: # Wide is still good
                    score += 15
                elif aspect_ratio < 1.0: # Portrait
                    score -= 30 # Heavy penalty for portrait main images
        
        # Content relevance bonus
        if 'panorama' in title or 'view' in title or 'skyline' in title:
            score += 10
            
        # Format bonus
        url = image_info.get('url', '').lower()
        if any(fmt in url for fmt in ['.jpg', '.jpeg']):
            score += 5
            
        return min(100, max(0, score))
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def fetch_from_wikimedia(self, query: str, limit: int = 20, refresh: bool = False) -> List[Dict]:
        images = []
        
        try:
            # Add 'landmark' or 'view' to query to avoid maps/flags if not already present
            search_query = query
            
            params = {
                'action': 'query',
                'generator': 'search',
                'gsrsearch': search_query,
                'gsrnamespace': '6', # File namespace
                'gsrlimit': limit * 2, # Fetch more to filter later
                'prop': 'imageinfo',
                'iiprop': 'url|size|mime|extmetadata',
                'iiurlwidth': 1024, # Request a decent thumbnail size
                'format': 'json'
            }
            
            data = request_handler.get_json_cached(
                self.wikimedia_api,
                params=params,
                cache_key=f"wikimedia:{search_query}",
                ttl=config.CACHE_TTL_IMAGES,
                refresh=refresh,
                timeout=min(config.REQUEST_TIMEOUT, 5)
            )
            
            if not data or 'query' not in data:
                logger.debug(f"Wikimedia search for '{query}' returned no results.")
                return []
            
            pages = data.get('query', {}).get('pages', {})
            
            for page in pages.values():
                if 'imageinfo' in page:
                    info = page['imageinfo'][0]
                    
                    # Basic validity check
                    if not self._is_valid_image(info):
                        continue
                        
                    # strict filtering for relevance
                    if not self._is_relevant_content(page.get('title', ''), info, query):
                        continue

                    image_data = {
                        'url': info.get('url'), # Use full size URL for main image if possible
                        'thumb': info.get('thumburl'),
                        'title': page.get('title', '').replace('File:', ''),
                        'description': self._extract_description(info),
                        'source': 'wikimedia',
                        'width': info.get('width'),
                        'height': info.get('height'),
                        'quality_score': 0, # Placeholder
                        'page_url': info.get('descriptionurl')
                    }
                    
                    # Calculate score
                    score = self.calculate_image_quality(image_data, "panorama" in query.lower())
                    image_data['quality_score'] = score
                    
                    images.append(image_data)
            
            # Sort by score
            images.sort(key=lambda x: x['quality_score'], reverse=True)
            logger.debug(f"Found {len(images)} images for '{query}'. Top score: {images[0]['quality_score'] if images else 0}")
            
        except Exception as e:
            logger.warning(f"Wikimedia fetch failed for {query}: {e}")
        
        return images[:limit]

    def _is_valid_image(self, info: dict) -> bool:
        """Technical validity checks"""
        mime = info.get('mime', '')
        if not mime.startswith('image/'): return False
        
        width = info.get('width', 0)
        height = info.get('height', 0)
        if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT: return False
        
        url = info.get('url', '').lower()
        if not any(fmt in url for fmt in config.PREFERRED_IMAGE_FORMATS): return False
        
        return True

    def _is_relevant_content(self, title: str, info: dict, query: str) -> bool:
        """Content relevance checks"""
        title_lower = title.lower()
        url_lower = info.get('url', '').lower()
        
        # 1. Exclude non-photographic content
        excluded_terms = [
            'map', 'flag', 'coat of arms', 'logo', 'diagram', 'location', 'svg', 'plan', 'chart', 
            'population', 'demography', 'currency', 'coa', 'symbol', 'icon', 'stub',
            'atlas', 'drawing', 'engraving', 'illustration', 'sketch', 'painting', 'art',
            'poster', 'schema', 'layout', 'print', 'lithograph'
        ]
        if any(term in title_lower for term in excluded_terms) or any(term in url_lower for term in excluded_terms):
            return False
            
        # 2. Strict City Matching (if query contains city name)
        # We want to avoid "Map of X" or "Flag of X" leaking through
        # But allow "View of X", "X Skyline"
        
        # If looking for a panorama/view, reject vertical images strictly
        if "panorama" in query.lower() or "skyline" in query.lower() or "view" in query.lower():
            width = info.get('width', 1)
            height = info.get('height', 1)
            if height > width: # Portrait
                return False
                
        return True

    def _extract_description(self, image_info: dict) -> str:
        extmetadata = image_info.get('extmetadata', {})
        for field in ['ImageDescription', 'ObjectName', 'Caption']:
            if field in extmetadata:
                value = extmetadata[field].get('value', '')
                if isinstance(value, str) and value.strip():
                    # Clean HTML tags
                    clean_value = re.sub(r'<[^>]+>', '', value)
                    return clean_value[:200]
        return ""

    def get_one_representative_image(self, city_name: str, refresh: bool = False) -> Optional[Dict]:
        """
        Get ONE best representative image using a cascaded search strategy.
        Prioritizes Panoramas -> Skylines -> Landmarks -> General Views.
        """
        cache_key = f"one_image_v2:{city_name}" # v2 cache key for new logic
        
        if not refresh:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        # Cascaded search strategy
        search_strategies = [
            f"Panorama view {city_name}",
            f"{city_name} skyline",
            f"{city_name} city view",
            f"{city_name} landmark",
            f"{city_name} architecture",
            f"{city_name} tourism"
        ]
        
        best_candidate = None
        
        for query in search_strategies:
            logger.info(f"Searching image for {city_name} with query: '{query}'")
            candidates = self.fetch_from_wikimedia(query, limit=5, refresh=refresh)
            
            if not candidates:
                continue
                
            # Filter candidates for "Main Image" worthiness
            # Must be landscape (width > height)
            landscape_candidates = [
                img for img in candidates 
                if img['width'] > img['height'] * 1.2 # At least slightly wide
            ]
            
            if landscape_candidates:
                # Pick the highest score
                top_pick = max(landscape_candidates, key=lambda x: x['quality_score'])
                
                # If we found a panorama/skyline (high priority), return immediately
                if "panorama" in query.lower() or "skyline" in query.lower():
                    if top_pick['quality_score'] > 70: # Ensure it's actually good
                        cache.set(cache_key, top_pick, config.CACHE_TTL_IMAGES)
                        return top_pick
                
                # Otherwise keep as candidate and try next strategy to see if we get something better
                if best_candidate is None or top_pick['quality_score'] > best_candidate['quality_score']:
                    best_candidate = top_pick
        
        # If we went through all strategies, return the best we found
        if best_candidate:
            cache.set(cache_key, best_candidate, config.CACHE_TTL_IMAGES)
            return best_candidate
            
        # Last resort: just search the city name
        fallback = self.fetch_from_wikimedia(city_name, limit=5, refresh=refresh)
        if fallback:
            best = max(fallback, key=lambda x: x['quality_score'])
            cache.set(cache_key, best, config.CACHE_TTL_IMAGES)
            return best
            
        return None

    def fetch_landmark_image(self, landmark_name: str, refresh: bool = False) -> Optional[Dict]:
        """Fetch image for a specific landmark"""
        return self.get_one_representative_image(landmark_name, refresh=refresh)

    def get_images_for_city(self, city_name: str, limit: int = None, refresh: bool = False) -> List[Dict]:
        """Get multiple images for gallery with fallback strategies"""
        limit = limit or config.MAX_IMAGES_PER_REQUEST
        
        # Strategy 1: "City tourism"
        images = self.fetch_from_wikimedia(f"{city_name} tourism", limit, refresh=refresh)
        if images and len(images) >= limit // 2:
            return images
            
        # Strategy 2: "City landmarks"
        more_images = self.fetch_from_wikimedia(f"{city_name} landmarks", limit, refresh=refresh)
        if more_images:
            images.extend(more_images)
            
        # Strategy 3: Just "City" (if we still need more)
        if len(images) < limit // 2:
            basic_images = self.fetch_from_wikimedia(city_name, limit, refresh=refresh)
            images.extend(basic_images)
            
        # Deduplicate by URL
        unique_images = []
        seen_urls = set()
        for img in images:
            if img['url'] not in seen_urls:
                unique_images.append(img)
                seen_urls.add(img['url'])
        
        # Sort by quality
        unique_images.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return unique_images[:limit]

image_fetcher = IntelligentImageFetcher()
