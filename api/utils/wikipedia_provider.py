import re
import wikipediaapi
from typing import List, Dict, Tuple, Optional, Set
from api.config import config
from api.utils.logger import logger
from api.utils.cache import cache
from api.utils.image_fetcher import image_fetcher

class WikipediaDataProvider:
    def __init__(self):
        # We use WIKI format to get raw wikitext with [[links]]
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='CityExplorer/3.0 (https://traveltto.com)',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        
        # STOPWORDS & GENERICS (Centralized for easy updating)
        self.stopwords = {
            'The', 'A', 'An', 'Many', 'Although', 'However', 'But', 'In', 'On', 'At', 
            'This', 'It', 'There', 'Some', 'Most', 'Popular', 'Famous', 'Notable',
            'Another', 'Other', 'These', 'Those', 'Which', 'Who', 'When', 'Where', 
            'While', 'Despite', 'He', 'She', 'They', 'We', 'You', 'For', 'With', 'I',
            'One', 'Two', 'Three', 'First', 'Second', 'Third', 'Fourth', 'Fifth',
            'Main', 'Major', 'Local', 'Great', 'Small', 'Big', 'Old', 'New', 'High', 'Low',
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
            'September', 'October', 'November', 'December',
            'BC', 'AD', 'BCE', 'CE', 'During', 'After', 'Before', 'Since', 'Until',
            'Early', 'Late', 'Mid', 'Modern', 'Ancient', 'Medieval', 'Pre', 'Post',
            'North', 'South', 'East', 'West', 'Central', 'Upper', 'Lower',
            'Situated', 'Located', 'Found', 'Built', 'Established', 'Founded',
            'Includes', 'Including', 'Such', 'As', 'Like', 'Especially', 'Particularly'
        }
        
        self.historical_terms = {
            'Phoenicians', 'Romans', 'Byzantines', 'Visigoths', 'Vandals', 'Carthaginians',
            'Umayyads', 'Almoravids', 'Almohads', 'Marinids', 'Wattasids', 'Saadians', 
            'Alaouites', 'Idrisids', 'Ottomans', 'French', 'Spanish', 'Portuguese', 
            'British', 'Arabs', 'Berbers', 'Moors', 'Jews', 'Christians', 'Muslims',
            'Islam', 'Christianity', 'Judaism', 'Catholic', 'Protestant', 'Orthodox',
            'Sunni', 'Shia', 'Sufi', 'Salafist',
            'Zenata', 'Dynasty', 'Empire', 'Kingdom', 'Republic', 'Protectorate',
            'Colony', 'Civilization', 'Culture', 'Era', 'Period', 'Age', 'War', 'Battle',
            'Treaty', 'Conference', 'Pact', 'Agreement', 'Declaration', 'Revolution',
            'Rebellion', 'Uprising', 'Conquest', 'Invasion', 'Occupation', 'Siege',
            'History', 'Background', 'Overview', 'Introduction', 'Conclusion',
            'Geography', 'Demographics', 'Economy', 'Politics', 'Government',
            'Century', 'Millennium', 'Decade', 'Year', 'Month', 'Day', 'Date',
            'Population', 'Inhabitants', 'Residents', 'Citizens', 'People',
            'Language', 'Dialect', 'Religion', 'Belief', 'Faith', 'Tradition'
        }
        
        self.generics = {
            'City', 'Town', 'Village', 'Capital', 'Center', 'Centre', 'Area', 'Region', 
            'Location', 'Place', 'Site', 'View', 
            'University', 'Airport', 'Station', 'Mosque', 'Church', 'Cathedral', 
            'Museum', 'Park', 'Square', 'Street', 'Avenue', 'Boulevard', 'Building',
            'Tower', 'Bridge', 'Stadium', 'Palace', 'Castle', 'Temple', 'Monument',
            'Orient', 'Occident', 'Archaeological', 'Ancient', 'Medieval', 'Modern', 
            'Traditional', 'Cultural', 'Historic', 'Historical', 'National', 'International',
            'Public', 'Private', 'Royal', 'Imperial', 'Colonial', 'Urban', 'Rural',
            'Moroccan', 'Spanish', 'French', 'Roman', 'Phoenician', 'Arab', 'Berber', 
            'African', 'European', 'American', 'Asian', 'World', 'Heritage', 'UNESCO',
            'Art', 'Nouveau', 'Deco', 'Style', 'Architecture', 'Design',
            'Art Nouveau', 'Art Deco', 'World Heritage Site', 'National Park',
            'Government', 'Municipality', 'Province', 'District', 'Department',
            'Byzantine', 'Visigothic', 'Islamic', 'Moorish', 'Christian', 'Jewish',
            'Ottoman', 'British', 'Portuguese', 'Carthaginian', 'Vandal', 'Almohad', 
            'Almoravid', 'Marinid', 'Saadian', 'Alaouite', 'Idrisid', 'Atlantic',
            'Morocco', 'Spain', 'France', 'Portugal', 'Algeria', 'Tunisia', 'Mauritania',
            'Rabat', 'Casablanca', 'Fes', 'Marrakech', 'Tangier', 'Agadir', 'Meknes',
            'Oujda', 'Kenitra', 'Tetouan', 'Safi', 'Mohammedia', 'Khouribga', 'El Jadida',
            'Arabic', 'English', 'French', 'Spanish', 'Berber', 'Amazigh', 'Romanized',
            'Census', 'Population', 'Inhabitants', 'Capital'
        }
        
        self.people_titles = {
            'King', 'Queen', 'Prince', 'Princess', 'Sultan', 'Emperor', 'Empress',
            'President', 'Prime Minister', 'General', 'Admiral', 'Saint', 'Pope',
            'Bishop', 'Cardinal', 'Sir', 'Lord', 'Lady', 'Duke', 'Duchess', 'Count',
            'Countess', 'Baron', 'Baroness', 'Dr', 'Mr', 'Mrs', 'Ms', 'Prof',
            'Governor', 'Mayor', 'Senator', 'Representative', 'Ambassador',
            'Ibn', 'Ben', 'Bin', 'Abd', 'Abdel', 'Abu', 'Tashfin', 'Yusuf', 'Moulay',
            'Ali', 'Mansur', 'Hassan', 'Mohammed'
        }

    def get_city_coordinates(self, city_name: str, refresh: bool = False) -> Optional[Dict]:
        """Get coordinates from Wikipedia API"""
        cache_key = f"wiki_coords:{city_name}"
        
        if not refresh:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        try:
            params = {
                "action": "query",
                "prop": "coordinates",
                "titles": city_name,
                "format": "json"
            }
            
            # Use request_handler directly for consistent caching/user-agent
            from api.utils.request_handler import request_handler
            data = request_handler.get_json_cached(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                cache_key=f"wiki_api_coords:{city_name}",
                ttl=config.CACHE_TTL_COORDS,
                refresh=refresh
            )
            
            if data:
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if "coordinates" in page:
                        coords = page["coordinates"][0]
                        result = {
                            "lat": coords["lat"],
                            "lon": coords["lon"]
                        }
                        cache.set(cache_key, result, config.CACHE_TTL_COORDS)
                        return result
                        
        except Exception as e:
            logger.debug(f"Wikipedia coordinates fetch failed for {city_name}: {e}")
            
        return None

    def get_city_summary(self, city_name: str, refresh: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Get short summary for a city"""
        cache_key = f"wiki_summary:{city_name}"
        
        if not refresh:
            cached = cache.get(cache_key)
            if cached:
                return cached.get('summary'), cached.get('title')
        
        try:
            page = self.wiki.page(city_name)
            
            if page.exists():
                summary = page.summary or ""
                if summary:
                    # Clean wikitext markup from summary if present
                    summary = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", summary)
                    summary = re.sub(r"'''|''", "", summary)
                    
                    # Get first 2-3 sentences max
                    sentences = re.split(r'(?<=[.!?])\s+', summary)
                    short_summary = ""
                    sentence_count = 0
                    for sentence in sentences:
                        if sentence.strip():
                            short_summary += sentence.strip() + ' '
                            sentence_count += 1
                            if sentence_count >= 3:
                                break
                    
                    short_summary = short_summary.strip()
                    if short_summary:
                        cache.set(cache_key, {
                            'summary': short_summary,
                            'title': page.title
                        }, config.CACHE_TTL)
                        return short_summary, page.title
        except Exception as e:
            logger.debug(f"Wikipedia summary failed for {city_name}: {e}")
        
        return None, None
    
    def extract_landmarks(self, city_name: str, refresh: bool = False) -> List[Dict]:
        """
        Extract landmarks using advanced wikitext parsing.
        """
        print(f"DEBUG: extract_landmarks called for {city_name}, refresh={refresh}", flush=True)
        cache_key = f"landmarks_v2:{city_name}"
        
        if not refresh:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        landmarks = []
        seen_landmarks = set()
        
        try:
            page = self.wiki.page(city_name)
            
            if page.exists():
                
                # Priority sections contain the best data
                priority_keywords = ['landmark', 'attraction', 'sight', 'monument', 'tourism', 'place of interest']
                
                # Get all links on the page for cross-referencing
                page_links = set(page.links.keys()) if page.links else set()
                logger.info(f"DEBUG: Found {len(page_links)} page links for {city_name}")

                def process_section(section, parent_context="", is_priority_section=False):
                    section_title = section.title.lower()
                    full_context = f"{parent_context} {section_title}".strip()
                    
                    # Determine if this is a priority section (if not already inherited)
                    current_is_priority = is_priority_section or any(k in section_title for k in priority_keywords)
                    
                    if current_is_priority:
                        logger.info(f"DEBUG: Processing priority section: {full_context}")

                    # Exclude irrelevant sections entirely
                    excluded_sections = [
                        'people', 'famous people', 'notable people', 'residents', 'alumni', 'births', 'deaths',
                        'demographics', 'climate', 'weather', 'geography', 'topography', 'economy', 'transport',
                        'transportation', 'education', 'sports', 'sister cities', 'twin towns', 'administration',
                        'government', 'politics', 'media', 'references', 'external links', 'see also', 'bibliography',
                        'notes', 'gallery', 'citations', 'sources', 'etymology', 'history' 
                        # Note: We exclude 'history' from general crawl, only strictly specific landmarks allowed if absolutely necessary
                    ]
                    
                    if any(ex in full_context for ex in excluded_sections) and not current_is_priority:
                        return

                    if section.text:
                        lines = section.text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            
                            # Check for list items (starting with *)
                            is_list_item = line.startswith('*')
                            
                            # Parse WikiLinks [[Target|Label]] or [[Target]]
                            links = re.findall(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", line)
                            
                            # Strategy 1: High confidence - List item with a link in a priority section
                            if current_is_priority and is_list_item and links:
                                for link in links:
                                    self._add_landmark(link, line, city_name, landmarks, seen_landmarks, high_confidence=True)
                                continue
                                
                            # Strategy 2: List item with a link in a relevant context (e.g. "Culture")
                            if is_list_item and links and self._is_relevant_context(full_context):
                                for link in links:
                                    self._add_landmark(link, line, city_name, landmarks, seen_landmarks)
                                continue

                            # Strategy 3: Heuristic extraction (fallback)
                            # Only if we are in a priority section but no links found (rare)
                            if current_is_priority and is_list_item:
                                clean_line = re.sub(r'^\*\s*', '', line) # Remove bullet
                                candidate = self._clean_name(clean_line.split(',')[0].split('.')[0]) # Take first part
                                if self._is_valid_landmark_name(candidate, city_name):
                                    self._add_landmark(candidate, line, city_name, landmarks, seen_landmarks)
                                continue

                            # Strategy 5: Prose Cross-Reference (Advanced)
                            # If not a list item, but we are in a priority section, use Page Links to find landmarks
                            if current_is_priority and not is_list_item and page_links:
                                # Regex to find capitalized phrases (e.g. "Kasbah of the Udayas", "Hassan Tower")
                                # Allows "The", "of", "de", etc. inside the phrase
                                candidates = re.findall(r'\b([A-Z][a-zA-Z0-9\']*(?:\s+(?:of|the|de|and|&|[A-Z][a-zA-Z0-9\']*)\b)*)', line)
                                # logger.info(f"DEBUG: Found {len(candidates)} candidates in line: {line[:50]}...")
                                for c in candidates:
                                    c_clean = c.strip()
                                    if len(c_clean) < 3: continue
                                    
                                    # Check exact match
                                    if c_clean in page_links:
                                         if self._is_valid_landmark_name(c_clean, city_name):
                                             logger.info(f"DEBUG: Found landmark via Strategy 5: {c_clean}")
                                             self._add_landmark(c_clean, line, city_name, landmarks, seen_landmarks)
                                    # Check match without "The" prefix
                                    elif c_clean.startswith("The ") and c_clean[4:] in page_links:
                                         if self._is_valid_landmark_name(c_clean[4:], city_name):
                                             logger.info(f"DEBUG: Found landmark via Strategy 5 (no 'The'): {c_clean[4:]}")
                                             self._add_landmark(c_clean[4:], line, city_name, landmarks, seen_landmarks)

                            # Strategy 4: Prose extraction with WikiLinks (Legacy/Complementary)
                            # If not a list item, but we are in a priority section, grab all valid WikiLinks
                            if current_is_priority and not is_list_item and links:
                                for link in links:
                                    # Use stricter validation for prose links to avoid "12th century", "Almohad", etc.
                                    if self._is_valid_landmark_name(link, city_name):
                                        self._add_landmark(link, line, city_name, landmarks, seen_landmarks)

                    # Recurse
                    for subsection in section.sections:
                        process_section(subsection, full_context, current_is_priority)

                # Process all sections
                for section in page.sections:
                    process_section(section)
                    
                # If we still have very few landmarks, try the summary as a last resort
                if len(landmarks) < 3 and page.summary:
                    self._extract_from_summary(page.summary, city_name, landmarks, seen_landmarks, page_links)
            
            # Limit to top 10
            landmarks = landmarks[:10]
            
            # Fetch images for them
            for landmark in landmarks:
                # Use the fetcher to get a nice image
                img = image_fetcher.fetch_landmark_image(landmark['name'], refresh=refresh)
                if img:
                    landmark['image'] = img
            
            cache.set(cache_key, landmarks, config.CACHE_TTL)
            
        except Exception as e:
            logger.error(f"Landmarks extraction failed for {city_name}: {e}", exc_info=True)
        
        return landmarks

    def _add_landmark(self, name: str, context: str, city_name: str, landmarks: List, seen: Set, high_confidence=False):
        """Add a landmark if valid and new"""
        clean_name = self._clean_name(name)
        
        if not clean_name: return
        if clean_name in seen: return
        
        if not self._is_valid_landmark_name(clean_name, city_name, high_confidence):
            return
            
        seen.add(clean_name)
        
        # Clean context for description
        description = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", context) # Resolve links
        description = re.sub(r"'''|''", "", description) # Remove bold/italic
        description = description.lstrip('* ').strip()
        
        landmarks.append({
            'name': clean_name,
            'description': description[:200] + '...' if len(description) > 200 else description,
            'raw_text': context
        })

    def _is_relevant_context(self, context: str) -> bool:
        keywords = ['architecture', 'culture', 'park', 'garden', 'square', 'building', 'religious', 'site']
        return any(k in context for k in keywords)

    def _clean_name(self, name: str) -> str:
        # Remove parenthetical info: "Hassan Tower (Tour Hassan)" -> "Hassan Tower"
        name = re.sub(r'\s*\(.*?\)', '', name)
        # Remove dates/years: "12th century", "1990s", "1450-1500"
        name = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s+century\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\b\d{4}s?\b', '', name)
        return name.strip(" ,.-*")

    def _is_valid_landmark_name(self, name: str, city_name: str, high_confidence: bool = False) -> bool:
        if len(name) < 3: 
            # logger.debug(f"Rejecting '{name}': too short")
            return False
        
        # Check against city name
        if city_name and city_name.lower() in name.lower():
            # Allow "Royal Palace of Rabat" but not just "Rabat"
            if name.lower() == city_name.lower(): 
                # logger.debug(f"Rejecting '{name}': matches city name")
                return False
            
        # 1. Exact match checks
        if name in self.stopwords: 
            # logger.debug(f"Rejecting '{name}': in stopwords")
            return False
        if name in self.generics: 
            # logger.debug(f"Rejecting '{name}': in generics")
            return False
        
        # 2. Token-based checks
        parts = name.split()
        first_word = parts[0]
        
        # Starts with a stopword? (e.g. "During the...", "In the...")
        # Exception: "The" is allowed if followed by valid words (e.g. "The Louvre")
        if first_word in self.stopwords and first_word != "The":
            # logger.debug(f"Rejecting '{name}': starts with stopword '{first_word}'")
            return False
            
        # Contains historical terms?
        for part in parts:
            clean_part = part.strip('.,')
            if clean_part in self.historical_terms:
                logger.debug(f"Rejecting '{name}': contains historical term '{clean_part}'")
                return False
                
        # 3. Structure checks
        # Too long? Likely a sentence fragment
        if len(parts) > 6: 
            # logger.debug(f"Rejecting '{name}': too long ({len(parts)} parts)")
            return False
        
        # Check if it starts with a person's title
        if first_word in self.people_titles and not high_confidence:
            # If starts with a title, must contain a structure type word to be a landmark
            structure_words = {
                'Palace', 'Castle', 'Tower', 'Mosque', 'Church', 'Cathedral', 'Museum', 
                'University', 'School', 'College', 'Hospital', 'Station', 'Airport', 
                'Bridge', 'Dam', 'Stadium', 'Park', 'Garden', 'Square', 'Plaza', 
                'Monument', 'Memorial', 'Mausoleum', 'Tomb', 'Cemetery', 'Fort', 
                'Fortress', 'Citadel', 'Temple', 'Shrine', 'Synagogue', 'Library',
                'Theater', 'Theatre', 'Hall', 'Center', 'Centre', 'Building', 'House'
            }
            if not any(w in name for w in structure_words):
                # logger.debug(f"Rejecting '{name}': title without structure")
                return False 
        
        return True

    def _extract_from_summary(self, summary: str, city_name: str, landmarks: List, seen: Set, page_links: Set):
        """
        Fallback extraction from summary using multiple strategies.
        """
        # Strategy 5: Prose Cross-Reference
        print(f"DEBUG: _extract_from_summary called for {city_name}", flush=True)
        
        # Strategy 6: Contextual Capitalization (For cities with no links/sections)
        # Look for sentences with "beach", "park", "museum", etc.
        context_keywords = ['beach', 'park', 'garden', 'museum', 'mosque', 'church', 'castle', 'fort', 
                           'palace', 'square', 'mountain', 'resort', 'promenade', 'harbour', 'port']
        
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        for sentence in sentences:
            # 1. Check for context keywords
            has_context = any(k in sentence.lower() for k in context_keywords)
            
            # 2. Extract Capitalized Phrases
            # Regex: Capitalized Word + (optional connector + Capitalized Word)*
            candidates = re.findall(r'\b[A-Z][a-zA-Z0-9\']+(?:\s+(?:of|the|de|and|&)\s+[A-Z][a-zA-Z0-9\']+)*(?:\s+[A-Z][a-zA-Z0-9\']+)*\b', sentence)
            
            for c in candidates:
                c_clean = c.strip(" ,.-")
                if len(c_clean) < 3: continue
                
                # Validation Logic
                is_valid = False
                
                # A. It's a known Page Link (Strongest)
                if c_clean in page_links:
                    is_valid = True
                elif c_clean.startswith("The ") and c_clean[4:] in page_links:
                    c_clean = c_clean[4:]
                    is_valid = True
                
                # B. It's in a Contextual Sentence (Heuristic)
                elif has_context:
                    # Stricter checks for non-link candidates
                    if self._is_valid_landmark_name(c_clean, city_name):
                        # Ensure it's not just a start-of-sentence word that happens to be capitalized
                        # (unless it's a multi-word phrase which is usually safe)
                        if " " in c_clean or sentence.find(c) > 0:
                             is_valid = True
                
                if is_valid and self._is_valid_landmark_name(c_clean, city_name):
                    logger.info(f"DEBUG: Found landmark via Summary ({'Link' if c_clean in page_links else 'Context'}): {c_clean}")
                    self._add_landmark(c_clean, sentence, city_name, landmarks, seen)


wikipedia_provider = WikipediaDataProvider()
