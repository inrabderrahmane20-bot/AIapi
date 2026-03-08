import wikipediaapi
import re

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='CityExplorer/3.0 (https://traveltto.com)',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

page = wiki.page("Al Hoceima")
summary = page.summary

print(f"Summary: {summary[:500]}...")

# Proposed Strategy 6: Contextual Capitalization in Summary
# Keywords indicating a list of landmarks/features
context_keywords = ['beach', 'beaches', 'park', 'parks', 'garden', 'gardens', 'museum', 'museums', 
                   'mosque', 'mosques', 'church', 'churches', 'castle', 'castles', 'fort', 'forts',
                   'palace', 'palaces', 'square', 'squares', 'mountain', 'mountains']

sentences = re.split(r'(?<=[.!?])\s+', summary)
found = []

stopwords = {'The', 'A', 'An', 'In', 'On', 'At', 'This', 'It', 'There', 'Some', 'Most', 'And', 'Of', 'With', 'To', 'From', 'As', 'By', 'For'}
generics = {'City', 'Town', 'Village', 'Capital', 'Center', 'Centre', 'Area', 'Region', 'Location', 'Place', 'Site', 'View', 'North', 'South', 'East', 'West', 'Province', 'District', 'Department', 'Morocco', 'Al Hoceima', 'Mediterranean', 'Rif', 'Census'}

for sentence in sentences:
    # Check if sentence contains relevant keywords
    if any(k in sentence.lower() for k in context_keywords):
        print(f"\nAnalyzing sentence: {sentence}")
        
        # Extract capitalized phrases (excluding beginning of sentence if it's a stopword)
        # Regex for Capitalized Phrase: [A-Z][a-z]+ (SPACE [A-Z][a-z]+)*
        candidates = re.findall(r'\b([A-Z][a-zA-Z0-9\']*(?:\s+(?:of|the|de|and|&|[A-Z][a-zA-Z0-9\']*)\b)*)', sentence)
        
        for i, c in enumerate(candidates):
            c_clean = c.strip(" ,.-")
            if len(c_clean) < 3: continue
            
            # Filter
            if c_clean in stopwords: continue
            if c_clean in generics: continue
            if c_clean.lower() == "al hoceima": continue # City name
            
            # Heuristic: If it's the first word of the sentence, it might just be capitalized because of that.
            # But here candidates are from the whole sentence.
            # If the sentence starts with this candidate, check if it's a stopword (already done).
            
            print(f"  Candidate: {c_clean}")
            found.append(c_clean)

print(f"\nFound Landmarks: {found}")
