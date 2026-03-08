import wikipediaapi
import re

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='CityExplorer/3.0 (https://traveltto.com)',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

page = wiki.page("Rabat")
links = set(page.links.keys())
print(f"Total links: {len(links)}")

text = """
The Kasbah of the Udayas (also spelled "Kasbah of the Oudaias") is the oldest part of the present-day city.
It was built by the Almohads in the 12th century. The Hassan Tower is another famous landmark.
"""

# Regex for capitalized phrases (allowing for 'of', 'the', 'de' in between)
pattern = r'\b([A-Z][a-zA-Z0-9\']*(?:\s+(?:of|the|de|and|&|[A-Z][a-zA-Z0-9\']*)\b)*)'

candidates = re.findall(pattern, text)
print(f"Candidates: {candidates}")

valid_landmarks = []
for c in candidates:
    c_clean = c.strip()
    if c_clean in links:
        valid_landmarks.append(c_clean)
    else:
        # Try removing "The " prefix
        if c_clean.startswith("The ") and c_clean[4:] in links:
             valid_landmarks.append(c_clean[4:])

print(f"Matched Landmarks: {valid_landmarks}")
