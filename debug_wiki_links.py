import wikipediaapi
import re

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='CityExplorer/3.0 (https://traveltto.com)',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

page = wiki.page("Rabat")
s = page.section_by_title("Culture")
if s:
    subs = s.section_by_title("Historic monuments")
    if subs:
        print(f"Has brackets: {'[[' in subs.text}")
        print(f"Sample text: {subs.text[:100]}")
        links = re.findall(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", subs.text)
        print(f"Found links: {links[:5]}")
