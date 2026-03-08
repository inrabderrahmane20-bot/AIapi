import wikipediaapi

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
        print("--- Historic monuments ---")
        print(subs.text)
    else:
        print("Historic monuments not found in Culture")
else:
    print("Culture section not found")
