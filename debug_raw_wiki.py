import requests

def get_wikitext(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": title
    }
    headers = {
        "User-Agent": "CityExplorer/3.0 (https://traveltto.com)"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["revisions"][0]["slots"]["main"]["*"]

text = get_wikitext("Rabat")
print(f"Has brackets: {'[[' in text}")
print(f"Sample text: {text[:500]}")
