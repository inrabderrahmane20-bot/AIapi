
import requests

def get_coords(city_name):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "coordinates",
        "titles": city_name,
        "format": "json"
    }
    headers = {
        "User-Agent": "CityExplorer/3.0 (https://traveltto.com)"
    }
    r = requests.get(url, params=params, headers=headers)
    try:
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if "coordinates" in page:
                print(f"{city_name}: {page['coordinates']}")
            else:
                print(f"{city_name}: No coordinates found. Keys: {page.keys()}")
    except Exception as e:
        print(f"Error: {e}, Content: {r.text[:200]}")

get_coords("Rabat")
get_coords("Al Hoceima")
