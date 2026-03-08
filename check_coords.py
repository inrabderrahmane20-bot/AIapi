import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def check_coords(city_name):
    print(f"Fetching {city_name}...")
    try:
        r = requests.get(f"{BASE_URL}/api/cities/{city_name}")
        response = r.json()
        
        if 'data' in response and 'coordinates' in response['data']:
            coords = response['data']['coordinates']
            print(f"Coords: {coords}")
        else:
            print(f"No coords found! Response keys: {response.keys()}")
            if 'data' in response:
                print(f"Data keys: {response['data'].keys()}")
    except Exception as e:
        print(f"Error: {e}")

check_coords("Rabat")
check_coords("Al Hoceima")
