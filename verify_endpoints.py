import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"
HEADERS = {"Origin": "https://www.traveltto.com"}

def test_cities_endpoint():
    print("\n--- Testing /api/cities ---")
    try:
        start_time = time.time()
        # Test default batch size (should be 100 now)
        response = requests.get(f"{BASE_URL}/api/cities", headers=HEADERS)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            cities = data.get('data', [])
            has_more = data.get('pagination', {}).get('next_page') is not None
            total = data.get('pagination', {}).get('total')
            
            print(f"✅ Status 200 OK ({elapsed:.2f}s)")
            print(f"📊 Received {len(cities)} cities (Batch Size Check: Expected 100)")
            print(f"📊 Total cities available: {total}")
            print(f"📊 Has more: {has_more}")
            
            if len(cities) == 100:
                print("✅ Batch size verified as 100")
            else:
                print(f"⚠️ Unexpected batch size: {len(cities)}")
                
            # Check for duplicate cities in response
            seen = set()
            dups = []
            for c in cities:
                if c['name'] in seen:
                    dups.append(c['name'])
                seen.add(c['name'])
            
            if dups:
                print(f"❌ Duplicate cities in response: {dups}")
            else:
                print("✅ No duplicates in response batch")
                
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_single_city_refresh():
    print("\n--- Testing /api/cities/<city_name> (Refresh Logic) ---")
    # Test a smaller city to ensure landmarks/images work for less famous places too
    cities_to_test = ["Rabat", "Al Hoceima"]
    
    for city_name in cities_to_test:
        print(f"\nTesting City: {city_name}")
        try:
            # First call (Normal)
            print(f"Fetching {city_name}...")
            r1 = requests.get(f"{BASE_URL}/api/cities/{city_name}", headers=HEADERS)
            if r1.status_code == 200:
                d1 = r1.json()
                data = d1.get('data', {})
                gallery = data.get('gallery', [])
                print(f"✅ First fetch success. Gallery Size: {len(gallery)}")
                
                # Second call (Simulating refresh)
                print(f"Fetching {city_name} again with refresh=true...")
                start_time = time.time()
                r2 = requests.get(f"{BASE_URL}/api/cities/{city_name}?refresh=true", headers=HEADERS)
                elapsed = time.time() - start_time
                
                if r2.status_code == 200:
                    d2 = r2.json()
                    data2 = d2.get('data', {})
                    gallery2 = data2.get('gallery', [])
                    print(f"✅ Second fetch success ({elapsed:.2f}s). Gallery Size: {len(gallery2)}")
                    print(f"   Details Loaded: {data2.get('details_loaded')}")
                    if data2.get('details_loaded') is None:
                        print(f"   Response Keys: {list(data2.keys())}")
                    if data2.get('error'):
                        print(f"   Error: {data2.get('error')}")
                    
                    # Check landmarks
                    landmarks = data2.get('landmarks', [])
                    print(f"✅ Landmarks found: {len(landmarks)}")
                    for l in landmarks:
                        img_status = "📸" if l.get('image') else "⚪"
                        print(f"   - {img_status} {l['name']}")
                        
                    # Check main image
                    main_image = data2.get('image', {})
                    if main_image:
                        print(f"✅ Main Image: {main_image.get('title')}")
                        print(f"   Size: {main_image.get('width')}x{main_image.get('height')}")
                        print(f"   URL: {main_image.get('url')}")
                    else:
                        print("❌ No main image found")
                else:
                    print(f"❌ Second fetch failed: {r2.status_code}")
            else:
                print(f"❌ First fetch failed: {r1.status_code}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cities_endpoint()
    test_single_city_refresh()
