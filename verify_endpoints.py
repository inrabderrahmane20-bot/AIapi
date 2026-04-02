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

            coords_present = 0
            for c in cities:
                coords = c.get("coordinates")
                if coords and coords.get("lat") is not None and coords.get("lon") is not None:
                    coords_present += 1
            print(f"📍 Coordinates present: {coords_present}/{len(cities)}")
            
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
    print("\n--- Testing /api/cities/<city_name> (Worldwide Sample + Refresh Budget) ---")
    try:
        non_morocco = []
        page = 1
        while len(non_morocco) < 25 and page <= 8:
            r = requests.get(f"{BASE_URL}/api/cities?page={page}", headers=HEADERS)
            if r.status_code != 200:
                print(f"❌ Failed to fetch /api/cities?page={page}: {r.status_code}")
                return
            data = r.json()
            cities = data.get("data", [])
            for c in cities:
                if c.get("country") == "Morocco":
                    continue
                if not c.get("name"):
                    continue
                non_morocco.append(c)
            page += 1

        if not non_morocco:
            print("❌ Could not find non-Morocco cities in first pages")
            return

        sample = []
        step = max(1, len(non_morocco) // 20)
        for i in range(0, len(non_morocco), step):
            sample.append(non_morocco[i].get("name"))
            if len(sample) >= 20:
                break
        sample = [s for s in sample if s]

        print(f"🌍 Sample (non-Morocco, {len(sample)} cities): {', '.join(sample)}")

        refresh_indices = {0, len(sample)//2, len(sample)-1}

        for idx, city_name in enumerate(sample):
            print(f"\nCity: {city_name}")

            t0 = time.time()
            r1 = requests.get(f"{BASE_URL}/api/cities/{city_name}", headers=HEADERS)
            t1 = time.time() - t0
            if r1.status_code != 200:
                print(f"❌ First fetch failed: {r1.status_code}")
                continue

            d1 = r1.json().get("data", {})
            coords = d1.get("coordinates")
            print(f"✅ Normal fetch: {t1:.2f}s | coords={'yes' if coords else 'no'} | landmarks={len(d1.get('landmarks', []))} | gallery={len(d1.get('gallery', []))}")

            if idx in refresh_indices:
                t0 = time.time()
                r2 = requests.get(f"{BASE_URL}/api/cities/{city_name}?refresh=true", headers=HEADERS)
                t2 = time.time() - t0
                if r2.status_code != 200:
                    print(f"❌ Refresh fetch failed: {r2.status_code}")
                    continue
                d2 = r2.json().get("data", {})
                coords2 = d2.get("coordinates")
                print(f"✅ Refresh fetch: {t2:.2f}s | coords={'yes' if coords2 else 'no'} | landmarks={len(d2.get('landmarks', []))} | gallery={len(d2.get('gallery', []))}")

                main_image = d2.get("image") or {}
                if main_image:
                    print(f"🖼️  Main image: {main_image.get('title')} ({main_image.get('width')}x{main_image.get('height')})")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cities_endpoint()
    test_single_city_refresh()
