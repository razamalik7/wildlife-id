"""Check which park IDs are invalid on iNaturalist"""
import json
import requests
import time

parks_path = "../frontend/public/parks.json"
with open(parks_path, "r") as f:
    parks = json.load(f)

print(f"Checking {len(parks)} parks against iNaturalist...\n")

valid = []
invalid = []

for i, park in enumerate(parks):
    try:
        r = requests.get(f"https://api.inaturalist.org/v1/places/{park['id']}", timeout=10)
        if r.status_code == 200:
            valid.append(park)
            print(f"✓ {park['name']} ({park['id']})")
        elif r.status_code == 422:
            invalid.append({"park": park, "error": "422 - Invalid ID"})
            print(f"❌ {park['name']} ({park['id']}) - 422 INVALID")
        elif r.status_code == 404:
            invalid.append({"park": park, "error": "404 - Not found"})
            print(f"❌ {park['name']} ({park['id']}) - 404 NOT FOUND")
        elif r.status_code == 429:
            print(f"⏳ Rate limited at park {i+1}, stopping...")
            break
        else:
            print(f"? {park['name']} ({park['id']}) - HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ {park['name']} ({park['id']}) - Error: {e}")
    
    time.sleep(0.5)  # Be gentle

print(f"\n{'='*50}")
print(f"Valid: {len(valid)}")
print(f"Invalid: {len(invalid)}")

if invalid:
    print("\nINVALID PARKS:")
    for item in invalid:
        print(f"  - {item['park']['name']} (ID: {item['park']['id']}) - {item['error']}")
