
import requests
import json

taxon_id = 41638 # American Black Bear

parks_to_check = [
    {"name": "Great Smoky Mountains NP", "id": 72645},
    {"name": "Yellowstone NP", "id": 10211},
    {"name": "Denali NP", "id": 71077},
    {"name": "Katmai NP", "id": 95257},
    {"name": "Yosemite NP", "id": 68542}
]

canadian_parks = [
    "Banff National Park",
    "Jasper National Park", 
    "Algonquin Provincial Park",
    "Yoho National Park",
    "Kootenay National Park",
    "Waterton Lakes National Park",
    "Pacific Rim National Park Reserve"
]

print("--- OBSERVATION COUNT ANALYSIS (Black Bear) ---")
print(f"{'Park':<30} | {'Research':<10} | {'Any Quality':<10}")
print("-" * 60)

for park in parks_to_check:
    try:
        # Research Grade
        url_r = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&place_id={park['id']}&quality_grade=research&per_page=0"
        count_r = requests.get(url_r).json()['total_results']
        
        # Any Quality
        url_a = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&place_id={park['id']}&per_page=0"
        count_a = requests.get(url_a).json()['total_results']
        
        print(f"{park['name']:<30} | {count_r:<10} | {count_a:<10}")
    except Exception as e:
        print(f"{park['name']:<30} | Error: {e}")

print("\n--- FETCHING CANADIAN PARK IDS ---")
found_canada = []
for name in canadian_parks:
    try:
        url = f"https://api.inaturalist.org/v1/places/autocomplete?q={name}&per_page=1"
        resp = requests.get(url).json()
        if resp['results']:
            p = resp['results'][0]
            # Verify it's actually in Canada (place type 12=Country, check ancestors or display name)
            print(f"Found {name}: ID {p['id']} - {p['display_name']}")
            found_canada.append({
                "name": name,
                "id": p['id'],
                "lat": float(p['location'].split(',')[0]),
                "lng": float(p['location'].split(',')[1]),
                "state": "Canada" # General label
            })
    except Exception as e:
        print(f"Error finding {name}: {e}")

with open('canadian_parks.json', 'w') as f:
    json.dump(found_canada, f, indent=2)
