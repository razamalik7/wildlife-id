
import requests
import json

def get_headers():
    return {
        "User-Agent": "WildlifeID-App/1.0 (contact@example.com)",
        "Accept": "application/json"
    }

# 1. Test finding a state ID
state = "California"
print(f"Fetching ID for {state}...")
url = f"https://api.inaturalist.org/v1/places/autocomplete?q={state}&per_page=1"
resp = requests.get(url, headers=get_headers())
print("State Response Status:", resp.status_code)

data = resp.json()
if data.get('results'):
    state_id = data['results'][0]['id']
    print(f"State ID: {state_id}")
    
    # 2. Test fetching parks in that state
    q = "State Park"
    print(f"Fetching '{q}' in {state} (ancestor_id={state_id})...")
    url = f"https://api.inaturalist.org/v1/places?ancestor_id={state_id}&q={q}&per_page=5&order_by=observations_count"
    print("URL:", url)
    
    resp = requests.get(url, headers=get_headers())
    print("Parks Response Status:", resp.status_code)
    
    park_data = resp.json()
    print(f"Parks Found: {len(park_data.get('results', []))}")
    
    forp = park_data.get('results', [])
    for p in forp:
        print(f" - {p['display_name']} (ID: {p['id']})")
else:
    print("Failed to find state ID")
