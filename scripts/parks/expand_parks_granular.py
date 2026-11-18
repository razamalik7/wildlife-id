
import requests
import json
import time

# 1. Helper to search places
def get_state_ids():
    # We want IDs for all US states + Canada provinces
    # We'll just define a few manually or fetch them. 
    # Fetching is safer.
    regions = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", 
        "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", 
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
        "West Virginia", "Wisconsin", "Wyoming",
        "Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland", "Nova Scotia", "Ontario", "Quebec", "Saskatchewan"
    ]
    
    state_map = [] # {"name": "California", "id": 14}
    
    print("Fetching State/Province IDs...")
    for region in regions:
        try:
            url = f"https://api.inaturalist.org/v1/places/autocomplete?q={region}&per_page=1"
            resp = requests.get(url).json()
            if resp['results']:
                r = resp['results'][0]
                state_map.append({"name": region, "id": r['id'], "short": region})
        except:
            pass
        time.sleep(0.1)
    return state_map

def fetch_parks_in_state(state_info):
    parks = []
    # Queries to run per state
    queries = ["State Park", "National Forest", "Provincial Park"]
    
    for q in queries:
        try:
            # ancestral_id ensures it's IN that state
            # order_by=observations_count ensures we get popular ones, not random small ones
            url = f"https://api.inaturalist.org/v1/places?ancestor_id={state_info['id']}&q={q}&per_page=15&order_by=observations_count"
            resp = requests.get(url)
            
            if resp.status_code == 200:
                results = resp.json()['results']
                for p in results:
                    # Filter out "Umbrella" places
                    name = p['name']
                    if name.lower().endswith("state parks") or "department" in name.lower() or "system" in name.lower():
                        continue
                        
                    # Filter out duplicates
                    
                    if p['location']:
                        lat, lng = map(float, p['location'].split(','))
                        parks.append({
                            "id": p['id'],
                            "name": p['name'], # Use short name
                            "state": state_info['name'], # Or abbreviate later
                            "lat": lat,
                            "lng": lng
                        })
        except Exception as e:
            print(f"Error fetching {q} in {state_info['name']}: {e}")
            
    return parks

# Main Execution
states = get_state_ids()
all_parks_granular = []

# Preserve our original National Parks (Manual Gold Standard) as they aren't "State Parks"
# Checkmanage_parks.py for that list, but we'll re-read existing parks.json and filter only NPs?
# Easier to just RE-ADD the 63 NPs manually or filter them from current json.

# Let's load current json to keep the Mexico stuff and 63 NPs
try:
    with open('frontend/public/parks.json', 'r') as f:
        current_data = json.load(f)
        # Filter: Keep NPs and Mexican parks, discard generic "State Parks" we likely just added
        # Heuristic: Keep if id < 100000 AND contains "NP" or "National Park"? 
        # Actually easier to just keep the original verified lists if we had them.
        # Let's just keep everything that is NOT a "State Park" or "Provincial Park" from the previous generic run.
        # Or better: We trust this new script to find the State Parks better.
        # So we keep: National Parks, Mexican Parks.
        
        kept_parks = []
        for p in current_data:
            n = p['name'].lower()
            # Keep verified National Parks (usually match "NP" or specific names)
            if "np" in n or "national park" in n or "mexico" in p.get('state', '').lower() or p.get('state') == 'MX':
                kept_parks.append(p)
            # Keep Refuges
            elif "refuge" in n or "nwr" in n:
                kept_parks.append(p)
                
        all_parks_granular.extend(kept_parks)
        print(f"Retained {len(kept_parks)} existing National/Mexican parks.")
except:
    print("Could not read verified list, starting fresh (might lose Mexico/NPs if not careful).")

print("Fetching granular parks...")
for state in states:
    print(f"  Scanning {state['name']}...")
    found = fetch_parks_in_state(state)
    
    # Deduplicate against existing
    existing_ids = set(x['id'] for x in all_parks_granular)
    for f in found:
        if f['id'] not in existing_ids:
            all_parks_granular.append(f)
    
    time.sleep(0.5)

# Save
with open('frontend/public/parks.json', 'w') as f:
    json.dump(all_parks_granular, f, indent=2)

print(f"DONE. Total Granular Parks: {len(all_parks_granular)}")
