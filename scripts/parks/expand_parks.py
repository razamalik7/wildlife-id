
import requests
import json
import time

# 1. Existing verified list (Keep this as "Gold Standard")
# Load from file if it exists, otherwise use the list we just made
try:
    with open('frontend/public/parks.json', 'r') as f:
        existing_parks = json.load(f)
except:
    existing_parks = []

known_ids = set(p['id'] for p in existing_parks)
final_list = existing_parks.copy()

print(f"Starting with {len(final_list)} verified parks.")

# 2. Expansion Strategy
# We will search for keywords in specific regions to get more results
regions = [
    # US States
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", 
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", 
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
    "West Virginia", "Wisconsin", "Wyoming",
    # Canadian Provinces
    "Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland", "Nova Scotia", "Ontario", "Quebec", "Saskatchewan"
]

keywords = ["State Park", "National Forest", "Provincial Park", "National Wildlife Refuge", "Nature Preserve", "Conservation Area"]

def search_region(region, keyword):
    new_finds = []
    query = f"{keyword} {region}"
    try:
        # Increase limit to 50
        url = f"https://api.inaturalist.org/v1/places/autocomplete?q={query}&per_page=50" 
        resp = requests.get(url)
        if resp.status_code == 200:
            results = resp.json()['results']
            for place in results:
                # Less strict filtering - just ensure it's not already known
                if place['id'] not in known_ids:
                    # Attempt to parse location
                    if place.get('location'):
                        lat, lng = map(float, place['location'].split(','))
                        
                        # Simplistic state code map or just use full region name
                        state_code = region # Default
                        parts = place.get('display_name', '').split(", ")
                        if len(parts) >= 2:
                            # Try to find a 2-letter code
                            for part in parts:
                                if len(part) == 2 and part.isupper():
                                    state_code = part
                                    break
                        
                        entry = {
                            "id": place['id'],
                            "name": place['name'],
                            "state": state_code,
                            "lat": lat,
                            "lng": lng
                        }
                        new_finds.append(entry)
                        known_ids.add(place['id'])
    except Exception as e:
        print(f"Error searching {query}: {e}")
    
    return new_finds

print("Expanding Park List (this may take a minute)...")
count = 0

for region in regions:
    print(f"Scanning {region}...")
    for kw in keywords:
        found = search_region(region, kw)
        if found:
            final_list.extend(found)
            count += len(found)
            # print(f"  + {len(found)} {kw}")
    
    time.sleep(0.5) # Friendly rate limiting per region block

# 3. Save Updated List
with open('frontend/public/parks.json', 'w') as f:
    json.dump(final_list, f, indent=2)

print(f"\nFINISHED. Total Parks: {len(final_list)} (Added {count} new ones).")
