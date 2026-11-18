
import requests
import json
import time

# List of all 63 US National Parks (names as they appear in the app)
national_parks = [
    "Acadia National Park", "National Park of American Samoa", "Arches National Park", 
    "Badlands National Park", "Big Bend National Park", "Biscayne National Park", 
    "Black Canyon of the Gunnison National Park", "Bryce Canyon National Park", 
    "Canyonlands National Park", "Capitol Reef National Park", "Carlsbad Caverns National Park", 
    "Channel Islands National Park", "Congaree National Park", "Crater Lake National Park", 
    "Cuyahoga Valley National Park", "Death Valley National Park", "Denali National Park", 
    "Dry Tortugas National Park", "Everglades National Park", "Gates of the Arctic National Park", 
    "Gateway Arch National Park", "Glacier National Park", "Glacier Bay National Park", 
    "Grand Canyon National Park", "Grand Teton National Park", "Great Basin National Park", 
    "Great Sand Dunes National Park", "Great Smoky Mountains National Park", 
    "Guadalupe Mountains National Park", "Haleakala National Park", "Hawaii Volcanoes National Park", 
    "Hot Springs National Park", "Indiana Dunes National Park", "Isle Royale National Park", 
    "Joshua Tree National Park", "Katmai National Park", "Kenai Fjords National Park", 
    "Kings Canyon National Park", "Kobuk Valley National Park", "Lake Clark National Park", 
    "Lassen Volcanic National Park", "Mammoth Cave National Park", "Mesa Verde National Park", 
    "Mount Rainier National Park", "New River Gorge National Park", "North Cascades National Park", 
    "Olympic National Park", "Petrified Forest National Park", "Pinnacles National Park", 
    "Redwood National Park", "Rocky Mountain National Park", "Saguaro National Park", 
    "Sequoia National Park", "Shenandoah National Park", "Theodore Roosevelt National Park", 
    "Virgin Islands National Park", "Voyageurs National Park", "White Sands National Park", 
    "Wind Cave National Park", "Wrangell-St Elias National Park", "Yellowstone National Park", 
    "Yosemite National Park", "Zion National Park"
]

results = []

def search_place(query):
    try:
        # Search generally first
        url = "https://api.inaturalist.org/v1/places/autocomplete"
        params = {'q': query, 'per_page': 5}
        resp = requests.get(url, params=params)
        data = resp.json()
        
        # Heuristic: Prefer "National Park" in the name and Place Type 100 (Park) or 8 (Place) or 10 (Protected Area)
        # But autocomplete usually sorts by relevance. We'll take the first good match.
        
        if data['results']:
            # Try to find exact match first
            for place in data['results']:
                if query.lower() in place['name'].lower():
                    return place
            return data['results'][0]
    except Exception as e:
        print(f"Error for {query}: {e}")
    return None

print("Validating National Park IDs...")

for park_name in national_parks:
    place = search_place(park_name)
    
    if place:
        # Map short name for the list
        short_name = park_name.replace("National Park", "NP").replace("Preserve", "").strip()
        state = place['display_name'].split(", ")[-2] if ", " in place['display_name'] else ""
        if len(state) > 2: state = "" # Reset if not a state code
        
        entry = {
            "name": short_name,
            "id": place['id'],
            "state": state,
            "found_name": place['display_name'],
            "lat": float(place['location'].split(",")[0]) if place['location'] else 0,
            "lng": float(place['location'].split(",")[1]) if place['location'] else 0
        }
        results.append(entry)
        print(f"✅ {park_name} -> {place['id']} ({place['name']})")
    else:
        print(f"❌ Failed to find {park_name}")
    
    time.sleep(0.8)

with open('national_parks_validated.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Done. Saved to national_parks_validated.json")
