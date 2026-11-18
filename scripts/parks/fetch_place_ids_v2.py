
import requests
import json
import time

# List of parks to find
target_parks = [
    # State Parks
    "Custer State Park", "Baxter State Park", "Adirondack Park", 
    "Harriman State Park", "Porcupine Mountains Wilderness State Park",
    "Itasca State Park", "Hocking Hills State Park", "Franconia Notch State Park",
    "Watkins Glen State Park", "Starved Rock State Park",
    
    # National Forests
    "Pisgah National Forest", "Nantahala National Forest",
    "George Washington & Jefferson National Forest", # Try combined
    "Chattahoochee-Oconee National Forests",
    "Ocala National Forest", "White Mountain National Forest",
    "Green Mountain National Forest", "Monongahela National Forest",
    "Allegheny National Forest", "Superior National Forest",
    
    # Wildlife Refuges
    "Okefenokee National Wildlife Refuge", 
    "Merritt Island National Wildlife Refuge",
    "Chincoteague National Wildlife Refuge",
    "Bear River Migratory Bird Refuge",
    "Bosque del Apache National Wildlife Refuge"
]

# Alternative names to try if first fails
variations = {
    "Harriman State Park": ["Harriman State Park, NY"],
    "George Washington & Jefferson National Forest": ["George Washington National Forest", "Jefferson National Forest"],
}

results = []

def search_place(query):
    try:
        url = "https://api.inaturalist.org/v1/places/autocomplete"
        params = {'q': query, 'per_page': 3}
        resp = requests.get(url, params=params)
        data = resp.json()
        if data['results']:
            return data['results'][0]
    except:
        return None
    return None

print("Starting ID fetch...")

for name in target_parks:
    place = search_place(name)
    
    # Try variations if not found
    if not place and name in variations:
        for var_name in variations[name]:
            print(f"Retrying with {var_name}...")
            place = search_place(var_name)
            if place: break
            time.sleep(1)

    if place:
        results.append({
            "name": name, # Keep original key
            "id": place['id'],
            "found_name": place['name'],
            "state": place['display_name'].split(", ")[-2] if ", " in place['display_name'] else "",
            "lat": float(place['location'].split(",")[0]) if place['location'] else 0,
            "lng": float(place['location'].split(",")[1]) if place['location'] else 0
        })
        print(f"✅ Found {name} -> {place['id']}")
    else:
        print(f"❌ Failed to find {name}")
    
    time.sleep(1.0) # Rate limit

# Save to file
with open('refined_park_ids.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Done. Saved to refined_park_ids.json")
