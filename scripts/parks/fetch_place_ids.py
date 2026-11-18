
import requests
import json
import time

parks = [
    # State Parks
    "Custer State Park",
    "Baxter State Park",
    "Adirondack Park",
    "Harriman State Park",
    "Porcupine Mountains Wilderness State Park", # Full name for better match
    "Itasca State Park",
    "Hocking Hills State Park",
    "Franconia Notch State Park",
    "Watkins Glen State Park",
    "Starved Rock State Park",
    
    # National Forests
    "Pisgah National Forest",
    "Nantahala National Forest",
    "George Washington and Jefferson National Forests", # Combined often
    "Chattahoochee-Oconee National Forests", # Official name
    "Ocala National Forest",
    "White Mountain National Forest",
    "Green Mountain National Forest",
    "Monongahela National Forest",
    "Allegheny National Forest",
    "Superior National Forest",
    
    # Wildlife Refuges
    "Okefenokee National Wildlife Refuge",
    "Merritt Island National Wildlife Refuge",
    "Chincoteague National Wildlife Refuge",
    "Bear River Migratory Bird Refuge",
    "Bosque del Apache National Wildlife Refuge"
]

results = {}

print("Fetching place IDs...")
for park_name in parks:
    try:
        # Use autocomplete
        url = f"https://api.inaturalist.org/v1/places/autocomplete?q={park_name}"
        resp = requests.get(url)
        data = resp.json()
        
        if data.get('results') and len(data['results']) > 0:
            # Take the first result
            place = data['results'][0]
            results[park_name] = {
                "id": place['id'],
                "name": place['name'],
                "display_name": place.get('display_name'),
                "location": place.get('location')
            }
            print(f"Found {park_name}: {place['id']} ({place['name']})")
        else:
            print(f"NOT FOUND: {park_name}")
            
    except Exception as e:
        print(f"Error fetching {park_name}: {e}")
    
    time.sleep(1) # Be nice to API

print("\nRESULTS JSON:")
print(json.dumps(results, indent=2))
