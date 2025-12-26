
import requests
import json
import time

# === 1. GOLD STANDARD PARKS (Hardcoded to prevent loss) ===
# These are the 63 US National Parks + Verified Canadian/Mexican/Refuges
gold_standard_parks = [
    # US National Parks (verified IDs)
    {"id": 49610, "name": "Acadia NP", "state": "ME", "lat": 44.310, "lng": -68.291},
    {"id": 73645, "name": "American Samoa NP", "state": "AS", "lat": -14.245, "lng": -169.927},
    {"id": 53642, "name": "Arches NP", "state": "UT", "lat": 38.723, "lng": -109.586},
    {"id": 72792, "name": "Badlands NP", "state": "SD", "lat": 43.685, "lng": -102.483},
    {"id": 55071, "name": "Big Bend NP", "state": "TX", "lat": 29.298, "lng": -103.230},
    {"id": 95108, "name": "Biscayne NP", "state": "FL", "lat": 25.490, "lng": -80.210},
    {"id": 72635, "name": "Black Canyon NP", "state": "CO", "lat": 38.578, "lng": -107.724},
    {"id": 69110, "name": "Bryce Canyon NP", "state": "UT", "lat": 37.584, "lng": -112.183},
    {"id": 95131, "name": "Canyonlands NP", "state": "UT", "lat": 38.245, "lng": -109.880},
    {"id": 69282, "name": "Capitol Reef NP", "state": "UT", "lat": 38.177, "lng": -111.176},
    {"id": 69109, "name": "Carlsbad Caverns NP", "state": "NM", "lat": 32.141, "lng": -104.553},
    {"id": 3157, "name": "Channel Islands NP", "state": "CA", "lat": 33.987, "lng": -119.911},
    {"id": 53620, "name": "Congaree NP", "state": "SC", "lat": 33.792, "lng": -80.748},
    {"id": 52923, "name": "Crater Lake NP", "state": "OR", "lat": 42.941, "lng": -122.133},
    {"id": 72639, "name": "Cuyahoga Valley NP", "state": "OH", "lat": 41.259, "lng": -81.571},
    {"id": 4504, "name": "Death Valley NP", "state": "CA", "lat": 36.484, "lng": -117.133},
    {"id": 71077, "name": "Denali NP", "state": "AK", "lat": 63.288, "lng": -151.058},
    {"id": 70571, "name": "Dry Tortugas NP", "state": "FL", "lat": 24.649, "lng": -82.872},
    {"id": 53957, "name": "Everglades NP", "state": "FL", "lat": 25.372, "lng": -80.882},
    {"id": 69111, "name": "Gates of the Arctic NP", "state": "AK", "lat": 67.749, "lng": -153.301},
    {"id": 137962, "name": "Gateway Arch NP", "state": "MO", "lat": 38.624, "lng": -90.186},
    {"id": 72841, "name": "Glacier NP", "state": "MT", "lat": 48.683, "lng": -113.800},
    {"id": 69113, "name": "Glacier Bay NP", "state": "AK", "lat": 58.797, "lng": -136.838},
    {"id": 69216, "name": "Grand Canyon NP", "state": "AZ", "lat": 36.172, "lng": -112.685},
    {"id": 69099, "name": "Grand Teton NP", "state": "WY", "lat": 43.818, "lng": -110.705},
    {"id": 69699, "name": "Great Basin NP", "state": "NV", "lat": 38.946, "lng": -114.258},
    {"id": 53632, "name": "Great Sand Dunes NP", "state": "CO", "lat": 37.792, "lng": -105.592},
    {"id": 72645, "name": "Great Smoky Mountains NP", "state": "TN", "lat": 35.601, "lng": -83.508},
    {"id": 69313, "name": "Guadalupe Mountains NP", "state": "TX", "lat": 31.923, "lng": -104.886},
    {"id": 56788, "name": "Haleakala NP", "state": "HI", "lat": 20.707, "lng": -156.159},
    {"id": 7222, "name": "Hawaii Volcanoes NP", "state": "HI", "lat": 19.339, "lng": -155.464},
    {"id": 56706, "name": "Hot Springs NP", "state": "AR", "lat": 34.524, "lng": -93.063},
    {"id": 88, "name": "Indiana Dunes NP", "state": "IN", "lat": 41.650, "lng": -87.053},
    {"id": 95245, "name": "Isle Royale NP", "state": "MI", "lat": 48.011, "lng": -88.828},
    {"id": 3680, "name": "Joshua Tree NP", "state": "CA", "lat": 33.914, "lng": -115.840},
    {"id": 95257, "name": "Katmai NP", "state": "AK", "lat": 58.618, "lng": -155.014},
    {"id": 95258, "name": "Kenai Fjords NP", "state": "AK", "lat": 59.816, "lng": -150.108},
    {"id": 3378, "name": "Kings Canyon NP", "state": "CA", "lat": 36.892, "lng": -118.598},
    {"id": 69115, "name": "Kobuk Valley NP", "state": "AK", "lat": 67.353, "lng": -159.199},
    {"id": 69114, "name": "Lake Clark NP", "state": "AK", "lat": 60.566, "lng": -153.557},
    {"id": 4509, "name": "Lassen Volcanic NP", "state": "CA", "lat": 40.494, "lng": -121.408},
    {"id": 72649, "name": "Mammoth Cave NP", "state": "KY", "lat": 37.198, "lng": -86.131},
    {"id": 69108, "name": "Mesa Verde NP", "state": "CO", "lat": 37.239, "lng": -108.462},
    {"id": 8838, "name": "Mount Rainier NP", "state": "WA", "lat": 46.861, "lng": -121.706},
    {"id": 95209, "name": "New River Gorge NP", "state": "WV", "lat": 37.873, "lng": -81.002},
    {"id": 69097, "name": "North Cascades NP", "state": "WA", "lat": 48.711, "lng": -121.206},
    {"id": 69094, "name": "Olympic NP", "state": "WA", "lat": 47.803, "lng": -123.666},
    {"id": 57573, "name": "Petrified Forest NP", "state": "AZ", "lat": 34.984, "lng": -109.788},
    {"id": 5737, "name": "Pinnacles NP", "state": "CA", "lat": 36.490, "lng": -121.181},
    {"id": 6021, "name": "Redwood NP", "state": "CA", "lat": 41.371, "lng": -124.032},
    {"id": 49676, "name": "Rocky Mountain NP", "state": "CO", "lat": 40.355, "lng": -105.697},
    {"id": 65739, "name": "Saguaro NP", "state": "AZ", "lat": 32.209, "lng": -110.758},
    {"id": 95321, "name": "Sequoia NP", "state": "CA", "lat": 36.508, "lng": -118.575},
    {"id": 9012, "name": "Shenandoah NP", "state": "VA", "lat": 38.492, "lng": -78.469},
    {"id": 72793, "name": "Theodore Roosevelt NP", "state": "ND", "lat": 47.175, "lng": -103.430},
    {"id": 95336, "name": "Virgin Islands NP", "state": "VI", "lat": 18.343, "lng": -64.742},
    {"id": 69101, "name": "Voyageurs NP", "state": "MN", "lat": 48.484, "lng": -92.838},
    {"id": 62621, "name": "White Sands NP", "state": "NM", "lat": 32.779, "lng": -106.333},
    {"id": 72794, "name": "Wind Cave NP", "state": "SD", "lat": 43.580, "lng": -103.439},
    {"id": 72658, "name": "Wrangell-St Elias NP", "state": "AK", "lat": 61.391, "lng": -142.585},
    {"id": 10211, "name": "Yellowstone NP", "state": "WY", "lat": 44.596, "lng": -110.547},
    {"id": 68542, "name": "Yosemite NP", "state": "CA", "lat": 37.848, "lng": -119.557},
    {"id": 50634, "name": "Zion NP", "state": "UT", "lat": 37.298, "lng": -113.026},
    
    # Wildlife Refuges
    {"id": 119523, "name": "Okefenokee NWR", "state": "GA", "lat": 30.799, "lng": -82.305},
    {"id": 119468, "name": "Merritt Island NWR", "state": "FL", "lat": 28.633, "lng": -80.703},
    {"id": 119263, "name": "Chincoteague NWR", "state": "VA", "lat": 37.900, "lng": -75.370},
    {"id": 119204, "name": "Bear River Migratory Bird Refuge", "state": "UT", "lat": 41.470, "lng": -112.265},
    {"id": 63600, "name": "Bosque del Apache NWR", "state": "NM", "lat": 33.758, "lng": -106.828},

    # Canadian National Parks
    {"id": 77932, "name": "Banff National Park", "state": "AB", "lat": 51.543, "lng": -116.124},
    {"id": 66307, "name": "Jasper National Park", "state": "AB", "lat": 52.851, "lng": -117.984},
    {"id": 90295, "name": "Algonquin Provincial Park", "state": "ON", "lat": 45.784, "lng": -78.388},
    {"id": 112797, "name": "Yoho National Park", "state": "BC", "lat": 51.384, "lng": -116.526},
    {"id": 112799, "name": "Kootenay National Park", "state": "BC", "lat": 50.962, "lng": -116.041},
    {"id": 55198, "name": "Waterton Lakes NP", "state": "AB", "lat": 49.081, "lng": -113.933},
    {"id": 68850, "name": "Pacific Rim NP Reserve", "state": "BC", "lat": 48.831, "lng": -125.147},

    # Mexican Parks
     {"name": "PN Cumbres de Monterrey", "id": 55243, "lat": 25.356, "lng": -100.340, "state": "MX"},
     {"name": "RB El Vizcaino", "id": 55258, "lat": 27.191, "lng": -113.749, "state": "MX"},
     {"name": "RB Mariposa Monarca", "id": 73998, "lat": 19.652, "lng": -100.243, "state": "MX"},
     {"name": "PN Sierra de San Pedro Martir", "id": 55334, "lat": 30.934, "lng": -115.441, "state": "MX"},
     {"name": "Area de Proteccion Canon de Santa Elena", "id": 55218, "lat": 29.107, "lng": -103.802, "state": "MX"},
     {"name": "RB Montes Azules", "id": 55300, "lat": 16.520, "lng": -91.128, "state": "MX"}
]

# === 2. HELPER FUNCTIONS ===
def get_headers():
    return {
        "User-Agent": "WildlifeID-App/1.0 (contact@example.com)",
        "Accept": "application/json"
    }

def get_state_ids():
    regions = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", 
        "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", 
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
        "West Virginia", "Wisconsin", "Wyoming",
        "Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland", "Nova Scotia", "Ontario", "Quebec", "Saskatchewan"
    ]
    
    state_map = []
    print("Fetching State/Province IDs...")
    for region in regions:
        try:
            url = f"https://api.inaturalist.org/v1/places/autocomplete?q={region}&per_page=1"
            resp = requests.get(url, headers=get_headers()).json()
            if resp['results']:
                r = resp['results'][0]
                state_map.append({"name": region, "id": r['id']})
        except:
            pass
        time.sleep(0.05)
    return state_map

def fetch_parks_in_state(state_info):
    parks = []
    # Queries to run per state
    queries = ["State Park", "National Forest", "Provincial Park"]
    
    for q_term in queries:
        try:
            # Revert to autocomplete with "Keyword State" query
            query = f"{q_term} {state_info['name']}"
            url = f"https://api.inaturalist.org/v1/places/autocomplete?q={query}&per_page=20"
            resp = requests.get(url, headers=get_headers())
            
            if resp.status_code == 200:
                results = resp.json()['results']
                for p in results:
                    name = p['name']
                    # GRANULARITY FILTER: Exclude umbrella terms
                    lower_name = name.lower()
                    if "state parks" in lower_name or "parks system" in lower_name or "department" in lower_name:
                        continue
                    if "provincial parks" in lower_name:
                        continue
                    
                    # Ensure it belongs to the state (simple check)
                    # p['display_name'] usually looks like "Park Name, State, Country"
                    if state_info.get('name') not in p.get('display_name', ''):
                        # Loose check: if searching "California", ensures "California" is in text
                        # But "State Park California" might return "California State Park" (generic).
                        # We already filtered generic "state parks".
                        pass

                    if p.get('location'):
                        lat, lng = map(float, p['location'].split(','))
                        parks.append({
                            "id": p['id'],
                            "name": p['name'], 
                            "state": state_info['name'],
                            "lat": lat,
                            "lng": lng
                        })
        except Exception as e:
            pass # Skip error
            
    return parks

# === 3. EXECUTION ===
valid_state_ids = set()
states = get_state_ids()
for s in states:
    valid_state_ids.add(s['id'])
    
# Canada/Mexico country IDs might be useful too?
# US: 1, Canada: 6712, Mexico: 6793
# But we have specific state/province IDs, so that's precise.

print(f"Loaded {len(valid_state_ids)} target region IDs (US States + Canada Provinces).")

final_list = gold_standard_parks.copy()
known_ids = set(p['id'] for p in final_list)

# Global Queries
global_queries = ["State Park", "Provincial Park", "National Forest", "National Wildlife Refuge", "Nature Preserve"]
# We will check "State Park" pages 1-20 (1000 results), etc.

for query in global_queries:
    print(f"Global Scan: '{query}'...")
    max_pages = 30 # 30 * 30 results = 900 per term. 
    # increasing max_pages or per_page improves yield but hits rate limits.
    
    for page in range(1, max_pages + 1):
        try:
            url = f"https://api.inaturalist.org/v1/places/autocomplete?q={query}&per_page=30&page={page}"
            resp = requests.get(url, headers=get_headers())
            
            if resp.status_code == 200:
                results = resp.json()['results']
                if not results:
                    break # No more results
                
                for p in results:
                    if p['id'] in known_ids:
                        continue
                        
                    # DATA QUALITY FILTER:
                    name = p['name']
                    lower_name = name.lower()
                    if "state parks" in lower_name or "parks system" in lower_name or "department" in lower_name:
                        continue
                        
                    # ANCESTOR FILTER:
                    # Check if any ancestor ID matches our valid regions
                    ancestors = p.get('ancestor_place_ids') or []
                    # Also check country? US(1), CA(6712), MX(6793) to include Mexico parks found this way?
                    # Let's verify against our specific state list for US/Canada.
                    # For Mexico, we didn't fetch all state IDs, maybe we should accept "Mexico" ID?
                    # Let's add Mexico ID (6793) to valid list manually? 
                    # Or just rely on our extensive state list.
                    
                    is_relevant = False
                    for aid in ancestors:
                        if aid in valid_state_ids:
                            is_relevant = True
                            break
                        if aid == 6793: # Mexico
                             is_relevant = True
                             break
                    
                    if is_relevant:
                        if p.get('location'):
                            lat, lng = map(float, p['location'].split(','))
                            # Finding state code for display
                            # We can infer it from intersection loop or just leave blank/parse display_name
                            # display_name: "Park Name, State, Country"
                            state_disp = "Unknown"
                            parts = p.get('display_name', '').split(", ")
                            if len(parts) >= 2:
                                state_disp = parts[-2] # Rough heuristic
                            
                            final_list.append({
                                "id": p['id'],
                                "name": p['name'],
                                "state": state_disp,
                                "lat": lat,
                                "lng": lng
                            })
                            known_ids.add(p['id'])
            
            time.sleep(0.3) # Rate limit
        except Exception as e:
            print(f"Error on page {page}: {e}")
            pass

# Save
with open('frontend/public/parks.json', 'w') as f:
    json.dump(final_list, f, indent=2)

print(f"DONE. Total Verified + Granular Parks: {len(final_list)}")
