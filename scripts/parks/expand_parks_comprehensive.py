
import requests
import json
import time

# === 1. COMPLETE GOLD STANDARD LIST ===
# This includes ALL NPS units that are not just "National Parks", but also:
# National Preserves, National Seashores, National Lakeshores, National Recreation Areas, etc.

gold_standard_parks = [
    # === 63 US NATIONAL PARKS ===
    {"id": 49610, "name": "Acadia NP", "state": "ME", "country": "US", "lat": 44.310, "lng": -68.291},
    {"id": 73645, "name": "American Samoa NP", "state": "AS", "country": "US", "lat": -14.245, "lng": -169.927},
    {"id": 53642, "name": "Arches NP", "state": "UT", "country": "US", "lat": 38.723, "lng": -109.586},
    {"id": 72792, "name": "Badlands NP", "state": "SD", "country": "US", "lat": 43.685, "lng": -102.483},
    {"id": 55071, "name": "Big Bend NP", "state": "TX", "country": "US", "lat": 29.298, "lng": -103.230},
    {"id": 95108, "name": "Biscayne NP", "state": "FL", "country": "US", "lat": 25.490, "lng": -80.210},
    {"id": 72635, "name": "Black Canyon NP", "state": "CO", "country": "US", "lat": 38.578, "lng": -107.724},
    {"id": 69110, "name": "Bryce Canyon NP", "state": "UT", "country": "US", "lat": 37.584, "lng": -112.183},
    {"id": 95131, "name": "Canyonlands NP", "state": "UT", "country": "US", "lat": 38.245, "lng": -109.880},
    {"id": 69282, "name": "Capitol Reef NP", "state": "UT", "country": "US", "lat": 38.177, "lng": -111.176},
    {"id": 69109, "name": "Carlsbad Caverns NP", "state": "NM", "country": "US", "lat": 32.141, "lng": -104.553},
    {"id": 3157, "name": "Channel Islands NP", "state": "CA", "country": "US", "lat": 33.987, "lng": -119.911},
    {"id": 53620, "name": "Congaree NP", "state": "SC", "country": "US", "lat": 33.792, "lng": -80.748},
    {"id": 52923, "name": "Crater Lake NP", "state": "OR", "country": "US", "lat": 42.941, "lng": -122.133},
    {"id": 72639, "name": "Cuyahoga Valley NP", "state": "OH", "country": "US", "lat": 41.259, "lng": -81.571},
    {"id": 4504, "name": "Death Valley NP", "state": "CA", "country": "US", "lat": 36.484, "lng": -117.133},
    {"id": 71077, "name": "Denali NP", "state": "AK", "country": "US", "lat": 63.288, "lng": -151.058},
    {"id": 70571, "name": "Dry Tortugas NP", "state": "FL", "country": "US", "lat": 24.649, "lng": -82.872},
    {"id": 53957, "name": "Everglades NP", "state": "FL", "country": "US", "lat": 25.372, "lng": -80.882},
    {"id": 69111, "name": "Gates of the Arctic NP", "state": "AK", "country": "US", "lat": 67.749, "lng": -153.301},
    {"id": 137962, "name": "Gateway Arch NP", "state": "MO", "country": "US", "lat": 38.624, "lng": -90.186},
    {"id": 72841, "name": "Glacier NP", "state": "MT", "country": "US", "lat": 48.683, "lng": -113.800},
    {"id": 69113, "name": "Glacier Bay NP", "state": "AK", "country": "US", "lat": 58.797, "lng": -136.838},
    {"id": 69216, "name": "Grand Canyon NP", "state": "AZ", "country": "US", "lat": 36.172, "lng": -112.685},
    {"id": 69099, "name": "Grand Teton NP", "state": "WY", "country": "US", "lat": 43.818, "lng": -110.705},
    {"id": 69699, "name": "Great Basin NP", "state": "NV", "country": "US", "lat": 38.946, "lng": -114.258},
    {"id": 53632, "name": "Great Sand Dunes NP", "state": "CO", "country": "US", "lat": 37.792, "lng": -105.592},
    {"id": 72645, "name": "Great Smoky Mountains NP", "state": "TN", "country": "US", "lat": 35.601, "lng": -83.508},
    {"id": 69313, "name": "Guadalupe Mountains NP", "state": "TX", "country": "US", "lat": 31.923, "lng": -104.886},
    {"id": 56788, "name": "Haleakala NP", "state": "HI", "country": "US", "lat": 20.707, "lng": -156.159},
    {"id": 7222, "name": "Hawaii Volcanoes NP", "state": "HI", "country": "US", "lat": 19.339, "lng": -155.464},
    {"id": 56706, "name": "Hot Springs NP", "state": "AR", "country": "US", "lat": 34.524, "lng": -93.063},
    {"id": 88, "name": "Indiana Dunes NP", "state": "IN", "country": "US", "lat": 41.650, "lng": -87.053},
    {"id": 95245, "name": "Isle Royale NP", "state": "MI", "country": "US", "lat": 48.011, "lng": -88.828},
    {"id": 3680, "name": "Joshua Tree NP", "state": "CA", "country": "US", "lat": 33.914, "lng": -115.840},
    {"id": 95257, "name": "Katmai NP", "state": "AK", "country": "US", "lat": 58.618, "lng": -155.014},
    {"id": 95258, "name": "Kenai Fjords NP", "state": "AK", "country": "US", "lat": 59.816, "lng": -150.108},
    {"id": 3378, "name": "Kings Canyon NP", "state": "CA", "country": "US", "lat": 36.892, "lng": -118.598},
    {"id": 69115, "name": "Kobuk Valley NP", "state": "AK", "country": "US", "lat": 67.353, "lng": -159.199},
    {"id": 69114, "name": "Lake Clark NP", "state": "AK", "country": "US", "lat": 60.566, "lng": -153.557},
    {"id": 4509, "name": "Lassen Volcanic NP", "state": "CA", "country": "US", "lat": 40.494, "lng": -121.408},
    {"id": 72649, "name": "Mammoth Cave NP", "state": "KY", "country": "US", "lat": 37.198, "lng": -86.131},
    {"id": 69108, "name": "Mesa Verde NP", "state": "CO", "country": "US", "lat": 37.239, "lng": -108.462},
    {"id": 8838, "name": "Mount Rainier NP", "state": "WA", "country": "US", "lat": 46.861, "lng": -121.706},
    {"id": 95209, "name": "New River Gorge NP", "state": "WV", "country": "US", "lat": 37.873, "lng": -81.002},
    {"id": 69097, "name": "North Cascades NP", "state": "WA", "country": "US", "lat": 48.711, "lng": -121.206},
    {"id": 69094, "name": "Olympic NP", "state": "WA", "country": "US", "lat": 47.803, "lng": -123.666},
    {"id": 57573, "name": "Petrified Forest NP", "state": "AZ", "country": "US", "lat": 34.984, "lng": -109.788},
    {"id": 5737, "name": "Pinnacles NP", "state": "CA", "country": "US", "lat": 36.490, "lng": -121.181},
    {"id": 6021, "name": "Redwood NP", "state": "CA", "country": "US", "lat": 41.371, "lng": -124.032},
    {"id": 49676, "name": "Rocky Mountain NP", "state": "CO", "country": "US", "lat": 40.355, "lng": -105.697},
    {"id": 65739, "name": "Saguaro NP", "state": "AZ", "country": "US", "lat": 32.209, "lng": -110.758},
    {"id": 95321, "name": "Sequoia NP", "state": "CA", "country": "US", "lat": 36.508, "lng": -118.575},
    {"id": 9012, "name": "Shenandoah NP", "state": "VA", "country": "US", "lat": 38.492, "lng": -78.469},
    {"id": 72793, "name": "Theodore Roosevelt NP", "state": "ND", "country": "US", "lat": 47.175, "lng": -103.430},
    {"id": 95336, "name": "Virgin Islands NP", "state": "VI", "country": "US", "lat": 18.343, "lng": -64.742},
    {"id": 69101, "name": "Voyageurs NP", "state": "MN", "country": "US", "lat": 48.484, "lng": -92.838},
    {"id": 62621, "name": "White Sands NP", "state": "NM", "country": "US", "lat": 32.779, "lng": -106.333},
    {"id": 72794, "name": "Wind Cave NP", "state": "SD", "country": "US", "lat": 43.580, "lng": -103.439},
    {"id": 72658, "name": "Wrangell-St Elias NP", "state": "AK", "country": "US", "lat": 61.391, "lng": -142.585},
    {"id": 10211, "name": "Yellowstone NP", "state": "WY", "country": "US", "lat": 44.596, "lng": -110.547},
    {"id": 68542, "name": "Yosemite NP", "state": "CA", "country": "US", "lat": 37.848, "lng": -119.557},
    {"id": 50634, "name": "Zion NP", "state": "UT", "country": "US", "lat": 37.298, "lng": -113.026},
    
    # === NATIONAL PRESERVES (Often Missing!) ===
    {"id": 95125, "name": "Big Cypress National Preserve", "state": "FL", "country": "US", "lat": 25.976, "lng": -81.098},
    {"id": 71194, "name": "Big Thicket National Preserve", "state": "TX", "country": "US", "lat": 30.455, "lng": -94.386},
    {"id": 69119, "name": "Mojave National Preserve", "state": "CA", "country": "US", "lat": 35.141, "lng": -115.510},
    {"id": 69118, "name": "Tallgrass Prairie National Preserve", "state": "KS", "country": "US", "lat": 38.432, "lng": -96.556},
    {"id": 95270, "name": "Timucuan Ecological & Historic Preserve", "state": "FL", "country": "US", "lat": 30.418, "lng": -81.459},
    
    # === NATIONAL SEASHORES ===
    {"id": 69324, "name": "Padre Island National Seashore", "state": "TX", "country": "US", "lat": 26.929, "lng": -97.397},
    {"id": 95178, "name": "Cape Cod National Seashore", "state": "MA", "country": "US", "lat": 41.901, "lng": -70.006},
    {"id": 95168, "name": "Assateague Island National Seashore", "state": "MD", "country": "US", "lat": 38.055, "lng": -75.167},
    {"id": 72799, "name": "Point Reyes National Seashore", "state": "CA", "country": "US", "lat": 38.070, "lng": -122.833},
    {"id": 95179, "name": "Cape Hatteras National Seashore", "state": "NC", "country": "US", "lat": 35.232, "lng": -75.672},
    {"id": 95180, "name": "Cape Lookout National Seashore", "state": "NC", "country": "US", "lat": 34.662, "lng": -76.515},
    {"id": 95183, "name": "Cumberland Island National Seashore", "state": "GA", "country": "US", "lat": 30.833, "lng": -81.443},
    {"id": 95185, "name": "Fire Island National Seashore", "state": "NY", "country": "US", "lat": 40.673, "lng": -73.000},
    {"id": 95187, "name": "Gulf Islands National Seashore", "state": "FL", "country": "US", "lat": 30.315, "lng": -87.295},
    
    # === NATIONAL LAKESHORES ===
    {"id": 95162, "name": "Apostle Islands National Lakeshore", "state": "WI", "country": "US", "lat": 46.979, "lng": -90.663},
    {"id": 95200, "name": "Indiana Dunes National Lakeshore", "state": "IN", "country": "US", "lat": 41.650, "lng": -87.053},
    {"id": 95231, "name": "Pictured Rocks National Lakeshore", "state": "MI", "country": "US", "lat": 46.558, "lng": -86.362},
    {"id": 95238, "name": "Sleeping Bear Dunes National Lakeshore", "state": "MI", "country": "US", "lat": 44.865, "lng": -86.053},
    
    # === Wildlife Refuges ===
    {"id": 119523, "name": "Okefenokee NWR", "state": "GA", "country": "US", "lat": 30.799, "lng": -82.305},
    {"id": 119468, "name": "Merritt Island NWR", "state": "FL", "country": "US", "lat": 28.633, "lng": -80.703},
    {"id": 119263, "name": "Chincoteague NWR", "state": "VA", "country": "US", "lat": 37.900, "lng": -75.370},
    {"id": 119204, "name": "Bear River Migratory Bird Refuge", "state": "UT", "country": "US", "lat": 41.470, "lng": -112.265},
    {"id": 63600, "name": "Bosque del Apache NWR", "state": "NM", "country": "US", "lat": 33.758, "lng": -106.828},

    # === Canadian National Parks ===
    {"id": 77932, "name": "Banff National Park", "state": "AB", "country": "CA", "lat": 51.543, "lng": -116.124},
    {"id": 66307, "name": "Jasper National Park", "state": "AB", "country": "CA", "lat": 52.851, "lng": -117.984},
    {"id": 90295, "name": "Algonquin Provincial Park", "state": "ON", "country": "CA", "lat": 45.784, "lng": -78.388},
    {"id": 112797, "name": "Yoho National Park", "state": "BC", "country": "CA", "lat": 51.384, "lng": -116.526},
    {"id": 112799, "name": "Kootenay National Park", "state": "BC", "country": "CA", "lat": 50.962, "lng": -116.041},
    {"id": 55198, "name": "Waterton Lakes NP", "state": "AB", "country": "CA", "lat": 49.081, "lng": -113.933},
    {"id": 68850, "name": "Pacific Rim NP Reserve", "state": "BC", "country": "CA", "lat": 48.831, "lng": -125.147},

    # === Mexican Parks ===
    {"name": "PN Cumbres de Monterrey", "id": 55243, "lat": 25.356, "lng": -100.340, "state": "MX", "country": "MX"},
    {"name": "RB El Vizcaino", "id": 55258, "lat": 27.191, "lng": -113.749, "state": "MX", "country": "MX"},
    {"name": "RB Mariposa Monarca", "id": 73998, "lat": 19.652, "lng": -100.243, "state": "MX", "country": "MX"},
    {"name": "PN Sierra de San Pedro Martir", "id": 55334, "lat": 30.934, "lng": -115.441, "state": "MX", "country": "MX"},
    {"name": "Area de Proteccion Canon de Santa Elena", "id": 55218, "lat": 29.107, "lng": -103.802, "state": "MX", "country": "MX"},
    {"name": "RB Montes Azules", "id": 55300, "lat": 16.520, "lng": -91.128, "state": "MX", "country": "MX"}
]

# === 2. HELPER FUNCTIONS ===
def get_headers():
    return {
        "User-Agent": "WildlifeID-App/1.0 (wildlife-id@example.com)",
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

# === 3. EXECUTION ===
valid_state_ids = set()
states = get_state_ids()
for s in states:
    valid_state_ids.add(s['id'])

# Add country IDs for broader matching
valid_state_ids.add(1)     # United States
valid_state_ids.add(6712)  # Canada
valid_state_ids.add(6793)  # Mexico

print(f"Loaded {len(valid_state_ids)} target region IDs.")

final_list = gold_standard_parks.copy()
known_ids = set(p['id'] for p in final_list)
print(f"Starting with {len(final_list)} gold standard parks.")

# === EXPANDED GLOBAL QUERIES ===
global_queries = [
    "State Park",
    "Provincial Park",
    "National Forest",
    "National Wildlife Refuge",
    "Nature Preserve",
    "National Preserve",
    "National Seashore",
    "National Recreation Area",
    "County Park",
    "Regional Park",
    "State Forest",
    "Wilderness Area"
]

MAX_PAGES = 50  # 50 pages * 30 per page = 1500 potential results per term

for query in global_queries:
    print(f"Global Scan: '{query}'...")
    
    for page in range(1, MAX_PAGES + 1):
        try:
            url = f"https://api.inaturalist.org/v1/places/autocomplete?q={query}&per_page=30&page={page}"
            resp = requests.get(url, headers=get_headers())
            
            if resp.status_code != 200:
                break
                
            results = resp.json().get('results', [])
            if not results:
                break # No more results for this query
            
            for p in results:
                if p['id'] in known_ids:
                    continue
                    
                # DATA QUALITY FILTER:
                name = p.get('name', '')
                lower_name = name.lower()
                
                # Exclude umbrella/system-level entries
                if any(skip in lower_name for skip in ["state parks", "parks system", "department", "division", "administration"]):
                    continue
                    
                # ANCESTOR FILTER:
                ancestors = p.get('ancestor_place_ids') or []
                
                is_relevant = False
                for aid in ancestors:
                    if aid in valid_state_ids:
                        is_relevant = True
                        break
                
                if is_relevant and p.get('location'):
                    lat, lng = map(float, p['location'].split(','))
                    
                    # Parse state from display_name
                    state_disp = "Unknown"
                    parts = p.get('display_name', '').split(", ")
                    if len(parts) >= 2:
                        state_disp = parts[-2]
                    
                    # Derive country code from ancestors
                    country = "Unknown"
                    if 1 in ancestors:
                        country = "US"
                    elif 6712 in ancestors:
                        country = "CA"
                    elif 6793 in ancestors:
                        country = "MX"
                    
                    final_list.append({
                        "id": p['id'],
                        "name": p['name'],
                        "state": state_disp,
                        "country": country,
                        "lat": lat,
                        "lng": lng
                    })
                    known_ids.add(p['id'])
        
            time.sleep(0.25) # Rate limit
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

# === 4. SAVE ===
with open('frontend/public/parks.json', 'w') as f:
    json.dump(final_list, f, indent=2)

print(f"\n=== DONE ===")
print(f"Total Parks: {len(final_list)}")
