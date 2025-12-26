
import requests
import json

# 1. Existing Data (Manual copy to ensure no data loss from previous steps)
us_nps = [
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
    {"id": 50634, "name": "Zion NP", "state": "UT", "lat": 37.298, "lng": -113.026}
]

state_parks = [
    {"id": 158608, "name": "Custer State Park", "state": "SD", "lat": 43.735, "lng": -103.424},
    {"id": 70218, "name": "Baxter State Park", "state": "ME", "lat": 46.026, "lng": -68.946},
    {"id": 77159, "name": "Adirondack Park", "state": "NY", "lat": 43.965, "lng": -74.307},
    {"id": 119781, "name": "Harriman State Park", "state": "ID", "lat": 44.339, "lng": -111.449},
    {"id": 147606, "name": "Porcupine Mts Wilderness", "state": "MI", "lat": 46.762, "lng": -89.775},
    {"id": 117396, "name": "Itasca State Park", "state": "MN", "lat": 47.189, "lng": -95.223},
    {"id": 146133, "name": "Watkins Glen State Park", "state": "NY", "lat": 42.368, "lng": -76.899},
    {"id": 129021, "name": "Starved Rock State Park", "state": "IL", "lat": 41.313, "lng": -88.996}
]

forests = [
    {"id": 68063, "name": "Pisgah National Forest", "state": "NC", "lat": 35.665, "lng": -82.341},
    {"id": 154466, "name": "Nantahala National Forest", "state": "NC", "lat": 35.096, "lng": -83.858},
    {"id": 159938, "name": "George Washington & Jefferson NF", "state": "VA", "lat": 38.212, "lng": -79.339},
    {"id": 111968, "name": "Ocala National Forest", "state": "FL", "lat": 29.207, "lng": -81.730},
    {"id": 124485, "name": "White Mountain National Forest", "state": "NH", "lat": 44.149, "lng": -71.418},
    {"id": 55713, "name": "Green Mountain National Forest", "state": "VT", "lat": 43.452, "lng": -72.797},
    {"id": 123834, "name": "Monongahela National Forest", "state": "WV", "lat": 38.288, "lng": -80.377},
    {"id": 151485, "name": "Allegheny National Forest", "state": "PA", "lat": 41.652, "lng": -79.021},
    {"id": 178553, "name": "Superior National Forest", "state": "MN", "lat": 47.887, "lng": -91.674}
]

refuges = [
    {"id": 119523, "name": "Okefenokee NWR", "state": "GA", "lat": 30.799, "lng": -82.305},
    {"id": 119468, "name": "Merritt Island NWR", "state": "FL", "lat": 28.633, "lng": -80.703},
    {"id": 119263, "name": "Chincoteague NWR", "state": "VA", "lat": 37.900, "lng": -75.370},
    {"id": 119204, "name": "Bear River Migratory Bird Refuge", "state": "UT", "lat": 41.470, "lng": -112.265},
    {"id": 63600, "name": "Bosque del Apache NWR", "state": "NM", "lat": 33.758, "lng": -106.828}
]

canada = [
    {"id": 77932, "name": "Banff National Park", "state": "AB", "lat": 51.543, "lng": -116.124},
    {"id": 66307, "name": "Jasper National Park", "state": "AB", "lat": 52.851, "lng": -117.984},
    {"id": 90295, "name": "Algonquin Provincial Park", "state": "ON", "lat": 45.784, "lng": -78.388},
    {"id": 112797, "name": "Yoho National Park", "state": "BC", "lat": 51.384, "lng": -116.526},
    {"id": 112799, "name": "Kootenay National Park", "state": "BC", "lat": 50.962, "lng": -116.041},
    {"id": 55198, "name": "Waterton Lakes NP", "state": "AB", "lat": 49.081, "lng": -113.933},
    {"id": 68850, "name": "Pacific Rim NP Reserve", "state": "BC", "lat": 48.831, "lng": -125.147}
]

# 2. Mexican Parks to Add
mexico_targets = [
    "Parque Nacional Cumbres de Monterrey",
    "Reserva de la Biosfera El Vizcaíno",
    "Reserva de la Biosfera Mariposa Monarca",
    "Parque Nacional Sierra de San Pedro Mártir",
    "Area de Protección de Flora y Fauna Cañón de Santa Elena",
    "Reserva de la Biosfera Montes Azules"
]

mexico_found = []

print("--- FETCHING MEXICAN PARK IDS ---")
for name in mexico_targets:
    try:
        url = f"https://api.inaturalist.org/v1/places/autocomplete?q={name}&per_page=1"
        resp = requests.get(url).json()
        if resp['results']:
            p = resp['results'][0]
            print(f"Found {name}: ID {p['id']} - {p['display_name']}")
            short_name = name.replace("Parque Nacional", "PN").replace("Reserva de la Biosfera", "RB")
            mexico_found.append({
                "name": short_name,
                "id": p['id'],
                "lat": float(p['location'].split(',')[0]),
                "lng": float(p['location'].split(',')[1]),
                "state": "MX"
            })
        else:
            print(f"Not found: {name}")
    except Exception as e:
        print(f"Error for {name}: {e}")

# 3. Consolidate ALL
all_parks = us_nps + state_parks + forests + refuges + canada + mexico_found

# 4. Debugging Great Smoky Mountains Count
print("\n--- DEBUGGING GRSM COUNT ---")
ids_to_check = [
    72645, # Current ID (National Park)
    59,    # Old ID
    1274,  # Potential larger region?
]
taxon_id = 41638 # Black Bear

for pid in ids_to_check:
    try:
        url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&place_id={pid}&per_page=0"
        count = requests.get(url).json()['total_results']
        print(f"Place ID {pid}: {count} observations")
    except:
        print(f"Failed to check ID {pid}")

# 5. Save Final JSON
with open('frontend/public/parks.json', 'w') as f:
    json.dump(all_parks, f, indent=2)

print(f"\nSaved {len(all_parks)} parks to frontend/public/parks.json")
