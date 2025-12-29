"""
Find and fix invalid park IDs in parks.json
Searches iNaturalist for the correct place_id by park name.
"""

import json
import requests
import time

PARKS_PATH = "../frontend/public/parks.json"

def test_park_id(park_id: int) -> bool:
    """Test if a park_id is valid on iNaturalist observations endpoint"""
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/observations",
            params={"place_id": park_id, "per_page": 0},
            timeout=10
        )
        return r.status_code == 200
    except:
        return False

def search_correct_id(park_name: str, state: str = None, country: str = None) -> dict:
    """Search iNaturalist for the correct place_id"""
    # Clean up name for search
    search_name = park_name.replace(" NP", " National Park")
    search_name = search_name.replace(" NM", " National Monument")
    search_name = search_name.replace(" NHP", " National Historical Park")
    search_name = search_name.replace(" NHS", " National Historic Site")
    
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/places/autocomplete",
            params={"q": search_name, "per_page": 10},
            timeout=10
        )
        
        if r.status_code == 200:
            results = r.json().get("results", [])
            for result in results:
                name = result.get("display_name", "")
                # Check if it's a national park type
                if any(x in name.lower() for x in ["national park", "national monument", "national", "park"]):
                    return {
                        "id": result["id"],
                        "name": result["display_name"],
                        "bbox": result.get("bounding_box_geojson")
                    }
            # Return first result if no park-specific match
            if results:
                return {
                    "id": results[0]["id"],
                    "name": results[0]["display_name"],
                    "bbox": results[0].get("bounding_box_geojson")
                }
    except Exception as e:
        print(f"    Search error: {e}")
    
    return None

def main():
    with open(PARKS_PATH, "r") as f:
        parks = json.load(f)
    
    print(f"🏞️  Checking {len(parks)} parks for invalid IDs...\n")
    
    invalid_parks = []
    fixed_parks = []
    
    for i, park in enumerate(parks):
        print(f"[{i+1}/{len(parks)}] {park['name']} (ID: {park['id']})", end=" ", flush=True)
        
        if test_park_id(park["id"]):
            print("✓")
        else:
            print("❌ INVALID")
            invalid_parks.append(park)
            
            # Try to find correct ID
            print(f"    Searching for correct ID...", end=" ", flush=True)
            result = search_correct_id(park["name"], park.get("state"), park.get("country"))
            
            if result:
                print(f"Found: {result['id']} ({result['name']})")
                fixed_parks.append({
                    "original": park,
                    "fixed_id": result["id"],
                    "fixed_name": result["name"]
                })
            else:
                print("NOT FOUND")
        
        time.sleep(0.8)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total parks: {len(parks)}")
    print(f"  Invalid IDs: {len(invalid_parks)}")
    print(f"  Fixed: {len(fixed_parks)}")
    
    if fixed_parks:
        print(f"\nFIXES TO APPLY:")
        for fix in fixed_parks:
            print(f"  {fix['original']['name']}: {fix['original']['id']} -> {fix['fixed_id']}")
        
        # Save fixes
        with open("park_id_fixes.json", "w") as f:
            json.dump(fixed_parks, f, indent=2)
        print(f"\n📁 Fixes saved to park_id_fixes.json")
        
        # Ask to apply
        apply = input("\nApply fixes to parks.json? (y/n): ")
        if apply.lower() == "y":
            for fix in fixed_parks:
                for park in parks:
                    if park["id"] == fix["original"]["id"]:
                        park["id"] = fix["fixed_id"]
                        break
            
            with open(PARKS_PATH, "w") as f:
                json.dump(parks, f, indent=2)
            print("✅ Fixes applied!")
    
    if invalid_parks and not fixed_parks:
        print("\n⚠️ Could not find fixes for invalid parks. Manual review needed.")
        with open("invalid_parks.json", "w") as f:
            json.dump(invalid_parks, f, indent=2)
        print("📁 Invalid parks saved to invalid_parks.json")

if __name__ == "__main__":
    main()
