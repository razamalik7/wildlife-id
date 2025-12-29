"""
FAST Park Cache Updater - Smart Batching

Uses a smarter approach:
1. Fetch all observations for a species (up to 10,000)
2. Extract place_ids and count matches with our parks
3. Much faster than individual park queries!

For species with >10,000 observations, falls back to sampling.
"""

import os
import json
import time
import requests
from datetime import datetime
from collections import defaultdict

# Supabase config
SUPABASE_URL = "https://tfbmczhkgyfdrvvzeupz.supabase.co"
SUPABASE_KEY = "sb_secret_j2PwpByYdrX6bAk_kyzA-w_cU8jIRKM"

# Rate limit settings  
DELAY_BETWEEN_PAGES = 1.0     # 1 second between pagination requests
DELAY_BETWEEN_SPECIES = 5    # 5 seconds between species
MAX_OBSERVATIONS = 10000     # Max observations to fetch per species

def load_species():
    with open("species_config.json", "r") as f:
        species = json.load(f)
    return [s for s in species if s.get("taxonomy", {}).get("taxon_id")]

def load_parks():
    parks_path = os.path.join("..", "frontend", "public", "parks.json")
    if not os.path.exists(parks_path):
        parks_path = "parks.json"
    with open(parks_path, "r") as f:
        parks = json.load(f)
    # Create a set of park IDs for fast lookup
    return {p["id"]: p["name"] for p in parks}

def get_total_observations(taxon_id: int) -> int:
    """Get total observation count for a taxon"""
    try:
        r = requests.get("https://api.inaturalist.org/v1/observations", params={
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "per_page": 0
        }, timeout=15)
        if r.status_code == 200:
            return r.json().get("total_results", 0)
    except:
        pass
    return 0

def fetch_observations_batch(taxon_id: int, park_ids: set, max_obs: int = 10000) -> dict:
    """Fetch observations and count by park - FAST method"""
    counts = defaultdict(int)
    per_page = 200
    page = 1
    total_fetched = 0
    
    # North America place IDs: US=1, Canada=6712, Mexico=6793
    NORTH_AMERICA_PLACES = "1,6712,6793"
    
    while total_fetched < max_obs:
        try:
            r = requests.get("https://api.inaturalist.org/v1/observations", params={
                "taxon_id": taxon_id,
                "place_id": NORTH_AMERICA_PLACES,  # Filter to North America only!
                "quality_grade": "research",
                "per_page": per_page,
                "page": page,
                "order": "desc",
                "order_by": "observed_on"
            }, timeout=30)
            
            if r.status_code == 429:
                print("\n    ⏳ Rate limited, waiting 60s...", end="", flush=True)
                time.sleep(60)
                continue
                
            if r.status_code != 200:
                print(f"\n    ⚠️ HTTP {r.status_code}", end="", flush=True)
                break
            
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            
            # Count observations by park
            for obs in results:
                obs_places = set(obs.get("place_ids", []))
                matching_parks = obs_places & park_ids
                for park_id in matching_parks:
                    counts[park_id] += 1
            
            total_fetched += len(results)
            page += 1
            
            # Progress indicator
            if page % 10 == 0:
                print(f".", end="", flush=True)
            
            time.sleep(DELAY_BETWEEN_PAGES)
            
            # Check if we've fetched all
            total_available = data.get("total_results", 0)
            if total_fetched >= total_available:
                break
                
        except Exception as e:
            print(f"\n    ❌ Error: {str(e)[:50]}", end="", flush=True)
            time.sleep(5)
            break
    
    return dict(counts), total_fetched

def upsert_to_supabase(records: list) -> bool:
    """Save records to Supabase"""
    if not records:
        return True
        
    url = f"{SUPABASE_URL}/rest/v1/park_observations"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    
    try:
        response = requests.post(url, headers=headers, json=records, timeout=30)
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"\n    Supabase error {response.status_code}: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"\n    Supabase exception: {str(e)[:50]}")
        return False

def clear_existing_cache():
    """Clear existing cache data"""
    url = f"{SUPABASE_URL}/rest/v1/park_observations"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    try:
        # Delete all records
        r = requests.delete(url, headers=headers, params={"taxon_id": "gt.0"}, timeout=30)
        return r.status_code in [200, 204]
    except:
        return False

def main():
    print("🚀 FAST PARK CACHE UPDATER")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    species_list = load_species()
    park_ids = load_parks()
    park_id_set = set(park_ids.keys())
    
    print(f"📊 Species: {len(species_list)}")
    print(f"🏞️  Parks: {len(park_ids)}")
    print()
    
    # Ask about clearing cache
    clear = input("Clear existing cache first? (y/n): ").lower() == 'y'
    if clear:
        print("🗑️  Clearing cache...", end=" ")
        if clear_existing_cache():
            print("Done!")
        else:
            print("Failed (continuing anyway)")
    print()
    
    total_records = 0
    
    for i, species in enumerate(species_list, 1):
        taxon_id = species["taxonomy"]["taxon_id"]
        species_name = species["name"]
        
        # Get total count first
        total_obs = get_total_observations(taxon_id)
        time.sleep(1)
        
        print(f"[{i}/{len(species_list)}] {species_name} (taxon: {taxon_id}) - {total_obs:,} observations")
        
        if total_obs == 0:
            print("    📭 No observations found")
            print()
            continue
        
        # Fetch and count
        print(f"    Fetching", end="", flush=True)
        counts, fetched = fetch_observations_batch(taxon_id, park_id_set, MAX_OBSERVATIONS)
        print(f" ({fetched:,} fetched)")
        
        if not counts:
            print("    📭 No parks matched")
            print()
            continue
        
        # Build records
        records = []
        for park_id, count in counts.items():
            # Scale up if we sampled
            if total_obs > fetched and fetched > 0:
                scaled_count = int(count * (total_obs / fetched))
            else:
                scaled_count = count
                
            records.append({
                "park_id": park_id,
                "taxon_id": taxon_id,
                "observation_count": scaled_count,
                "updated_at": datetime.utcnow().isoformat() + "+00:00"
            })
        
        # Sort by count for display
        records.sort(key=lambda x: -x["observation_count"])
        
        # Show top parks
        top_parks = records[:5]
        print(f"    Top parks: ", end="")
        for r in top_parks:
            name = park_ids.get(r["park_id"], f"ID:{r['park_id']}")
            print(f"{name[:15]}({r['observation_count']})", end=" ")
        print()
        
        # Save to Supabase
        if upsert_to_supabase(records):
            print(f"    ✅ Saved {len(records)} parks")
            total_records += len(records)
        else:
            print(f"    ❌ Failed to save!")
        
        print()
        time.sleep(DELAY_BETWEEN_SPECIES)
    
    print("=" * 50)
    print(f"✅ COMPLETE!")
    print(f"   Total records saved: {total_records}")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
