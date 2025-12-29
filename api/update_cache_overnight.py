"""
Overnight Park Cache Updater - Rate Limit Safe

Runs slowly to avoid iNaturalist rate limits. Safe to run overnight.
Saves progress after each species so it can be resumed if interrupted.

Usage:
    python update_cache_overnight.py           # Run all species
    python update_cache_overnight.py --resume  # Skip already cached species
"""

import os
import json
import time
import argparse
import requests
from datetime import datetime

# Supabase config
SUPABASE_URL = "https://tfbmczhkgyfdrvvzeupz.supabase.co"
SUPABASE_KEY = "sb_secret_j2PwpByYdrX6bAk_kyzA-w_cU8jIRKM"

# Rate limit settings - EXTRA SLOW to respect iNaturalist
DELAY_BETWEEN_PARKS = 5       # 5 seconds between each park query
DELAY_BETWEEN_SPECIES = 30    # 30 seconds between species
DELAY_ON_RATE_LIMIT = 300     # 5 minutes if rate limited
MAX_RETRIES = 3               # Retry up to 3 times per park

def load_species():
    with open("species_config.json", "r") as f:
        species = json.load(f)
    return [s for s in species if s.get("taxonomy", {}).get("taxon_id")]

def load_parks():
    parks_path = os.path.join("..", "frontend", "public", "parks.json")
    if not os.path.exists(parks_path):
        parks_path = "parks.json"
    with open(parks_path, "r") as f:
        return json.load(f)

def get_cached_taxons():
    """Get list of taxon_ids already in the cache"""
    url = f"{SUPABASE_URL}/rest/v1/park_observations"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    
    try:
        r = requests.get(url, headers=headers, params={"select": "taxon_id", "limit": 10000}, timeout=15)
        if r.status_code == 200:
            return set([row["taxon_id"] for row in r.json()])
    except:
        pass
    return set()

def fetch_observation_count(taxon_id: int, park_id: int) -> int:
    """Fetch with aggressive retry and backoff"""
    url = f"https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "place_id": park_id,
        "quality_grade": "research",
        "per_page": 0
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json().get("total_results", 0)
            
            elif response.status_code == 429:
                wait_time = DELAY_ON_RATE_LIMIT * (attempt + 1)
                print(f"\n    ⏳ Rate limited! Waiting {wait_time}s...", end="", flush=True)
                time.sleep(wait_time)
                continue
            
            else:
                print(f"\n    ⚠️ HTTP {response.status_code}", end="", flush=True)
                time.sleep(5)
                
        except requests.exceptions.Timeout:
            print(f"\n    ⏱️ Timeout, retrying...", end="", flush=True)
            time.sleep(10)
        except Exception as e:
            print(f"\n    ❌ Error: {str(e)[:30]}", end="", flush=True)
            time.sleep(5)
    
    return -1  # Failed after all retries

def upsert_to_supabase(records: list):
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
            print(f"\n    Supabase error {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"\n    Supabase exception: {str(e)[:100]}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip species already in cache")
    args = parser.parse_args()
    
    print("🦌 OVERNIGHT CACHE UPDATER")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Settings: {DELAY_BETWEEN_PARKS}s between parks, {DELAY_BETWEEN_SPECIES}s between species")
    print()
    
    species_list = load_species()
    parks = load_parks()
    
    print(f"📊 Species: {len(species_list)}")
    print(f"🏞️  Parks: {len(parks)}")
    
    # Calculate estimated time
    total_calls = len(species_list) * len(parks)
    est_hours = (total_calls * DELAY_BETWEEN_PARKS + len(species_list) * DELAY_BETWEEN_SPECIES) / 3600
    print(f"⏱️  Estimated time: {est_hours:.1f} hours")
    print()
    
    if args.resume:
        cached = get_cached_taxons()
        print(f"📦 Already cached: {len(cached)} species")
        species_list = [s for s in species_list if s["taxonomy"]["taxon_id"] not in cached]
        print(f"📝 Remaining: {len(species_list)} species to process")
        print()
    
    total_records = 0
    failed_parks = 0
    
    for i, species in enumerate(species_list, 1):
        taxon_id = species["taxonomy"]["taxon_id"]
        species_name = species["name"]
        
        print(f"[{i}/{len(species_list)}] {species_name} (taxon: {taxon_id})")
        print(f"    Progress: ", end="", flush=True)
        
        species_records = []
        species_failed = 0
        
        for j, park in enumerate(parks):
            count = fetch_observation_count(taxon_id, park["id"])
            
            if count > 0:
                species_records.append({
                    "park_id": park["id"],
                    "taxon_id": taxon_id,
                    "observation_count": count,
                    "updated_at": datetime.utcnow().isoformat()
                })
                print("█", end="", flush=True)  # Found observations
            elif count == 0:
                print(".", end="", flush=True)  # No observations
            else:
                print("x", end="", flush=True)  # Failed
                species_failed += 1
            
            # Progress indicator every 50 parks
            if (j + 1) % 50 == 0:
                print(f" [{j+1}/{len(parks)}] ", end="", flush=True)
            
            time.sleep(DELAY_BETWEEN_PARKS)
        
        print()  # New line after progress
        
        # Save this species
        if species_records:
            if upsert_to_supabase(species_records):
                print(f"    ✅ Saved {len(species_records)} parks with observations")
                total_records += len(species_records)
            else:
                print(f"    ❌ Failed to save to Supabase!")
        else:
            print(f"    📭 No observations found in any park")
        
        if species_failed > 0:
            print(f"    ⚠️ {species_failed} parks failed (rate limited)")
            failed_parks += species_failed
        
        print()
        
        # Longer pause between species
        if i < len(species_list):
            print(f"    💤 Sleeping {DELAY_BETWEEN_SPECIES}s before next species...")
            time.sleep(DELAY_BETWEEN_SPECIES)
    
    print("=" * 50)
    print(f"✅ COMPLETE!")
    print(f"   Total records saved: {total_records}")
    print(f"   Failed parks: {failed_parks}")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
