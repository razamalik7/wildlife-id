from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
import time
from datetime import datetime
from ai_engine import predict_animal

# Trigger Hot Reload for Taxonomy ✨
app = FastAPI()

# --- CONFIGURATION ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
SPECIES_CONFIG_PATH = "species_config.json"

def load_species_data():
    if os.path.exists(SPECIES_CONFIG_PATH):
        with open(SPECIES_CONFIG_PATH, 'r') as f:
            return json.load(f)
    return []

# Load data at startup
SPECIES_DATA = load_species_data()

@app.get("/")
def read_root():
    return {"Status": "Wildlife AI is Online 🟢"}

@app.get("/species")
def get_all_species():
    """
    Returns the Master List of animals so the frontend can 
    display the Anidex (Locked/Unlocked) and Native/Invasive tabs.
    Reloads from file to pick up taxonomy changes.
    """
    return load_species_data()

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Save file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run AI (Now returns a Dictionary with Top 3 Candidates)
    start_time = time.time()
    result = predict_animal(file_path)
    exec_time = int((time.time() - start_time) * 1000)
    
    # 3. Clean up
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 4. Log to Supabase (Background Task)
    if "candidates" in result and len(result["candidates"]) > 0:
        top = result["candidates"][0]
        background_tasks.add_task(
            log_prediction, 
            species=top["name"], 
            confidence=top["score"], 
            family=top["taxonomy"]["family"], 
            class_name=top["taxonomy"]["class"],
            filename=file.filename,
            exec_time=exec_time
        )
    
    # 5. Return the result directly (Frontend expects .data.candidates)
    return result

def log_prediction(species: str, confidence: float, family: str, class_name: str, filename: str, exec_time: int):
    """Logs prediction to Supabase without blocking the main thread."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return

    try:
        url = f"{SUPABASE_URL}/rest/v1/prediction_logs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        payload = {
            "species_prediction": species,
            "confidence": confidence,
            "family": family,
            "class_name": class_name,
            "filename": filename,
            "execution_time_ms": exec_time,
            "created_at": datetime.utcnow().isoformat()       
        }
        requests.post(url, headers=headers, json=payload, timeout=2)
    except Exception as e:
        print(f"⚠️ Logger Error: {e}")


# --- PARK OBSERVATIONS CACHE ---
import requests

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

@app.get("/parks/observations")
def get_park_observations(taxon_id: int):
    """
    Returns cached park observation counts for a given taxon_id.
    If not cached, returns empty and triggers background cache population.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"data": [], "cached": False, "error": "Supabase not configured"}
    
    try:
        url = f"{SUPABASE_URL}/rest/v1/park_observations"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        params = {
            "taxon_id": f"eq.{taxon_id}",
            "observation_count": "gt.0",
            "order": "observation_count.desc",
            "limit": "50"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                return {"data": data, "cached": True, "count": len(data)}
            else:
                # Not cached yet - tell frontend to use live query
                return {"data": [], "cached": False, "message": "Species not cached, use live query"}
        else:
            return {"data": [], "cached": False, "error": f"Supabase error: {response.status_code}"}
    except Exception as e:
        return {"data": [], "cached": False, "error": str(e)}


@app.get("/inat/observations")
def proxy_inat_observations(taxon_id: int, place_id: int = None, lat: float = None, lng: float = None, radius: int = 50):
    """
    Proxy endpoint to query iNaturalist API - bypasses CORS issues.
    Returns observation count for a taxon at a given location or place.
    Prefer place_id for accurate park boundary matching.
    """
    try:
        url = f"https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "per_page": 0
        }
        
        # Prefer place_id for accurate park boundaries
        if place_id:
            params["place_id"] = place_id
        elif lat is not None and lng is not None:
            params["lat"] = lat
            params["lng"] = lng
            params["radius"] = radius
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return {"total_results": data.get("total_results", 0)}
        else:
            return {"total_results": 0, "error": f"iNaturalist error: {response.status_code}"}
    except Exception as e:
        return {"total_results": 0, "error": str(e)}


import time

@app.get("/inat/batch")
def batch_inat_observations(taxon_id: int, place_ids: str):
    """
    Batch query for multiple parks - more efficient and handles rate limiting.
    place_ids should be comma-separated, e.g. "10211,72645,69216"
    Returns observation counts for each place_id.
    """
    results = {}
    
    # Parse comma-separated place_ids
    ids = [int(x.strip()) for x in place_ids.split(",") if x.strip()]
    
    for i, place_id in enumerate(ids):
        try:
            url = f"https://api.inaturalist.org/v1/observations"
            params = {
                "taxon_id": taxon_id,
                "place_id": place_id,
                "quality_grade": "research",
                "per_page": 0
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results[place_id] = data.get("total_results", 0)
            else:
                results[place_id] = 0
                
            # Rate limit: 1 request per 100ms to avoid iNat throttling
            if i < len(ids) - 1:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error fetching place_id {place_id}: {e}")
            results[place_id] = 0
    
    return {"results": results, "count": len(results)}


# --- RECENT SIGHTINGS PROXY (The Solution) ---
import math
import asyncio
import httpx

def get_distance_km(lat1, lng1, lat2, lng2):
    R = 6371  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLng / 2) * math.sin(dLng / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Load parks data once
PARKS_PATH = os.path.join(os.path.dirname(__file__), "../frontend/public/parks.json")
try:
    with open(PARKS_PATH, 'r') as f:
        ALL_PARKS = json.load(f)
except Exception as e:
    print(f"Error loading parks.json: {e}")
    ALL_PARKS = []

@app.get("/parks/recent-sightings")
async def get_recent_sightings(taxon_id: int):
    """
    Fetches recent observations (~600) from iNaturalist and aggregates them by park.
    Solves CORS and Rate Limiting by handling it server-side.
    """
    if not ALL_PARKS:
        return {"error": "Parks data not loaded", "results": []}

    obs_list = []
    
    # Fetch 3 pages in parallel (Async)
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        headers = {
            "User-Agent": "WildlifeID-App/1.0 (contact@wildlife-id.com)", 
            "Accept": "application/json"
        }
        
        urls = [
            f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&per_page=200&page=1&order_by=observed_on&quality_grade=research",
            f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&per_page=200&page=2&order_by=observed_on&quality_grade=research",
            f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&per_page=200&page=3&order_by=observed_on&quality_grade=research"
        ]
        
        # Helper to fetch with simple retry for 429
        async def fetch(url):
            retries = 3
            while retries > 0:
                try:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        return resp.json().get('results', [])
                    elif resp.status_code == 429:
                        print(f"Rate limited {url}, waiting 5s...")
                        await asyncio.sleep(5) 
                        retries -= 1
                    else:
                        print(f"HTTP {resp.status_code} for {url}")
                        return []
                except Exception as e:
                    print(f"Fetch error {url}: {e}")
                    retries -= 1
            return []

        # Sequential fetch to be nice to API
        results = []
        for url in urls:
            res = await fetch(url)
            results.append(res)
            # Small delay between requests
            await asyncio.sleep(1.0)
            
        for res in results:
            obs_list.extend(res)

    # Process Observations against Parks
    park_counts = {}
    park_last_seen = {}

    for obs in obs_list:
        geojson = obs.get('geojson')
        if not geojson or not geojson.get('coordinates'):
            continue
            
        obs_lng = geojson['coordinates'][0]
        obs_lat = geojson['coordinates'][1]
        obs_date = obs.get('observed_on', '')

        # Check distance to all parks (optimized loop)
        for park in ALL_PARKS:
            # Quick bounding box check could go here, but with 250 parks it's fast enough
            dist = get_distance_km(obs_lat, obs_lng, park['lat'], park['lng'])
            
            if dist <= 25: # 25km radius approximation
                pid = park['id']
                if pid not in park_counts:
                    park_counts[pid] = 0
                park_counts[pid] += 1
                
                if pid not in park_last_seen or obs_date > park_last_seen[pid]:
                    park_last_seen[pid] = obs_date

    # Format Output
    output = []
    for pid, count in park_counts.items():
        # Find park metadata
        park = next((p for p in ALL_PARKS if p['id'] == pid), None)
        if park:
            output.append({
                "id": park['id'],
                "name": park['name'],
                "lat": park['lat'],
                "lng": park['lng'],
                "state": park['state'],
                "country": park['country'],
                "observationCount": count,
                "recentObsDate": park_last_seen.get(pid)
            })

    # Sort by count desc
    output.sort(key=lambda x: x['observationCount'], reverse=True)
    
    return {
        "results": output, 
        "total_observations_scanned": len(obs_list),
        "parks_found": len(output)
    }