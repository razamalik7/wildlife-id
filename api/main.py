from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # <--- New import for serving images
from pydantic import BaseModel
import sqlite3
import random
import time
import shutil # <--- New import for saving files
import os
import uuid   # <--- New import for unique filenames
import reverse_geocoder as rg

app = FastAPI()

# 1. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. IMAGE VAULT SETUP
# Create 'uploads' folder if it doesn't exist
os.makedirs("uploads", exist_ok=True) 
# "Mount" it so the frontend can access images at /uploads/filename.jpg
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- DATA MODELS ---
class CaptureRequest(BaseModel):
    species_name: str
    latitude: float
    longitude: float
    region: str
    invasive_status: bool
    image_path: str # <--- Added this field

# --- HELPERS ---
def get_region_from_code(cc):
    north_america = ['US', 'CA', 'MX', 'GT', 'BZ', 'SV', 'HN', 'NI', 'CR', 'PA']
    europe = ['GB', 'FR', 'DE', 'IT', 'ES', 'PT', 'NL', 'BE', 'CH', 'AT', 'SE', 'NO', 'DK', 'FI', 'PL', 'UA', 'RU']
    asia = ['CN', 'JP', 'IN', 'KR', 'ID', 'TH', 'VN', 'MY', 'PH', 'PK', 'BD', 'IR', 'IQ', 'SA', 'IL', 'TR']
    south_america = ['BR', 'AR', 'CO', 'PE', 'VE', 'CL', 'EC', 'BO', 'PY', 'UY', 'GY', 'SR']
    africa = ['NG', 'ET', 'EG', 'CD', 'TZ', 'ZA', 'KE', 'UG', 'DZ', 'SD', 'MA']
    australia = ['AU', 'NZ', 'PG']

    if cc in north_america: return "North America"
    if cc in europe: return "Europe"
    if cc in asia: return "Asia"
    if cc in south_america: return "South America"
    if cc in africa: return "Africa"
    if cc in australia: return "Australia"
    return "Unknown"

# --- ENDPOINTS ---

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    latitude: float = Form(...), 
    longitude: float = Form(...)
):
    print(f"Analyzing file: {file.filename}")
    
    # STEP 1: Save the file to the hard drive
    # Generate unique name (e.g., "a1b2c3d4_bear.jpg")
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = f"uploads/{unique_filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    time.sleep(1) 

    # STEP 2: Geolocation Logic
    coordinates = (latitude, longitude)
    try:
        results = rg.search(coordinates)
        country_code = results[0]['cc']
        user_region = get_region_from_code(country_code)
    except:
        user_region = "Unknown"

    # STEP 3: Check the Database
    conn = sqlite3.connect('../wildlife.db') 
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM species ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row: return {"success": False, "error": "DB Empty"}

    is_native = user_region in row["native_regions"]
    
    return {
        "success": True,
        "result": row["name"],
        "confidence": "92%",
        "invasive_status": not is_native,
        "info": row["fact"],
        "detected_region": user_region,
        "image_path": unique_filename # <--- Send filename back to frontend
    }

@app.post("/api/capture")
def save_capture(capture: CaptureRequest):
    conn = sqlite3.connect('../wildlife.db')
    cursor = conn.cursor()
    
    # Save the new capture WITH the image path
    cursor.execute('''
        INSERT INTO captures (species_name, latitude, longitude, region, invasive_status, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (capture.species_name, capture.latitude, capture.longitude, capture.region, capture.invasive_status, capture.image_path))
    
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/anidex") 
def get_anidex():
    conn = sqlite3.connect('../wildlife.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get basic species info (The "Slots")
    cursor.execute("SELECT * FROM species ORDER BY name")
    all_species = [dict(row) for row in cursor.fetchall()]
    
    # Get user history (The "Stickers")
    cursor.execute("SELECT * FROM captures ORDER BY timestamp DESC")
    all_captures = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    # Merge them to create the "Locked/Unlocked" status
    anidex = [] 
    for animal in all_species:
        # Find all sightings for this specific animal
        my_sightings = [c for c in all_captures if c['species_name'] == animal['name']]
        
        anidex.append({
            "id": animal['id'],
            "name": animal['name'],
            "fact": animal['fact'],
            "unlocked": len(my_sightings) > 0, # True if we have found it at least once
            "sightings": my_sightings 
        })
        
    return {"anidex": anidex}