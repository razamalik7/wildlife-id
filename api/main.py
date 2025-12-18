from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import time
import sqlite3
import reverse_geocoder as rg
from pydantic import BaseModel

# This describes the JSON data the frontend will send us
class CaptureRequest(BaseModel):
    species_name: str
    latitude: float
    longitude: float
    region: str
    invasive_status: bool

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

def get_region_from_code(cc):
    # This is a simple map. In a real app, you'd use a full library like pycountry_convert
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

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    latitude: float = Form(...),  # <--- Changed from location: str
    longitude: float = Form(...)  # <--- Now accepts numbers
):
    print(f"Received file: {file.filename} at {latitude}, {longitude}")
    time.sleep(1) 
    
    # 1. Reverse Geocode: Turn (Lat, Lng) -> Country Code (e.g., "US")
    coordinates = (latitude, longitude)
    results = rg.search(coordinates) # Returns a list of dicts
    country_code = results[0]['cc']  # Get the 'cc' (e.g., 'US')
    
    # 2. Convert Country Code -> Continent String
    user_region = get_region_from_code(country_code)
    print(f"Detected Region: {user_region}")

    # 3. Connect to Database
    conn = sqlite3.connect('../wildlife.db') 
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()

    # 4. Pick a random animal
    cursor.execute("SELECT * FROM species ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"success": False, "error": "Database is empty!"}

    # 5. The Logic: Check Invasive Status
    # We compare the Detected Region (e.g., "North America") against the Native Regions
    is_native = user_region in row["native_regions"]
    
    return {
        "success": True,
        "result": row["name"],
        "confidence": "92%",
        "invasive_status": not is_native, # True if NOT native
        "info": row["fact"],
        "detected_region": user_region # Optional: Send back where we think they are
    }

@app.post("/api/capture")
def save_capture(capture: CaptureRequest):
    conn = sqlite3.connect('../wildlife.db')
    cursor = conn.cursor()
    
    print(f"Saving capture: {capture.species_name} in {capture.region}")
    
    cursor.execute('''
        INSERT INTO captures (species_name, latitude, longitude, region, invasive_status)
        VALUES (?, ?, ?, ?, ?)
    ''', (capture.species_name, capture.latitude, capture.longitude, capture.region, capture.invasive_status))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "message": "Captured!"}