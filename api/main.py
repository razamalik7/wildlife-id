from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
from ai_engine import predict_animal

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
    """
    """
    Returns the Master List of animals so the frontend can 
    display the Anidex (Locked/Unlocked) and Native/Invasive tabs.
    """
    # The JSON data is already structured perfectly for the frontend
    return SPECIES_DATA

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Save file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run AI (Now returns a Dictionary with Top 3 Candidates)
    result = predict_animal(file_path)
    
    # 3. Clean up
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 4. Return the result directly (Frontend expects .data.candidates)
    return result