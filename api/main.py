from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import time 
import random

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

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    print(f"Server received: {file.filename}")
    
    # 1. Simulate the AI thinking time
    time.sleep(1)

    # 2. Create a "Database" of Dictionaries
    # Each {} is one complete unit of data. They cannot get out of sync.
    wildlife_data = [
        {
            "name": "North American Beaver",
            "fact": "Known for building dams, canals, and lodges.",
            "invasive": False
        },
        {
            "name": "Grizzly Bear",
            "fact": "Can run up to 35 mph despite their large size.",
            "invasive": False
        },
        {
            "name": "European Starling",
            "fact": "An invasive bird that causes damage to crops.",
            "invasive": True 
        },
        {
            "name": "Mountain Lion",
            "fact": "Can leap up to 45 feet horizontally and 18 feet vertically",
            "invasive": False 
        }
    ]

    # 3. Pick ONE object randomly
    # We don't need index numbers anymore. We just grab a whole 'box' of data.
    selection = random.choice(wildlife_data)

    # 4. Return the data from that specific box
    return {
        "success": True,
        "result": selection["name"],
        "confidence": "92%",
        "invasive_status": selection["invasive"],
        "info": selection["fact"]
    }