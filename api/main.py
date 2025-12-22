from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
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

# --- THE MASTER DATABASE ---
# Format: (Common Name, Scientific Name, Fact, Origin)
RAW_SPECIES_DATA = [
    # Large Mammals
    ("American Black Bear", "Ursus americanus", "Excellent tree climbers, often found in forests.", "North America"),
    ("Grizzly Bear", "Ursus arctos horribilis", "Identifiable by the distinct hump on their shoulders.", "North America"),
    ("Moose", "Alces alces", "The largest member of the deer family.", "North America, Europe, Asia"),
    ("White-tailed Deer", "Odocoileus virginianus", "Raises its tail to show a white 'flag' when alarmed.", "North America, South America"),
    ("American Bison", "Bison bison", "The national mammal of the United States.", "North America"),
    ("Mountain Lion", "Puma concolor", "Also known as cougar, puma, or panther.", "North America, South America"),
    ("Coyote", "Canis latrans", "Highly adaptable and can be found in every US state except Hawaii.", "North America"),
    ("Bobcat", "Lynx rufus", "Named for their short, 'bobbed' tail.", "North America"),
    ("Gray Wolf", "Canis lupus", "The largest member of the dog family and a social animal that lives in packs.", "North America, Europe, Asia"),

    # Small/Medium Mammals
    ("Raccoon", "Procyon lotor", "Known for its dexterity and 'masked' face.", "North America"),
    ("North American Beaver", "Castor canadensis", "Their dams create wetlands that support thousands of other species.", "North America"),
    ("Striped Skunk", "Mephitis mephitis", "Can spray its defensive musk up to 10 feet.", "North America"),
    ("Virginia Opossum", "Didelphis virginiana", "The only marsupial (pouched mammal) found north of Mexico.", "North America"),
    ("Eastern Gray Squirrel", "Sciurus carolinensis", "Plays a crucial role in forest regeneration by burying nuts.", "North America"),
    ("Red Fox", "Vulpes vulpes", "The largest of the true foxes and highly adaptable to urban environments.", "North America, Europe, Asia, North Africa"),

    # Birds
    ("Bald Eagle", "Haliaeetus leucocephalus", "Builds the largest nest of any North American bird.", "North America"),
    ("Red-tailed Hawk", "Buteo jamaicensis", "Known for its rasping scream, often used in movies as a generic eagle sound.", "North America"),
    ("Great Blue Heron", "Ardea herodias", "Hunts by standing motionless in shallow water waiting for fish.", "North America"),
    ("Wild Turkey", "Meleagris gallopavo", "Benjamin Franklin famously preferred this bird over the Bald Eagle.", "North America"),
    ("Canada Goose", "Branta canadensis", "Famous for their V-shaped flying formation.", "North America, Europe"),

    # Reptiles/Amphibians
    ("American Alligator", "Alligator mississippiensis", "Have been around for about 37 million years.", "North America"),
    ("Eastern Box Turtle", "Terrapene carolina", "Can live for over 100 years in the wild.", "North America"),
    ("American Crocodile", "Crocodylus acutus", "Unlike the alligator's U-shaped snout, crocodiles have a V-shaped snout and visible lower teeth.", "North America, South America, Central America"),

    # --- INVASIVE SPECIES ---
    ("Burmese Python", "Python bivittatus", "Massive constrictors decimating mammals in the Florida Everglades.", "Asia"),
    ("Green Iguana", "Iguana iguana", "Cause damage to infrastructure by digging burrows; thrive in Florida.", "South America, Central America"),
    ("Argentine Black and White Tegu", "Salvator merianae", "Large lizard that eats the eggs of native birds and reptiles.", "South America"),
    ("Cane Toad", "Rhinella marina", "Toxic amphibian that kills pets and native predators who try to eat it.", "South America, Australia"),
    ("European Starling", "Sturnus vulgaris", "Introduced to NY in 1890; huge flocks damage crops and airplanes.", "Europe, Asia"),
    ("House Sparrow", "Passer domesticus", "Aggressively competes with native birds for nesting cavities.", "Europe, Asia"),
    ("Rock Pigeon", "Columba livia", "The common city pigeon; carries diseases and damages buildings.", "Europe, Asia, Africa"),
    ("Monk Parakeet", "Myiopsitta monachus", "Builds massive communal nests on power lines, causing outages.", "South America"),
    ("Wild Boar", "Sus scrofa", "Highly destructive to crops and native habitats due to rooting behavior.", "Europe, Asia, North Africa"),
    ("Nutria", "Myocastor coypus", "Large rodent that destroys wetlands; looks like a beaver with a rat tail.", "South America")
]

# Create a set of invasive names for easy lookup
INVASIVE_NAMES = {
    "Burmese Python", "Green Iguana", "Argentine Black and White Tegu", "Cane Toad",
    "European Starling", "House Sparrow", "Rock Pigeon", "Monk Parakeet", 
    "Wild Boar", "Nutria"
}

@app.get("/")
def read_root():
    return {"Status": "Wildlife AI is Online 🟢"}

@app.get("/species")
def get_all_species():
    """
    Returns the Master List of animals so the frontend can 
    display the Anidex (Locked/Unlocked) and Native/Invasive tabs.
    """
    structured_data = []
    
    for entry in RAW_SPECIES_DATA:
        common_name = entry[0]
        # Determine category based on our invasive list
        category = "Invasive" if common_name in INVASIVE_NAMES else "Native"
        
        structured_data.append({
            "name": common_name,
            "scientific_name": entry[1],
            "description": entry[2],
            "origin": entry[3],
            "category": category
        })
        
    return structured_data

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