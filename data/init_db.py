import sqlite3
import json

def init_db():
    # 1. Connect to the database (it creates the file if it doesn't exist)
    conn = sqlite3.connect('wildlife.db')
    cursor = conn.cursor()

    # 2. Create the Table (The "Excel Sheet")
    # We store native_regions as a simple text string "North America, Europe"
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS species (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        scientific_name TEXT,
        fact TEXT,
        native_regions TEXT
    )
    ''')

    # 3. The Data (Our Encyclopedia)
    animals = [
        # --- NATIVES (20) ---
        # Large Mammals
        ("American Black Bear", "Ursus americanus", "Excellent tree climbers, often found in forests.", "North America"),
        ("Grizzly Bear", "Ursus arctos horribilis", "Identifiable by the distinct hump on their shoulders.", "North America"),
        ("Moose", "Alces alces", "The largest member of the deer family.", "North America, Europe, Asia"),
        ("White-tailed Deer", "Odocoileus virginianus", "Raises its tail to show a white 'flag' when alarmed.", "North America, South America"),
        ("American Bison", "Bison bison", "The national mammal of the United States.", "North America"),
        ("Mountain Lion", "Puma concolor", "Also known as cougar, puma, or panther.", "North America, South America"),
        ("Coyote", "Canis latrans", "Highly adaptable and can be found in every US state except Hawaii.", "North America"),
        ("Bobcat", "Lynx rufus", "Named for their short, 'bobbed' tail.", "North America"),

        # Small/Medium Mammals
        ("Raccoon", "Procyon lotor", "Known for its dexterity and 'masked' face.", "North America"),
        ("North American Beaver", "Castor canadensis", "Their dams create wetlands that support thousands of other species.", "North America"),
        ("Striped Skunk", "Mephitis mephitis", "Can spray its defensive musk up to 10 feet.", "North America"),
        ("Virginia Opossum", "Didelphis virginiana", "The only marsupial (pouched mammal) found north of Mexico.", "North America"),
        ("Eastern Gray Squirrel", "Sciurus carolinensis", "Plays a crucial role in forest regeneration by burying nuts.", "North America"),

        # Birds
        ("Bald Eagle", "Haliaeetus leucocephalus", "Builds the largest nest of any North American bird.", "North America"),
        ("Red-tailed Hawk", "Buteo jamaicensis", "Known for its rasping scream, often used in movies as a generic eagle sound.", "North America"),
        ("Great Blue Heron", "Ardea herodias", "Hunts by standing motionless in shallow water waiting for fish.", "North America"),
        ("Wild Turkey", "Meleagris gallopavo", "Benjamin Franklin famously preferred this bird over the Bald Eagle.", "North America"),
        ("Canada Goose", "Branta canadensis", "Famous for their V-shaped flying formation.", "North America, Europe"),

        # Reptiles/Amphibians
        ("American Alligator", "Alligator mississippiensis", "Have been around for about 37 million years.", "North America"),
        ("Eastern Box Turtle", "Terrapene carolina", "Can live for over 100 years in the wild.", "North America"),

        # --- INVASIVE VERTEBRATES (10) --- 
        # These are all land-based animals introduced to North America.
        
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

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS captures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        species_name TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        region TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        invasive_status BOOLEAN
    )
    ''')

    # 4. Insert the data (only if table is empty to avoid duplicates)
    cursor.execute('SELECT count(*) FROM species')
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("Seeding database with animals...")
        cursor.executemany('''
            INSERT INTO species (name, scientific_name, fact, native_regions)
            VALUES (?, ?, ?, ?)
        ''', animals)
        conn.commit()
    else:
        print("Database already has data.")

    conn.close()
    print("Database initialized successfully!")

if __name__ == '__main__':
    init_db()

    