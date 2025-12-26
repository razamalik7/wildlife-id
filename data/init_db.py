import sqlite3
import json
import os

def init_db():
    print("🦁 Initializing Wildlife Database...")
    
    # 1. Connect
    db_path = os.path.join(os.path.dirname(__file__), 'wildlife.db') # Explicit path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. Reset Species Table (Force Schema Update)
    cursor.execute('DROP TABLE IF EXISTS species')

    # 3. Create Species Table (New 'iucn' column)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS species (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        scientific_name TEXT,
        fact TEXT,
        native_regions TEXT,
        iucn TEXT
    )
    ''')

    # 4. Create Captures Table (Keep existing data)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS captures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        species_name TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        region TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        invasive_status BOOLEAN,
        image_path TEXT
    )
    ''')

    # 5. Load Data
    config_path = os.path.join(os.path.dirname(__file__), '../api/species_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            
            species_tuples = []
            for entry in data:
                species_tuples.append((
                    entry['name'],
                    entry.get('scientific_name', 'Unknown'),
                    entry.get('description', 'No fact available'),
                    entry.get('origin', 'Unknown'),
                    entry.get('iucn', 'Unknown') # Default if missing
                ))
            
            print(f"📖 Seeding {len(species_tuples)} species into DB...")
            
            cursor.executemany('''
            INSERT INTO species (name, scientific_name, fact, native_regions, iucn)
            VALUES (?, ?, ?, ?, ?)
            ''', species_tuples)
            
            conn.commit()
            print("✅ Database successfully seeded.")
            
    else:
        print(f"❌ Error: Config file not found at {config_path}")

    conn.close()

if __name__ == '__main__':
    init_db()