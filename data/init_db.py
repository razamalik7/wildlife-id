import sqlite3
import json
import os

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
    config_path = os.path.join(os.path.dirname(__file__), '../api/species_config.json')
    
    animals = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            # Convert list of dicts to list of tuples for SQL
            for entry in data:
                animals.append((
                    entry['name'],
                    entry['scientific_name'],
                    entry['description'],
                    entry['origin']
                ))
    else:
        print(f"Error: Could not find config at {config_path}")
        return

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS captures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        species_name TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        region TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        invasive_status BOOLEAN,
        image_path TEXT  -- <--- NEW COLUMN
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

    