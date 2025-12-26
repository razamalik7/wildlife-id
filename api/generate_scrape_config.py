
import re
import json
from collections import defaultdict

def main():
    print("generating scrape targets...")
    
    # 1. Parse Report
    pairs = []
    with open('final_report.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_reading = False
    for line in lines:
        if "PAIR (A <-> B)" in line:
            start_reading = True
            continue
        if not start_reading:
            continue
        if "-------" in line:
            continue
        if not line.strip():
            continue
            
        # Parse line: "brown_anole <-> green_anole | 10 | ..."
        parts = line.split('|')
        if len(parts) < 2: 
            continue
            
        names = parts[0].strip().split('<->')
        if len(names) != 2:
            continue
            
        name_a = names[0].strip()
        name_b = names[1].strip()
        try:
            count = int(parts[1].strip())
        except:
            continue
            
        if count >= 5:
            pairs.append((name_a, name_b, count))

    # 2. Aggregate Scores
    # If a class is in multiple pairs, sum the errors
    class_errors = defaultdict(int)
    for a, b, count in pairs:
        class_errors[a] += count
        class_errors[b] += count
        
    # 3. Calculate Targets
    # Multiplier: 50 images per error count
    # Min cap: 200, Max cap: 1000 ? Or just let it fly?
    # User said "proportional". 
    # Wolf: 18 errors -> 900 images. Seems okay.
    # Anole: 10 errors -> 500 images.
    # Badger: 5 errors -> 250 images.
    
    MULTIPLIER = 50
    
    scrape_config = {}
    print(f"\n🎯 Scrape Plan (>= 5 confusions, {MULTIPLIER}x multiplier):")
    
    sorted_classes = sorted(class_errors.items(), key=lambda x: x[1], reverse=True)
    
    for cls, errors in sorted_classes:
        target = errors * MULTIPLIER
        scrape_config[cls] = target
        print(f"   {cls:<30} | Errors: {errors:<3} | Target: {target}")
        
    # Save
    with open('scrape_targets.json', 'w') as f:
        json.dump(scrape_config, f, indent=2)
        
    print(f"\n✅ Configuration saved to scrape_targets.json ({len(scrape_config)} classes)")

if __name__ == "__main__":
    main()
