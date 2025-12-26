
import json
import os
import scrape_targeted

# Update this to point to the correct data directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data_refined')

def main():
    print("🚀 STARTING REFINED SCRAPE")
    print(f"📂 Output Directory: {OUTPUT_DIR}")
    
    # Load targets
    with open('scrape_targets_refined.json', 'r') as f:
        targets = json.load(f)
        
    print(f"📋 Found {len(targets)} classes to scrape")
    
    total_new = 0
    
    # Sort by count descending so we do the big ones first
    sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_targets:
        print(f"\n📸 Processing {class_name} (Target: {count} new images)...")
        
        try:
            # We call the imported function directly
            # scrape_class returns number of DOWNLOADED images
            downloaded = scrape_targeted.scrape_class(class_name, count, OUTPUT_DIR)
            total_new += downloaded
            print(f"   ✅ Added {downloaded} images for {class_name}")
            
        except Exception as e:
            print(f"   ❌ Error scraping {class_name}: {e}")
            
    print(f"\n{'='*60}")
    print(f"🏆 SCRAPE BATCH COMPLETE")
    print(f"   Total new images added: {total_new}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
