
import os
import glob
import json
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS_FILE = os.path.join(BASE_DIR, 'scrape_targets_refined.json')
REFINED_DIR = os.path.join(BASE_DIR, 'training_data_refined', 'train')

def trim_all_excess():
    if not os.path.exists(TARGETS_FILE):
        print("❌ Targets file not found.")
        return

    with open(TARGETS_FILE, 'r') as f:
        targets = json.load(f)

    print(f"🔍 Verifying {len(targets)} classes in {REFINED_DIR}...")
    
    total_trimmed = 0
    
    for class_name, target_count in targets.items():
        class_dir = os.path.join(REFINED_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Get all jpgs
        files = glob.glob(os.path.join(class_dir, "*.*")) # Check all files just in case
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)
        
        if current_count > target_count:
            excess = current_count - target_count
            print(f"⚠️  {class_name}: Found {current_count}, Target {target_count}. Trimming {excess}...")
            
            # Sort by modification time (Newest last)
            # We want to keep the oldest (original valid run) and delete the newest (duplicate run)
            images.sort(key=os.path.getmtime)
            
            to_delete = images[target_count:]
            
            for img_path in to_delete:
                try:
                    os.remove(img_path)
                    # Try removing associated json
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    if os.path.exists(json_path):
                        os.remove(json_path)
                except Exception as e:
                    print(f"   Error deleting {img_path}: {e}")
            
            total_trimmed += excess
        else:
            # specific check for user peace of mind
            if current_count > 0:
                print(f"✅ {class_name}: {current_count}/{target_count} (OK)")

    print(f"\n✨ Verification Complete. Total files trimmed: {total_trimmed}")

if __name__ == "__main__":
    trim_all_excess()
