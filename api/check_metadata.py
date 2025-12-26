import os
import json

train_dir = './training_data_cropped/train'

total_images = 0
images_with_json = 0
images_with_location = 0
images_with_date = 0

for cls in os.listdir(train_dir):
    cls_dir = os.path.join(train_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    
    for f in os.listdir(cls_dir):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        total_images += 1
        
        # Check for JSON sidecar
        json_path = os.path.splitext(os.path.join(cls_dir, f))[0] + '.json'
        if os.path.exists(json_path):
            images_with_json += 1
            try:
                with open(json_path, 'r') as jf:
                    meta = json.load(jf)
                    lat = meta.get('lat')  # Correct key from scraper
                    lng = meta.get('lng')  # Correct key from scraper
                    date = meta.get('date')  # Correct key from scraper
                    
                    if lat and lng and lat != 0 and lng != 0:
                        images_with_location += 1
                    if date:
                        images_with_date += 1
            except:
                pass

print(f"📊 METADATA ANALYSIS")
print(f"   Total images: {total_images}")
print(f"   With JSON sidecar: {images_with_json} ({images_with_json/total_images*100:.1f}%)")
print(f"   With valid location: {images_with_location} ({images_with_location/total_images*100:.1f}%)")
print(f"   With observation date: {images_with_date} ({images_with_date/total_images*100:.1f}%)")
