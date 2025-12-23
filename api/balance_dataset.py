import os
import shutil
import random

DATA_DIR = 'training_data'
TARGET_COUNT = 0  # Will automatically find the max count in your folder

def balance_classes():
    # 1. Find the target count (The max number of images in any folder)
    max_count = 0
    train_dir = os.path.join(DATA_DIR, 'train')
    
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print("Analyzing dataset...")
    for class_name in classes:
        folder_path = os.path.join(train_dir, class_name)
        count = len(os.listdir(folder_path))
        if count > max_count:
            max_count = count
            
    print(f"🎯 Target Count per Class: {max_count}")
    
    # 2. Duplicate images in small folders
    for class_name in classes:
        folder_path = os.path.join(train_dir, class_name)
        images = os.listdir(folder_path)
        current_count = len(images)
        
        if current_count < max_count:
            needed = max_count - current_count
            print(f"  ⚖️ Balancing {class_name}: Needs {needed} more...")
            
            for i in range(needed):
                # Pick a random image to clone
                src_image = random.choice(images)
                src_path = os.path.join(folder_path, src_image)
                
                # Create a clone name
                name, ext = os.path.splitext(src_image)
                dst_name = f"{name}_clone_{i}{ext}"
                dst_path = os.path.join(folder_path, dst_name)
                
                shutil.copy(src_path, dst_path)
                
                # Clone Metadata (CRITICAL for Grandmaster)
                # If src_image matches "001.jpg", json is "001.json"
                src_json = os.path.splitext(src_path)[0] + ".json"
                dst_json = os.path.splitext(dst_path)[0] + ".json"
                
                if os.path.exists(src_json):
                    shutil.copy(src_json, dst_json)
                
    print("✅ Dataset Balanced! All classes have equal weight.")

if __name__ == "__main__":
    balance_classes()