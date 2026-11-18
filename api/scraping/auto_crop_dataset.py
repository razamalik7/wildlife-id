import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not found. Please install: pip install ultralytics")
    exit()

# CONFIG
SOURCE_DIR = "./training_data"
TARGET_DIR = "./training_data_cropped"
MODEL_NAME = "yolov8m.pt" # Medium model for better accuracy
CONF_THRESHOLD = 0.4
MARGIN_PCT = 0.10 # Add 10% context around the box

# MS COCO Classes that are "Animals"
# 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 
# 20: elephant, 21: bear, 22: zebra, 23: giraffe
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

def main():
    print(f"🦁 INITIALIZING AUTO-CROPPER (YOLOv8) | GPU: {torch.cuda.is_available()}")
    
    # 1. Load Model
    model = YOLO(MODEL_NAME)
    
    # 2. Walk through folders
    source_path = Path(SOURCE_DIR)
    target_path = Path(TARGET_DIR)
    
    if not source_path.exists():
        print(f"❌ Source directory {SOURCE_DIR} not found!")
        return

    # Count files
    all_files = list(source_path.rglob("*.*"))
    img_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"📂 Found {len(img_files)} images to process.")

    processed_count = 0
    cropped_count = 0
    failed_count = 0

    for img_path in tqdm(img_files, desc="Cropping"):
        try:
            # Replicate folder structure
            rel_path = img_path.relative_to(source_path)
            dest_path = target_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run Inference
            results = model(str(img_path), verbose=False, conf=CONF_THRESHOLD)
            result = results[0] # Single image
            
            # Find best animal box
            best_box = None
            max_conf = 0
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Check if it's an animal
                if cls_id in ANIMAL_CLASSES:
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box.xyxy[0].tolist() # x1, y1, x2, y2

            # Use PIL to Crop
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            
            if best_box:
                x1, y1, x2, y2 = best_box
                
                # Add Margin
                bw = x2 - x1
                bh = y2 - y1
                margin_x = bw * MARGIN_PCT
                margin_y = bh * MARGIN_PCT
                
                nx1 = max(0, x1 - margin_x)
                ny1 = max(0, y1 - margin_y)
                nx2 = min(w, x2 + margin_x)
                ny2 = min(h, y2 + margin_y)
                
                # Crop
                img = img.crop((nx1, ny1, nx2, ny2))
                cropped_count += 1
            else:
                # Fallback: Just keep original if no animal found (safe fallback)
                pass

            # Update Metadata? (Not yet, just pixel saving)
            
            # Save Image
            img.save(dest_path, quality=95)
            
            # Copy Metadata (CRITICAL FIX)
            json_src = img_path.with_suffix('.json')
            if json_src.exists():
                json_dest = dest_path.with_suffix('.json')
                shutil.copy2(json_src, json_dest)
            
            processed_count += 1

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            failed_count += 1

    print("\n✅ CROP COMPLETE")
    print(f"   Processed: {processed_count}")
    print(f"   Cropped:   {cropped_count}")
    print(f"   Failed:    {failed_count}")
    print(f"   Output:    {TARGET_DIR}")

if __name__ == "__main__":
    main()
