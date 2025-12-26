"""
Debug version of scraper with verbose logging
"""
import os
import json
import time
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Load config
with open('species_config.json') as f:
    config = json.load(f)

moose = [e for e in config if 'moose' in e['name'].lower()][0]
taxon_id = moose['taxonomy']['taxon_id']

print(f"Testing moose (taxon_id: {taxon_id})")
print(f"YOLO available: {YOLO_AVAILABLE}")

# Load YOLO if available
yolo_model = None
if YOLO_AVAILABLE:
    print("Loading YOLO...")
    yolo_model = YOLO('yolov8m.pt')

# Fetch observations
print("\nFetching observations...")
r = requests.get(
    'https://api.inaturalist.org/v1/observations',
    params={
        'taxon_id': taxon_id,
        'quality_grade': 'research',
        'photos': 'true',
        'per_page': 10,
        'iconic_taxa': 'Animalia'
    }
)

results = r.json().get('results', [])
print(f"Got {len(results)} observations\n")

def calculate_blur_score(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_dead_animal(obs):
    dead_keywords = ['dead', 'roadkill', 'deceased', 'carcass', 'mortality']
    description = (obs.get('description') or '').lower()
    for keyword in dead_keywords:
        if keyword in description:
            return True
    return False

accepted = 0
for i, obs in enumerate(results[:5]):
    print(f"\n{'='*60}")
    print(f"Observation {i+1}/{min(5, len(results))} (ID: {obs['id']})")
    print(f"{'='*60}")
    
    # Check dead animal
    if is_dead_animal(obs):
        print("❌ REJECTED: Dead animal mention")
        continue
    
    # Get photo
    photos = obs.get('photos', [])
    if not photos:
        print("❌ REJECTED: No photos")
        continue
    
    photo_url = photos[0]['url']
    print(f"Photo URL: {photo_url}")
    
    # Download
    url = photo_url.replace('square', 'medium')
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"❌ REJECTED: HTTP {response.status_code}")
            continue
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✓ Downloaded: {image.size}")
        
        # Check original size
        if min(image.size) < 100:
            print(f"❌ REJECTED: Original too small ({min(image.size)} < 100)")
            continue
        
        # YOLO crop
        if yolo_model:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            yolo_results = yolo_model(img_cv, verbose=False, conf=0.10)
            any_detection = len(yolo_results[0].boxes) > 0
            
            wildlife_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            best_box = None
            
            for box in yolo_results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in wildlife_classes:
                    best_box = box.xyxy[0].tolist()
                    break
            
            print(f"YOLO: any_detection={any_detection}, animal_box={best_box is not None}")
            
            if best_box:
                # Crop
                x1, y1, x2, y2 = best_box
                margin = 0.15
                bw, bh = x2-x1, y2-y1
                x1 = max(0, int(x1 - bw*margin))
                y1 = max(0, int(y1 - bh*margin))
                x2 = min(image.size[0], int(x2 + bw*margin))
                y2 = min(image.size[1], int(y2 + bh*margin))
                cropped = image.crop((x1, y1, x2, y2))
                print(f"✓ Cropped to animal: {cropped.size}")
            elif not any_detection and min(image.size) < 400:
                print(f"❌ REJECTED: Small ({min(image.size)}) with no YOLO detection")
                continue
            else:
                cropped = image
                print(f"✓ Kept full image (no crop): {cropped.size}")
        else:
            cropped = image
            print(f"✓ No YOLO, kept full image")
        
        # Check blur
        blur = calculate_blur_score(cropped)
        print(f"Blur score: {blur:.1f}")
        
        # Check size
        size = min(cropped.size)
        print(f"Final size: {size}px")
        
        # Determine tier
        if blur >= 200 and size >= 100:
            tier = "HIGH_QUALITY"
        elif blur >= 80 and size >= 50:
            tier = "CHALLENGING"
        else:
            print(f"❌ REJECTED: Blur {blur:.1f} or size {size} below thresholds")
            continue
        
        print(f"✅ ACCEPTED: {tier}")
        accepted += 1
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        continue

print(f"\n{'='*60}")
print(f"SUMMARY: Accepted {accepted}/{min(5, len(results))} images")
print(f"{'='*60}")
