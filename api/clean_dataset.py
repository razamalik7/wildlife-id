import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import shutil
from PIL import Image

# CONFIG
DATA_DIR = 'training_data'
MODEL_PATH = 'wildlife_model_resnet.pth'
CONFIDENCE_THRESHOLD = 0.05 # If model is less than 5% sure it's the right class, it's garbage.
QUARANTINE_DIR = 'quarantine'

def get_model(num_classes, device):
    # Load the ResNet we just trained
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    return model

def clean_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧹 STARTING CLEANUP on {device}...")
    
    # 1. Setup Transforms (Validation/Inference mode only)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Get Classes
    train_dir = os.path.join(DATA_DIR, 'train')
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # 3. Load Model
    print("Loading Judge Model (ResNet)...")
    model = get_model(len(classes), device)
    
    # 4. Iterate and Judge
    if not os.path.exists(QUARANTINE_DIR):
        os.makedirs(QUARANTINE_DIR)
        
    removed_count = 0
    total_images = 0
    
    print(f"Scanning {len(classes)} classes...")
    
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        target_idx = class_to_idx[class_name]
        
        images = os.listdir(class_dir)
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # Load Image
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                # Check confidence of the CORRECT class
                correct_prob = probs[target_idx].item()
                
                total_images += 1
                
                if correct_prob < CONFIDENCE_THRESHOLD:
                    # HE IS GUILTY!
                    print(f"❌ REJECTED: {class_name}/{img_name} (Conf: {correct_prob:.4f})")
                    
                    # Move to quarantine
                    quarantine_class_dir = os.path.join(QUARANTINE_DIR, class_name)
                    if not os.path.exists(quarantine_class_dir):
                        os.makedirs(quarantine_class_dir)
                        
                    shutil.move(img_path, os.path.join(quarantine_class_dir, img_name))
                    removed_count += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    print("-" * 30)
    print(f"✅ CLEANUP COMPLETE")
    print(f"Total Scanned: {total_images}")
    print(f"Removed: {removed_count}")
    print(f"Quarantined images are in: {QUARANTINE_DIR}")

if __name__ == "__main__":
    clean_data()
