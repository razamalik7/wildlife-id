import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import shutil
from PIL import Image

# CONFIG
# Updated to v2 (Oogway era)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data_v2')
MODEL_PATH = 'wildlife_model_resnet.pth'
CONFIDENCE_THRESHOLD = 0.05 # If model is less than 5% sure it's the right class, it's garbage.
QUARANTINE_DIR = 'quarantine'

def get_model(num_classes, device):
    # Load the B3 Hero model - it has classifier with dropout + 2 linear layers
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    
    # Recreate Hero's classifier structure
    model.classifier = nn.Sequential(
        model.classifier[0],  # Dropout
        nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),  # Extra layer in Hero
            nn.ReLU(),
            nn.Linear(num_ftrs, num_classes)
        )
    )
    
    ckpt = torch.load('wildlife_model_b3_hero.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
        
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
    print("Loading Judge Model (B3 Hero)...")
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
