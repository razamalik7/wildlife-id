import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

print("--- 🦁 AI Engine: COUNCIL OF ELDERS (Ensemble) Initializing ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SETUP MODEL ARCHITECTURES ---

def get_b3(num_classes):
    """Rebuilds the EfficientNet B3 Architecture"""
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_convnext(num_classes):
    """Rebuilds the ConvNeXt Tiny Architecture (The New Hero)"""
    model = models.convnext_tiny(weights=None)
    # ConvNeXt classifier head structure
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# --- 2. LOAD THE BRAINS ---

model_b3 = None
model_hero = None
class_names = []

# Load Elder 1: EfficientNet B3 (Hero Version)
if os.path.exists("wildlife_model_b3_hero.pth"):
    try:
        ckpt = torch.load("wildlife_model_b3_hero.pth", map_location=device)
        class_names = ckpt['class_names']
        model_b3 = get_b3(len(class_names))
        model_b3.load_state_dict(ckpt['model_state_dict'])
        model_b3.to(device)
        model_b3.eval()
        print("✅ Elder 1 (EfficientNet B3 Hero) Ready")
    except Exception as e:
        print(f"❌ Failed to load B3 Hero: {e}")

# Load Elder 2: ConvNeXt (Hero Version)
if os.path.exists("wildlife_model_hero.pth"):
    try:
        ckpt = torch.load("wildlife_model_hero.pth", map_location=device)
        # We assume class_names are identical since they trained on same data
        if not class_names: class_names = ckpt['class_names'] 
        
        model_hero = get_convnext(len(class_names))
        model_hero.load_state_dict(ckpt['model_state_dict'])
        model_hero.to(device)
        model_hero.eval()
        print("✅ Elder 2 (ConvNeXt Hero) Ready")
    except Exception as e:
        print(f"❌ Failed to load ConvNeXt Hero: {e}")

# --- 3. DEFINE TRANSFORMS (Each needs its own!) ---

# B3 likes 300x300
trans_b3 = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ResNet likes 224x224
trans_resnet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. PREDICTION LOGIC ---

    try:
        img = Image.open(image_path).convert('RGB')
        
        # --- TTA PREPARATION (TenCrop: 4 corners + center, plus flips) ---
        # We use a larger resize to get good crops
        tta_transforms = transforms.Compose([
            transforms.Resize(320), # Resize slightly larger
            transforms.TenCrop(300), # Get 10 images of size 300x300
            # We need to manually stack them into a batch
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
        ])
        
        # For ConvNeXt (224), we need a separate crop size
        tta_transforms_hero = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
        ])

        print("\n🔎 --- COUNCIL SESSION STARTED ---")

        # --- ASK ELDER 1 (B3) ---
        probs_b3 = None
        if model_b3:
            # Shape: [10, 3, 300, 300]
            inputs = tta_transforms(img).to(device)
            with torch.no_grad():
                out = model_b3(inputs) # [10, num_classes]
                batch_probs = torch.nn.functional.softmax(out, dim=1)
                probs_b3 = torch.mean(batch_probs, dim=0) # Average of 10 views
                
                # Verbose Voting Log
                print("🦉 EfficientNet B3 (10 Views):")
                _, top_idxs = torch.max(batch_probs, 1)
                votes = [class_names[idx].replace('_', ' ') for idx in top_idxs]
                from collections import Counter
                print(f"    Votes: {dict(Counter(votes))}")

        # --- ASK ELDER 2 (ConvNeXt) ---
        probs_hero = None
        if model_hero:
            inputs = tta_transforms_hero(img).to(device)
            with torch.no_grad():
                out = model_hero(inputs)
                batch_probs = torch.nn.functional.softmax(out, dim=1)
                probs_hero = torch.mean(batch_probs, dim=0)
                
                # Verbose Voting Log
                print("🦊 ConvNeXt Tiny (10 Views):")
                _, top_idxs = torch.max(batch_probs, 1)
                votes = [class_names[idx].replace('_', ' ') for idx in top_idxs]
                from collections import Counter
                print(f"    Votes: {dict(Counter(votes))}")

        # --- THE COUNCIL VOTES ---
        if probs_b3 is not None and probs_hero is not None:
            final_probs = (probs_b3 + probs_hero) / 2
        elif probs_b3 is not None:
            final_probs = probs_b3
        elif probs_hero is not None:
            final_probs = probs_hero
            
        # --- GET TOP 3 CANDIDATES ---
        top_probs, top_idxs = torch.topk(final_probs, 3)
        
        candidates = []
        for i in range(3):
            score = top_probs[i].item() * 100
            name = class_names[top_idxs[i].item()].replace('_', ' ').title()
            candidates.append({"name": name, "score": score})
            
        print(f"🏆 Final Decision: {candidates[0]['name']} ({candidates[0]['score']:.1f}%)")
        print(f"    Runner-ups: {candidates[1]['name']} ({candidates[1]['score']:.1f}%), {candidates[2]['name']} ({candidates[2]['score']:.1f}%)")
        print("-------------------------------")
        
        return {"candidates": candidates}

    except Exception as e:
        print(f"Inference Error: {e}")
        return {"error": str(e)}