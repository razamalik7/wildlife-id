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

def get_resnet(num_classes):
    """Rebuilds the ResNet50 Architecture"""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# --- 2. LOAD THE BRAINS ---

model_b3 = None
model_resnet = None
class_names = []

# Load Elder 1: EfficientNet B3
if os.path.exists("wildlife_model_b3.pth"):
    try:
        ckpt = torch.load("wildlife_model_b3.pth", map_location=device)
        class_names = ckpt['class_names']
        model_b3 = get_b3(len(class_names))
        model_b3.load_state_dict(ckpt['model_state_dict'])
        model_b3.to(device)
        model_b3.eval()
        print("✅ Elder 1 (EfficientNet B3) Ready")
    except Exception as e:
        print(f"❌ Failed to load B3: {e}")

# Load Elder 2: ResNet50
if os.path.exists("wildlife_model_resnet.pth"):
    try:
        ckpt = torch.load("wildlife_model_resnet.pth", map_location=device)
        # We assume class_names are identical since they trained on same data
        if not class_names: class_names = ckpt['class_names'] 
        
        model_resnet = get_resnet(len(class_names))
        model_resnet.load_state_dict(ckpt['model_state_dict'])
        model_resnet.to(device)
        model_resnet.eval()
        print("✅ Elder 2 (ResNet50) Ready")
    except Exception as e:
        print(f"❌ Failed to load ResNet: {e}")

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

def predict_animal(image_path):
    if model_b3 is None and model_resnet is None:
        return {"error": "AI Not Ready"}
    
    try:
        img = Image.open(image_path).convert('RGB')
        
        # --- ASK ELDER 1 (B3) ---
        probs_b3 = None
        if model_b3:
            input_b3 = trans_b3(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_b3(input_b3)
                probs_b3 = torch.nn.functional.softmax(out[0], dim=0)

        # --- ASK ELDER 2 (ResNet) ---
        probs_resnet = None
        if model_resnet:
            input_res = trans_resnet(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_resnet(input_res)
                probs_resnet = torch.nn.functional.softmax(out[0], dim=0)

        # --- THE COUNCIL VOTES ---
        if probs_b3 is not None and probs_resnet is not None:
            final_probs = (probs_b3 + probs_resnet) / 2
        elif probs_b3 is not None:
            final_probs = probs_b3
        else:
            final_probs = probs_resnet
            
        # --- GET TOP 3 CANDIDATES ---
        # We grab the top 3 highest probabilities
        top_probs, top_idxs = torch.topk(final_probs, 3)
        
        candidates = []
        for i in range(3):
            score = top_probs[i].item() * 100
            name = class_names[top_idxs[i].item()].replace('_', ' ').title()
            candidates.append({"name": name, "score": score})
            
        print(f"🦁 Top 3 Candidates: {candidates}")
        
        return {"candidates": candidates}

    except Exception as e:
        print(f"Inference Error: {e}")
        return {"error": str(e)}