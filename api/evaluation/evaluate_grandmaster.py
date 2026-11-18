"""
Grandmaster Model Evaluator
- Supports Late Fusion architecture
- Supports ensemble evaluation
- Supports TTA (Test-Time Augmentation)
- Uses cropped validation data with metadata
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import json
import math
from datetime import datetime
import numpy as np
import argparse
from tqdm import tqdm

# Config
DATA_DIR = './training_data_cropped'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DATASET WITH METADATA ---
class WildlifeGeoDataset(Dataset):
    """Loads images + metadata for evaluation."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load Metadata
        lat, lng, month = 0.0, 0.0, 6
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    lat = meta.get('lat', 0.0) or 0.0
                    lng = meta.get('lng', 0.0) or 0.0
                    date_str = meta.get('date', '')
                    if date_str:
                        try:
                            month = datetime.strptime(date_str, '%Y-%m-%d').month
                        except:
                            month = 6
            except:
                pass
        
        meta_vector = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32)
        
        return image, meta_vector, label


# --- 2. LATE FUSION MODEL ---
class WildlifeLateFusion(nn.Module):
    def __init__(self, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            self.image_model = models.efficientnet_b3(weights=None)
            in_features = self.image_model.classifier[1].in_features
            self.image_model.classifier[1] = nn.Linear(in_features, num_classes)
            
        elif model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights=None)
            in_features = self.image_model.classifier[2].in_features
            self.image_model.classifier[2] = nn.Linear(in_features, num_classes)
            
        elif model_type == 'resnet':
            self.image_model = models.resnet50(weights=None)
            in_features = self.image_model.fc.in_features
            self.image_model.fc = nn.Linear(in_features, num_classes)
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, meta):
        image_logits = self.image_model(x)
        meta_logits = self.meta_mlp(meta)
        combined = image_logits + self.fusion_weight * meta_logits
        return combined


# --- 3. EVALUATION FUNCTIONS ---
def evaluate_single_model(model, dataloader, device):
    """Evaluate a single model without TTA."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, meta, labels in tqdm(dataloader, desc="Evaluating"):
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)
            outputs = model(imgs, meta)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def evaluate_ensemble(models, dataloader, device):
    """Evaluate ensemble of models by averaging probabilities."""
    for m in models:
        m.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, meta, labels in tqdm(dataloader, desc="Evaluating Ensemble"):
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)
            
            # Average probabilities from all models
            ensemble_probs = None
            for model in models:
                outputs = model(imgs, meta)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                if ensemble_probs is None:
                    ensemble_probs = probs
                else:
                    ensemble_probs += probs
            
            ensemble_probs /= len(models)
            _, predicted = torch.max(ensemble_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate Grandmaster Models')
    parser.add_argument('--model', type=str, help='Path to single .pth model file')
    parser.add_argument('--b3', type=str, help='Path to B3 Grandmaster model')
    parser.add_argument('--convnext', type=str, help='Path to ConvNeXt Grandmaster model')
    parser.add_argument('--ensemble', action='store_true', help='Evaluate ensemble of --b3 and --convnext')
    args = parser.parse_args()
    
    print(f"🔍 GRANDMASTER EVALUATOR | GPU: {torch.cuda.is_available()}")
    
    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize(int(300 * 1.1)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'val'), val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    num_classes = len(val_ds.classes)
    print(f"📂 Loaded {len(val_ds)} validation images across {num_classes} classes")
    
    # Single model evaluation
    if args.model:
        model_type = 'b3' if 'b3' in args.model.lower() else 'convnext' if 'convnext' in args.model.lower() else 'resnet'
        model = WildlifeLateFusion(num_classes, model_type=model_type).to(DEVICE)
        
        ckpt = torch.load(args.model, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        
        print(f"\n📊 Evaluating {args.model}...")
        acc = evaluate_single_model(model, val_loader, DEVICE)
        print(f"✅ Validation Accuracy: {acc*100:.2f}%")
        
    # Ensemble evaluation
    elif args.ensemble and args.b3 and args.convnext:
        models = []
        
        # Load B3
        model_b3 = WildlifeLateFusion(num_classes, model_type='b3').to(DEVICE)
        ckpt_b3 = torch.load(args.b3, map_location=DEVICE)
        model_b3.load_state_dict(ckpt_b3['model_state_dict'])
        models.append(model_b3)
        print(f"✅ Loaded B3: {args.b3}")
        
        # Load ConvNeXt  
        model_cnx = WildlifeLateFusion(num_classes, model_type='convnext').to(DEVICE)
        ckpt_cnx = torch.load(args.convnext, map_location=DEVICE)
        model_cnx.load_state_dict(ckpt_cnx['model_state_dict'])
        models.append(model_cnx)
        print(f"✅ Loaded ConvNeXt: {args.convnext}")
        
        print(f"\n📊 Evaluating Ensemble...")
        acc = evaluate_ensemble(models, val_loader, DEVICE)
        print(f"🏆 ENSEMBLE Accuracy: {acc*100:.2f}%")
        
    else:
        print("Usage:")
        print("  Single model:  python evaluate_grandmaster.py --model grandmaster_b3_final.pth")
        print("  Ensemble:      python evaluate_grandmaster.py --ensemble --b3 grandmaster_b3_final.pth --convnext grandmaster_convnext_final.pth")


if __name__ == "__main__":
    main()
