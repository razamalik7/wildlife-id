"""
Hierarchical Evaluation Script
==============================
Evaluates model accuracy at multiple taxonomic levels:
- Species (100 classes)
- Family (e.g., Canidae, Felidae)
- Class (e.g., Mammalia, Aves, Reptilia)

Also reports "taxonomic distance" of errors to understand severity.

Usage:
  python evaluate_hierarchical.py --model oogway_b3_final.pth --type b3
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os, json, math
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
METADATA_DROPOUT = 0.0  # Disabled for eval

# --- TAXONOMY HELPERS ---
try:
    TAXONOMY = json.load(open(os.path.join(BASE_DIR, 'taxonomy_hierarchy.json')))
    SPECIES_TO_FAMILY = {}
    SPECIES_TO_CLASS = {}
    for family, species_list in TAXONOMY['family_to_species'].items():
        for species in species_list:
            SPECIES_TO_FAMILY[species] = family if family != "null" else None
    for class_name, species_list in TAXONOMY['class_to_species'].items():
        for species in species_list:
            SPECIES_TO_CLASS[species] = class_name if class_name != "null" else None
except:
    print("⚠️ Taxonomy file missing!")
    SPECIES_TO_FAMILY = {}
    SPECIES_TO_CLASS = {}

def get_taxonomic_distance(species_a, species_b):
    """
    Returns: 0 (same species), 1 (same family), 2 (same class), 3 (different class)
    """
    if species_a == species_b: return 0
    fam_a = SPECIES_TO_FAMILY.get(species_a)
    fam_b = SPECIES_TO_FAMILY.get(species_b)
    cls_a = SPECIES_TO_CLASS.get(species_a)
    cls_b = SPECIES_TO_CLASS.get(species_b)
    if fam_a and fam_b and fam_a == fam_b: return 1
    if cls_a and cls_b and cls_a == cls_b: return 2
    return 3

# --- DATASET ---
class WildlifeGeoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build hierarchy maps
        self.families = sorted(list(set(SPECIES_TO_FAMILY.get(c, "unknown") for c in self.classes) - {None}))
        self.tax_classes = sorted(list(set(SPECIES_TO_CLASS.get(c, "unknown") for c in self.classes) - {None}))
        self.family_to_idx = {f: idx for idx, f in enumerate(self.families)}
        self.tax_class_to_idx = {c: idx for idx, c in enumerate(self.tax_classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, f)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, self.class_to_idx[class_name], class_name))
                        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path, label, class_name = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (300, 300))
            
        if self.transform:
            image = self.transform(image)
            
        lat, lng, month = 0.0, 0.0, 6
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                    lat = meta.get('lat', 0.0) or 0.0
                    lng = meta.get('lng', 0.0) or 0.0
                    if meta.get('date'):
                        try: month = datetime.strptime(meta['date'], '%Y-%m-%d').month
                        except: pass
            except: pass
            
        meta_vec = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32)
        
        fam = SPECIES_TO_FAMILY.get(class_name)
        cls = SPECIES_TO_CLASS.get(class_name)
        fam_idx = self.family_to_idx.get(fam, 0)
        cls_idx = self.tax_class_to_idx.get(cls, 0)
        
        return image, meta_vec, label, fam_idx, cls_idx, class_name

# --- MODEL (Exact copy from train_oogway.py) ---
class WildlifeLateFusion(nn.Module):
    def __init__(self, num_species, num_families, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            self.image_model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_features = self.image_model.classifier[1].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        elif model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_features = self.image_model.classifier[2].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        
        self.species_head = nn.Linear(feature_dim, num_species)
        self.family_head = nn.Linear(feature_dim, num_families)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, meta):
        image_features = self.image_model(x)
        meta_features = self.meta_mlp(meta)
        combined = image_features + self.fusion_weight * meta_features
        
        return {
            'species': self.species_head(combined),
            'family': self.family_head(combined),
            'class': self.class_head(combined)
        }

def evaluate(args):
    print(f"📊 HIERARCHICAL EVALUATION")
    print(f"   Model: {args.model}")
    print(f"   Type: {args.type}")
    
    # Transforms
    val_tf = transforms.Compose([
        transforms.Resize((340, 340)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'val'), val_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"✅ Loaded {len(val_ds)} validation samples")
    print(f"   {len(val_ds.classes)} species, {len(val_ds.families)} families, {len(val_ds.tax_classes)} classes")
    
    # Model
    model = WildlifeLateFusion(len(val_ds.classes), len(val_ds.families), len(val_ds.tax_classes), args.type).to(DEVICE)
    
    ckpt = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print("✅ Weights loaded")
    
    # Evaluation
    species_correct = 0
    family_correct = 0
    class_correct = 0
    total = 0
    
    # Track error severity
    error_distances = defaultdict(int)  # distance -> count
    
    print("\n🔍 Evaluating...")
    with torch.no_grad():
        for imgs, meta, sp_labels, fam_labels, cls_labels, class_names in tqdm(val_loader):
            imgs = imgs.to(DEVICE)
            meta = meta.to(DEVICE)
            sp_labels = sp_labels.to(DEVICE)
            fam_labels = fam_labels.to(DEVICE)
            cls_labels = cls_labels.to(DEVICE)
            
            out = model(imgs, meta)
            
            # Species predictions
            _, sp_pred = torch.max(out['species'], 1)
            species_correct += (sp_pred == sp_labels).sum().item()
            
            # Family predictions
            _, fam_pred = torch.max(out['family'], 1)
            family_correct += (fam_pred == fam_labels).sum().item()
            
            # Class predictions
            _, cls_pred = torch.max(out['class'], 1)
            class_correct += (cls_pred == cls_labels).sum().item()
            
            # Track error severity for species
            for i in range(len(sp_labels)):
                if sp_pred[i] != sp_labels[i]:
                    true_name = class_names[i]
                    pred_idx = sp_pred[i].item()
                    pred_name = val_ds.classes[pred_idx]
                    dist = get_taxonomic_distance(true_name, pred_name)
                    error_distances[dist] += 1
            
            total += sp_labels.size(0)
    
    # Results
    species_acc = species_correct / total * 100
    family_acc = family_correct / total * 100
    class_acc = class_correct / total * 100
    
    print(f"\n{'='*60}")
    print(f"📊 HIERARCHICAL ACCURACY REPORT")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Accuracy':>10}")
    print(f"{'-'*60}")
    print(f"{'Species-Level (100 classes)':<30} {species_acc:>9.2f}%")
    print(f"{'Family-Level (e.g., Canidae)':<30} {family_acc:>9.2f}%")
    print(f"{'Class-Level (e.g., Mammalia)':<30} {class_acc:>9.2f}%")
    print(f"{'='*60}")
    
    # Error Analysis
    total_errors = sum(error_distances.values())
    print(f"\n📈 ERROR SEVERITY ANALYSIS ({total_errors} total errors)")
    print(f"{'-'*60}")
    print(f"{'Distance 1 (Same Family)':<30} {error_distances.get(1, 0):>6} ({error_distances.get(1, 0)/max(1,total_errors)*100:.1f}%)")
    print(f"{'Distance 2 (Same Class, Diff Family)':<30} {error_distances.get(2, 0):>6} ({error_distances.get(2, 0)/max(1,total_errors)*100:.1f}%)")
    print(f"{'Distance 3 (Different Class)':<30} {error_distances.get(3, 0):>6} ({error_distances.get(3, 0)/max(1,total_errors)*100:.1f}%)")
    print(f"{'='*60}")
    
    # Summary
    print(f"\n💡 INTERPRETATION:")
    if error_distances.get(3, 0) > total_errors * 0.1:
        print("   ⚠️  >10% of errors are CROSS-CLASS (e.g., mammal->bird). Consider more data for affected species.")
    else:
        print("   ✅ Most errors are within-family or within-class (less severe).")
    
    # Save to file for reliable output
    with open('hierarchical_results.txt', 'w') as f:
        f.write(f"HIERARCHICAL ACCURACY REPORT\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Species-Level: {species_acc:.2f}%\n")
        f.write(f"Family-Level:  {family_acc:.2f}%\n")
        f.write(f"Class-Level:   {class_acc:.2f}%\n")
        f.write(f"{'='*60}\n")
        f.write(f"Error Distances ({total_errors} total):\n")
        f.write(f"  Same Family:     {error_distances.get(1, 0)} ({error_distances.get(1, 0)/max(1,total_errors)*100:.1f}%)\n")
        f.write(f"  Same Class:      {error_distances.get(2, 0)} ({error_distances.get(2, 0)/max(1,total_errors)*100:.1f}%)\n")
        f.write(f"  Different Class: {error_distances.get(3, 0)} ({error_distances.get(3, 0)/max(1,total_errors)*100:.1f}%)\n")
    print("\n📁 Results saved to hierarchical_results.txt")
    
    return species_acc, family_acc, class_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--type', default='b3')
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
