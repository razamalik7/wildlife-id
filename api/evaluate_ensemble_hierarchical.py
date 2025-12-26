"""
HIERARCHICAL ENSEMBLE EVALUATION
================================
Tests the ADVANCED ensemble (B3 + ConvNeXt + Expert Overrides)
at all taxonomic levels: Species, Family, Class

Usage:
  python evaluate_ensemble_hierarchical.py
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, json, math
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 16

# --- TAXONOMY ---
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
    SPECIES_TO_FAMILY = {}
    SPECIES_TO_CLASS = {}

def get_taxonomic_distance(a, b):
    if a == b: return 0
    if SPECIES_TO_FAMILY.get(a) == SPECIES_TO_FAMILY.get(b) and SPECIES_TO_FAMILY.get(a): return 1
    if SPECIES_TO_CLASS.get(a) == SPECIES_TO_CLASS.get(b) and SPECIES_TO_CLASS.get(a): return 2
    return 3

# --- MODEL ---
class WildlifeLateFusion(nn.Module):
    def __init__(self, num_species, num_families, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            self.image_model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_features = self.image_model.classifier[1].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
        elif model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_features = self.image_model.classifier[2].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, in_features)
        )
        self.species_head = nn.Linear(in_features, num_species)
        self.family_head = nn.Linear(in_features, num_families)
        self.class_head = nn.Linear(in_features, num_classes)
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, meta):
        img_feat = self.image_model(x)
        meta_feat = self.meta_mlp(meta)
        combined = img_feat + self.fusion_weight * meta_feat
        return {
            'species': self.species_head(combined),
            'family': self.family_head(combined),
            'class': self.class_head(combined)
        }

# --- DATASET ---
class EnsembleDataset(Dataset):
    def __init__(self, root_dir, tf_b3, tf_cx):
        self.tf_b3, self.tf_cx = tf_b3, tf_cx
        self.samples = []
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.families = sorted(list(set(SPECIES_TO_FAMILY.get(c, "unk") for c in self.classes) - {None}))
        self.tax_classes = sorted(list(set(SPECIES_TO_CLASS.get(c, "unk") for c in self.classes) - {None}))
        self.family_to_idx = {f: i for i, f in enumerate(self.families)}
        self.tax_class_to_idx = {c: i for i, c in enumerate(self.tax_classes)}
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, f), os.path.splitext(os.path.join(cls_dir, f))[0]+'.json', self.class_to_idx[cls], cls))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, json_path, label, name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        lat, lng, month = 0.0, 0.0, 6
        if os.path.exists(json_path):
            try:
                m = json.load(open(json_path))
                lat, lng = m.get('lat',0) or 0, m.get('lng',0) or 0
                if m.get('date'):
                    try: month = datetime.strptime(m['date'], '%Y-%m-%d').month
                    except: pass
            except: pass
        
        meta = torch.tensor([math.sin(2*math.pi*month/12), math.cos(2*math.pi*month/12), lat/90, lng/180], dtype=torch.float32)
        
        fam = SPECIES_TO_FAMILY.get(name)
        cls = SPECIES_TO_CLASS.get(name)
        fam_idx = self.family_to_idx.get(fam, 0)
        cls_idx = self.tax_class_to_idx.get(cls, 0)
        
        return self.tf_b3(img), self.tf_cx(img), meta, label, fam_idx, cls_idx, name

def load_model(path, mtype, n_sp, n_fam, n_cls):
    model = WildlifeLateFusion(n_sp, n_fam, n_cls, mtype).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model

def main():
    print("📊 HIERARCHICAL ENSEMBLE EVALUATION (Expert Overrides)")
    
    tf_b3 = transforms.Compose([transforms.Resize((320,320)), transforms.CenterCrop(300), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_cx = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    ds = EnsembleDataset(os.path.join(DATA_DIR, 'val'), tf_b3, tf_cx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"✅ {len(ds)} samples | {len(ds.classes)} species | {len(ds.families)} families | {len(ds.tax_classes)} classes")
    
    model_b3 = load_model('oogway_b3_final_finetuned.pth', 'b3', len(ds.classes), len(ds.families), len(ds.tax_classes))
    model_cx = load_model('oogway_convnext_final_finetuned.pth', 'convnext', len(ds.classes), len(ds.families), len(ds.tax_classes))
    
    # Expert Overrides
    EXPERT_OVERRIDES = {
        ('striped_skunk', 'wild_boar'), ('sea_otter', 'harbor_seal'), ('coyote', 'gray_wolf'),
        ('american_alligator', 'american_crocodile'), ('nile_monitor', 'argentine_black_and_white_tegu'),
        ('white-tailed_deer', 'moose'), ('california_sea_lion', 'northern_elephant_seal'),
        ('burmese_python', 'western_diamondback_rattlesnake')
    }
    B3_SUPERIOR = {'nile_monitor', 'yellow-bellied_marmot', 'harbor_seal', 'virginia_opossum', 'thinhorn_sheep', 
                   'spotted_salamander', 'western_diamondback_rattlesnake', 'mountain_lion', 'american_marten', 'common_box_turtle', 'great_horned_owl'}
    
    sp_corr, fam_corr, cls_corr, total = 0, 0, 0, 0
    error_dist = defaultdict(int)
    
    print("🔍 Evaluating...")
    with torch.no_grad():
        for b3_img, cx_img, meta, sp_lbl, fam_lbl, cls_lbl, names in tqdm(loader):
            b3_img, cx_img, meta = b3_img.to(DEVICE), cx_img.to(DEVICE), meta.to(DEVICE)
            sp_lbl, fam_lbl, cls_lbl = sp_lbl.to(DEVICE), fam_lbl.to(DEVICE), cls_lbl.to(DEVICE)
            
            out_b3 = model_b3(b3_img, meta)
            out_cx = model_cx(cx_img, meta)
            
            probs_b3 = torch.softmax(out_b3['species'], 1)
            probs_cx = torch.softmax(out_cx['species'], 1)
            
            _, idx_b3 = torch.max(probs_b3, 1)
            _, idx_cx = torch.max(probs_cx, 1)
            
            # Ensemble with expert logic
            probs_ens = torch.zeros_like(probs_cx)
            for i in range(len(sp_lbl)):
                pred_b3 = ds.classes[idx_b3[i].item()]
                pred_cx = ds.classes[idx_cx[i].item()]
                
                if (pred_cx, pred_b3) in EXPERT_OVERRIDES:
                    probs_ens[i] = 0.7 * probs_b3[i] + 0.3 * probs_cx[i]
                elif pred_b3 in B3_SUPERIOR:
                    probs_ens[i] = 0.6 * probs_b3[i] + 0.4 * probs_cx[i]
                else:
                    probs_ens[i] = 0.4 * probs_b3[i] + 0.6 * probs_cx[i]
            
            _, sp_pred = torch.max(probs_ens, 1)
            
            # Family + Class from weighted ensemble (use species prediction to derive)
            for i in range(len(sp_lbl)):
                pred_name = ds.classes[sp_pred[i].item()]
                true_name = names[i]
                
                pred_fam = SPECIES_TO_FAMILY.get(pred_name)
                true_fam = SPECIES_TO_FAMILY.get(true_name)
                pred_cls = SPECIES_TO_CLASS.get(pred_name)
                true_cls = SPECIES_TO_CLASS.get(true_name)
                
                if sp_pred[i] == sp_lbl[i]:
                    sp_corr += 1
                else:
                    error_dist[get_taxonomic_distance(true_name, pred_name)] += 1
                    
                if pred_fam == true_fam and pred_fam:
                    fam_corr += 1
                if pred_cls == true_cls and pred_cls:
                    cls_corr += 1
                    
            total += sp_lbl.size(0)
    
    sp_acc = sp_corr / total * 100
    fam_acc = fam_corr / total * 100
    cls_acc = cls_corr / total * 100
    tot_err = sum(error_dist.values())
    
    print(f"\n{'='*60}")
    print(f"📊 ENSEMBLE HIERARCHICAL ACCURACY")
    print(f"{'='*60}")
    print(f"Species-Level:  {sp_acc:.2f}%")
    print(f"Family-Level:   {fam_acc:.2f}%")
    print(f"Class-Level:    {cls_acc:.2f}%")
    print(f"{'='*60}")
    print(f"Error Distances ({tot_err} total):")
    print(f"  Same Family:     {error_dist.get(1,0)} ({error_dist.get(1,0)/max(1,tot_err)*100:.1f}%)")
    print(f"  Same Class:      {error_dist.get(2,0)} ({error_dist.get(2,0)/max(1,tot_err)*100:.1f}%)")
    print(f"  Different Class: {error_dist.get(3,0)} ({error_dist.get(3,0)/max(1,tot_err)*100:.1f}%)")
    print(f"{'='*60}")
    
    with open('hierarchical_ensemble_results.txt', 'w') as f:
        f.write(f"ENSEMBLE HIERARCHICAL ACCURACY\n{'='*60}\n")
        f.write(f"Species: {sp_acc:.2f}%\nFamily: {fam_acc:.2f}%\nClass: {cls_acc:.2f}%\n")
        f.write(f"Errors: SameFam={error_dist.get(1,0)}, SameCls={error_dist.get(2,0)}, DiffCls={error_dist.get(3,0)}\n")
    print("📁 Results saved to hierarchical_ensemble_results.txt")

if __name__ == "__main__":
    main()
