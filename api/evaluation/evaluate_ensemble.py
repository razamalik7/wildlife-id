
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, json, math, random
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Must match training!) ---
class WildlifeLateFusion(nn.Module):
    """Late Fusion: Image pathway + Metadata pathway with hierarchical outputs."""
    
    def __init__(self, num_species, num_families, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            self.image_model = models.efficientnet_b3(weights=None)
            in_features = self.image_model.classifier[1].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        elif model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights=None)
            in_features = self.image_model.classifier[2].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        
        # Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        
        # Hierarchical prediction heads
        self.species_head = nn.Linear(feature_dim, num_species)
        self.family_head = nn.Linear(feature_dim, num_families)
        self.class_head = nn.Linear(feature_dim, num_classes)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, meta):
        image_features = self.image_model(x)
        meta_features = self.meta_mlp(meta)
        combined_features = image_features + self.fusion_weight * meta_features
        
        return {
            'species': self.species_head(combined_features),
            'family': self.family_head(combined_features),
            'class': self.class_head(combined_features)
        }

# --- DATASET LOADER ---
class EvaluationDataset(Dataset):
    def __init__(self, root_dir, transform_b3, transform_convnext):
        self.root_dir = root_dir
        self.transform_b3 = transform_b3
        self.transform_convnext = transform_convnext
        self.samples = []
        
        if os.path.exists(root_dir):
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        else:
            self.classes = []
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (300, 300))
            
        # 1. Models need different transforms/sizes
        img_b3 = self.transform_b3(image)
        img_convnext = self.transform_convnext(image)
        
        # 2. Load Metadata
        lat, lng, month = 0.0, 0.0, 6
        if os.path.exists(json_path):
            try:
                meta = json.load(open(json_path))
                lat = meta.get('lat', 0.0) or 0.0
                lng = meta.get('lng', 0.0) or 0.0
                # Date parsing skipped for speed/simplicity here, assuming mid-year
            except: pass
            
        meta_vector = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32)
        
        return img_b3, img_convnext, meta_vector, label

def load_model(path, model_type, device):
    print(f"Loading {model_type} from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # Handle both old and new checkpoint formats
    class_names = checkpoint.get('class_names') or checkpoint.get('classes')
    families = checkpoint.get('families', [None] * 57)  # Default 57 families
    tax_classes = checkpoint.get('tax_classes', [None] * 4)  # Default 4 classes
    
    # Use fixed sizes if not in checkpoint (finetuned models)
    num_species = len(class_names) if class_names else 100
    num_families = len(families) if families and families[0] else 57
    num_classes = len(tax_classes) if tax_classes and tax_classes[0] else 4
    
    model = WildlifeLateFusion(num_species, num_families, num_classes, model_type=model_type)
    
    # Handle state dict (remove 'module.' prefix if from SWA/DataParallel)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    print(f"\n🦁 OOGWAY ENSEMBLE EVALUATION")
    print(f"Device: {DEVICE}")
    
    # 1. Define Transforms
    # B3: 300x300
    tf_b3 = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ConvNeXt: 224x224
    tf_cx = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Load Dataset
    val_dir = os.path.join(DATA_DIR, 'val')
    dataset = EvaluationDataset(val_dir, tf_b3, tf_cx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"📂 Evaluating on {len(dataset)} validation images")
    
    # 3. Load Models
    # Look for best/final models
    b3_path = 'oogway_b3_best.pth'
    cx_path = 'oogway_convnext_final.pth' # Or _best or _swa
    
    if not os.path.exists(b3_path) or not os.path.exists(cx_path):
        print("❌ Model files missing!")
        return

    model_b3 = load_model(b3_path, 'b3', DEVICE)
    model_cx = load_model(cx_path, 'convnext', DEVICE)
    
    correct_b3 = 0
    correct_cx = 0
    correct_ens = 0
    total = 0
    
    # 4. Evaluation Loop
    print("\n🚀 Starting Evaluation...")
    with torch.no_grad():
        for b3_imgs, cx_imgs, meta, labels in tqdm(loader):
            b3_imgs, cx_imgs = b3_imgs.to(DEVICE), cx_imgs.to(DEVICE)
            meta, labels = meta.to(DEVICE), labels.to(DEVICE)
            
            # --- MODEL 1: B3 ---
            out_b3 = model_b3(b3_imgs, meta)
            probs_b3 = torch.softmax(out_b3['species'], dim=1)
            
            # --- MODEL 2: CONVNEXT ---
            out_cx = model_cx(cx_imgs, meta)
            probs_cx = torch.softmax(out_cx['species'], dim=1)
            
            # --- EXPERT OVERRIDE LOGIC (From ai_engine.py) ---
            
            # 1. Get initial class predictions
            _, idx_b3 = torch.max(probs_b3, 1)
            _, idx_cx = torch.max(probs_cx, 1)
            
            # 2. DEFINITIVE EXPERT OVERRIDE SETS
            EXPERT_OVERRIDES = {
                ('striped_skunk', 'wild_boar'),        # Skunk/Boar
                ('sea_otter', 'harbor_seal'),          # Otter/Seal
                ('coyote', 'gray_wolf'),               # Canids
                ('american_alligator', 'american_crocodile'), # Crocs
                ('nile_monitor', 'argentine_black_and_white_tegu'), # Monitors
                ('white-tailed_deer', 'moose'),        # Deer/Moose
                ('california_sea_lion', 'northern_elephant_seal'), # Seals
                ('burmese_python', 'western_diamondback_rattlesnake') # Snake
            }
            
            B3_SUPERIOR_CLASSES = {
                'nile_monitor', 'yellow-bellied_marmot', 'harbor_seal', 
                'virginia_opossum', 'thinhorn_sheep', 'spotted_salamander',
                'western_diamondback_rattlesnake', 'mountain_lion', 
                'american_marten', 'common_box_turtle', 'great_horned_owl'
            }
            
            # Dynamic weighting loop
            probs_ens = torch.zeros_like(probs_cx)
            
            for i in range(len(labels)):
                idx_b3_i = idx_b3[i].item()
                idx_cx_i = idx_cx[i].item()
                
                pred_b3_name = dataset.classes[idx_b3_i]
                pred_cx_name = dataset.classes[idx_cx_i]
                
                # CASE 1: Specific Conflict Pair
                if (pred_cx_name, pred_b3_name) in EXPERT_OVERRIDES:
                    # Trust B3 heavily (70/30)
                    probs_ens[i] = (0.7 * probs_b3[i]) + (0.3 * probs_cx[i])
                    
                # CASE 2: B3 is a known specialist for this class
                elif pred_b3_name in B3_SUPERIOR_CLASSES:
                    # Trust B3 moderately (60/40)
                    probs_ens[i] = (0.6 * probs_b3[i]) + (0.4 * probs_cx[i])
                    
                # CASE 3: Standard (ConvNeXt is stronger generally)
                else:
                    probs_ens[i] = (0.4 * probs_b3[i]) + (0.6 * probs_cx[i])

            # --- PREDICTIONS ---
            _, pred_b3 = torch.max(probs_b3, 1)
            _, pred_cx = torch.max(probs_cx, 1)
            _, pred_ens = torch.max(probs_ens, 1)
            
            total += labels.size(0)
            correct_b3 += (pred_b3 == labels).sum().item()
            correct_cx += (pred_cx == labels).sum().item()
            correct_ens += (pred_ens == labels).sum().item()
            
    print("\n🏆 FINAL RESULTS")
    print(f"{'='*40}")
    print(f"EfficientNet-B3:   {100*correct_b3/total:.2f}%")
    print(f"ConvNeXt-Tiny:     {100*correct_cx/total:.2f}%")
    print(f"🌟 ENSEMBLE:       {100*correct_ens/total:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
