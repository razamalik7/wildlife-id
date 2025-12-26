"""
Targeted Fine-Tuning Script V2 (Robust / Gold Standard)
=======================================================
Combines the features of the main Oogway training (Hierarchical Loss, Hard Negative Mining, 
Mixup/CutMix, SWA) with the ability to load from MULTIPLE dataset sources (V2 + Refined).

Usage:
  python targeted_finetune_v2.py --model oogway_b3_final.pth --type b3 --roots training_data_v2 training_data_refined
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
from PIL import Image
import os, json, math, random
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse

# --- CONFIGURATION V3 (Aggressive) ---
BATCH_SIZE = 16  # Reverted - 32 caused memory issues
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Feature Flags
METADATA_DROPOUT = 0.2
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
FOCAL_GAMMA = 2.0
HARD_PAIRS_PER_BATCH = 8  # Doubled from 4

# Top Confused Pairs - REBUILT from actual baseline confusion analysis
# Priority scores now reflect ACTUAL confusion counts from ensemble analysis
CONFUSED_PAIRS = [
    # High Priority (8+ confusions in baseline)
    (('wild_boar', 'striped_skunk'), 12),           # TOP confusion!
    (('brown_anole', 'green_anole'), 12),           # Anole confusion
    (('gray_wolf', 'coyote'), 11),                  # Canid confusion
    (('domestic_dog', 'gray_wolf'), 8),             # Dog/wolf
    
    # Medium Priority (5-7 confusions)
    (('american_alligator', 'american_crocodile'), 7),
    (('harbor_seal', 'sea_otter'), 6),
    (('argentine_black_and_white_tegu', 'nile_monitor'), 6),
    (('elk', 'moose'), 6),
    (('gray_wolf', 'mountain_lion'), 6),
    (('moose', 'white-tailed_deer'), 6),
    (('american_crocodile', 'nile_monitor'), 5),
    (('eastern_fox_squirrel', 'eastern_gray_squirrel'), 5),
    (('california_sea_lion', 'harbor_seal'), 5),
    (('jaguar', 'mountain_lion'), 5),
    (('raccoon', 'virginia_opossum'), 5),
    (('harbor_seal', 'northern_elephant_seal'), 5),
    
    # Lower Priority (3-4 confusions but still important)
    (('american_bullfrog', 'cane_toad'), 4),
    (('american_mink', 'north_american_river_otter'), 4),
    (('bighorn_sheep', 'thinhorn_sheep'), 4),
    (('bobcat', 'mountain_lion'), 4),
    (('american_black_bear', 'grizzly_bear'), 4),
    (('gray_fox', 'red_fox'), 4),
    (('groundhog', 'yellow-bellied_marmot'), 4),
    (('jaguar', 'ocelot'), 4),
    (('north_american_beaver', 'nutria'), 4),
    (('north_american_river_otter', 'sea_otter'), 4),
    (('caribou', 'moose'), 3),
    (('coyote', 'gray_fox'), 3),
    (('california_sea_lion', 'northern_elephant_seal'), 3),
    (('american_red_squirrel', 'eastern_gray_squirrel'), 3),
    (('arctic_fox', 'red_fox'), 3),
    (('eastern_newt', 'spotted_salamander'), 3),
    (('mountain_lion', 'ocelot'), 3),
    (('american_crocodile', 'common_snapping_turtle'), 3),
]

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
    print("⚠️ Taxonomy file missing or invalid. Hierarchical loss will be flat.")
    SPECIES_TO_FAMILY = {}
    SPECIES_TO_CLASS = {}

def get_taxonomic_distance(species_a, species_b):
    if species_a == species_b: return 0
    fam_a = SPECIES_TO_FAMILY.get(species_a)
    fam_b = SPECIES_TO_FAMILY.get(species_b)
    cls_a = SPECIES_TO_CLASS.get(species_a)
    cls_b = SPECIES_TO_CLASS.get(species_b)
    if fam_a and fam_b and fam_a == fam_b: return 1
    if cls_a and cls_b and cls_a == cls_b: return 2
    return 3

# --- MULTI-SOURCE DATASET ---
class WildlifeGeoDataset(Dataset):
    """Loads images from MULTIPLE root directories seamlessly."""
    def __init__(self, root_dirs, transform=None, is_train=False):
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        # 1. Discover all classes across all roots
        self.classes = set()
        valid_roots = []
        for root in root_dirs:
            split_path = os.path.join(root, 'train' if is_train else 'val')
            if os.path.exists(split_path):
                valid_roots.append(split_path)
                self.classes.update([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build hierarchy map for this dataset
        self.families = sorted(list(set(SPECIES_TO_FAMILY.get(c, "unknown") for c in self.classes) - {None}))
        self.tax_classes = sorted(list(set(SPECIES_TO_CLASS.get(c, "unknown") for c in self.classes) - {None}))
        self.family_to_idx = {f: idx for idx, f in enumerate(self.families)}
        self.tax_class_to_idx = {c: idx for idx, c in enumerate(self.tax_classes)}
        
        # 2. Collect samples
        for split_root in valid_roots:
            for class_name in self.classes:
                class_dir = os.path.join(split_root, class_name)
                if not os.path.isdir(class_dir): continue
                
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
            image = Image.new('RGB', (300, 300)) # Fallback
            
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
            
        # Augment location
        if self.is_train and (lat != 0 or lng != 0):
            lat = np.clip(lat + np.random.uniform(-1, 1), -90, 90)
            lng = np.clip(lng + np.random.uniform(-1, 1), -180, 180)
            
        meta_vec = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32)
        
        # Hierarchical Labels
        fam = SPECIES_TO_FAMILY.get(class_name)
        cls = SPECIES_TO_CLASS.get(class_name)
        fam_idx = self.family_to_idx.get(fam, 0)
        cls_idx = self.tax_class_to_idx.get(cls, 0)
        
        return image, meta_vec, label, fam_idx, cls_idx

# --- AUGMENTATION & SAMPLER ---
def mixup_data(x, meta, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    mixed_meta = lam * meta + (1 - lam) * meta[idx, :]
    return mixed_x, mixed_meta, y, y[idx], lam, idx

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam, idx

class HardNegativeSampler(Sampler):
    """
    Force-feeds the model with hard negative pairs to fix confusions.
    Does NOT decay during finetuning (we want constant pressure).
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_indices = defaultdict(list)
        for idx, (_, _, _, name) in enumerate(dataset.samples):
            self.class_indices[name].append(idx)
            
        self.valid_pairs = [p[0] for p in CONFUSED_PAIRS if p[0][0] in self.class_indices and p[0][1] in self.class_indices]
        self.num_batches = len(dataset) // batch_size
        
        print(f"  🎯 Hard Negative Sampler active for {len(self.valid_pairs)} pairs")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            # 1. Grab confusion pairs (DOUBLED to 8)
            if self.valid_pairs:
                for _ in range(min(HARD_PAIRS_PER_BATCH, len(self.valid_pairs))):
                    p = random.choice(self.valid_pairs)
                    if self.class_indices[p[0]] and self.class_indices[p[1]]:
                        batch.append(random.choice(self.class_indices[p[0]]))
                        batch.append(random.choice(self.class_indices[p[1]]))
            
            # 2. Fill rest with random data
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                batch.extend(random.sample(range(len(self.dataset)), remaining))
                
            random.shuffle(batch)
            yield batch
            
    def __len__(self): return self.num_batches

# --- MODEL & LOSS ---
class HierarchicalMultiTaskLoss(nn.Module):
    def __init__(self, class_names, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.class_names = class_names
        # Build weight matrix
        n = len(class_names)
        self.register_buffer('weights', torch.ones(n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = get_taxonomic_distance(class_names[i], class_names[j])
                    self.weights[i, j] = float(dist) # 1, 2, or 3
                    
    def forward(self, preds, target_s, target_f, target_c):
        # Focal Loss on Species
        ce = F.cross_entropy(preds['species'], target_s, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()
        
        # Hierarchical Penalty (Taxon Aware)
        # If model predicts wrong species, check distance
        # (Simplified for finetuning speed - focus mostly on Focal)
        
        # Simple Family/Class support
        fam_loss = F.cross_entropy(preds['family'], target_f)
        cls_loss = F.cross_entropy(preds['class'], target_c)
        
        return focal_loss + 0.3 * fam_loss + 0.1 * cls_loss

class WildlifeLateFusion(nn.Module):
    """Late Fusion: Image pathway + Metadata pathway with hierarchical outputs."""
    
    def __init__(self, num_species, num_families, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            self.image_model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_features = self.image_model.classifier[1].in_features
            # Remove final layer - we'll add our own heads
            self.image_model.classifier = self.image_model.classifier[:-1]  # Keep dropout
            feature_dim = in_features
        elif model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_features = self.image_model.classifier[2].in_features
            # Remove final layer
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        
        # Metadata MLP (outputs into feature space, not logits)
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, feature_dim)  # Match image feature dim
        )
        
        # Hierarchical prediction heads
        self.species_head = nn.Linear(feature_dim, num_species)
        self.family_head = nn.Linear(feature_dim, num_families)
        self.class_head = nn.Linear(feature_dim, num_classes)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, meta):
        # Image pathway
        image_features = self.image_model(x)
        
        # Metadata dropout: force model to work without location sometimes
        if self.training and torch.rand(1).item() < METADATA_DROPOUT:
            meta = torch.zeros_like(meta)
        
        # Metadata pathway
        meta_features = self.meta_mlp(meta)
        
        # Fuse features
        combined_features = image_features + self.fusion_weight * meta_features
        
        # Multi-task predictions
        return {
            'species': self.species_head(combined_features),
            'family': self.family_head(combined_features),
            'class': self.class_head(combined_features)
        }
    
    @property
    def features(self):
        if self.model_type == 'b3':
            return self.image_model.features
        else:
            return self.image_model.features
    
    @property
    def classifier(self):
        # For FixRes compatibility - return species head
        return self.species_head

# --- MAIN FINETUNING ---
def fine_tune(args):
    print(f"🔧 TARGETED FINE-TUNING V2 (ROBUST MODE)")
    print(f"   Model: {args.model}")
    print(f"   Roots: {args.roots}")
    
    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((340, 340)),
        transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((340, 340)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("📂 Loading Multi-Source Datasets...")
    train_ds = WildlifeGeoDataset(args.roots, train_tf, is_train=True)
    val_ds = WildlifeGeoDataset(args.roots, val_tf, is_train=False)
    
    sampler = HardNegativeSampler(train_ds, BATCH_SIZE)
    # Note: On Windows, num_workers > 0 can be problematic. Keep it simple.
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"✅ Loaded {len(train_ds)} train samples ({len(train_ds.classes)} classes)")
    
    # Model Init
    model = WildlifeLateFusion(len(train_ds.classes), len(train_ds.families), len(train_ds.tax_classes), args.type).to(DEVICE)
    
    # Load Weights (Carefully)
    try:
        if os.path.exists(args.model):
            ckpt = torch.load(args.model, map_location=DEVICE)
            # Need strict=False because V2 model might have different head sizes if classes changed
            # But here we assume classes are same + refined data is valid species
            print(model.load_state_dict(ckpt['model_state_dict'], strict=False))
            print("INFO: Weights loaded (strict=False)")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # Optimizer (Low LR)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = HierarchicalMultiTaskLoss(train_ds.classes, gamma=FOCAL_GAMMA).to(DEVICE)
    
    # SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    
    best_acc = 0.0
    
    print(f"\n🚀 Starting Fine-Tuning ({args.epochs} Epochs)...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, meta, lbls, fams, clss in pbar:
            imgs, meta, lbls, fams, clss = imgs.to(DEVICE), meta.to(DEVICE), lbls.to(DEVICE), fams.to(DEVICE), clss.to(DEVICE)
            
            # Curriculum: No augmentation for first 5 epochs (clean learning)
            use_augmentation = epoch >= 5
            
            if use_augmentation and random.random() < 0.5:
                imgs, meta, l_a, l_b, lam, idx = mixup_data(imgs, meta, lbls, MIXUP_ALPHA)
                f_a, f_b = fams, fams[idx]
                c_a, c_b = clss, clss[idx]
                
                optimizer.zero_grad()
                out = model(imgs, meta)
                loss = lam * criterion(out, l_a, f_a, c_a) + (1-lam) * criterion(out, l_b, f_b, c_b)
            elif use_augmentation:
                imgs, l_a, l_b, lam, idx = cutmix_data(imgs, lbls, CUTMIX_ALPHA)
                f_a, f_b = fams, fams[idx]
                c_a, c_b = clss, clss[idx]
                
                optimizer.zero_grad()
                out = model(imgs, meta)
                loss = lam * criterion(out, l_a, f_a, c_a) + (1-lam) * criterion(out, l_b, f_b, c_b)
            else:
                # Clean training (no augmentation)
                optimizer.zero_grad()
                out = model(imgs, meta)
                loss = criterion(out, lbls, fams, clss)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        # SWA Update (Start at epoch 10 for 15-epoch training)
        swa_start = max(args.epochs * 2 // 3, 10)  # 2/3 through or epoch 10
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, meta, lbls, _, _ in val_loader:
                imgs, meta, lbls = imgs.to(DEVICE), meta.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs, meta)
                _, pred = torch.max(out['species'], 1)
                total += lbls.size(0)
                correct += (pred == lbls).sum().item()
                
        acc = correct / total
        print(f"   Val Acc: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            save_path = args.model.replace('.pth', '_finetuned.pth')
            # Save SWA model if available, else standard
            state_dict = swa_model.module.state_dict() if epoch >= args.epochs//2 else model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'accuracy': acc,
                'epoch': epoch,
                'classes': train_ds.classes
            }, save_path)
            print(f"   ⭐ Model Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--type', default='b3')
    parser.add_argument('--roots', nargs='+', default=['training_data_v2', 'training_data_refined'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-5)  # REDUCED - 1e-4 caused catastrophic forgetting
    args = parser.parse_args()
    fine_tune(args)

if __name__ == "__main__":
    main()
