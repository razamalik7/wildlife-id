"""
OOGWAY TRAINING SCRIPT
Next-generation wildlife classification with advanced techniques:
- Fixed SWA (proper LR, BatchNorm update, saves averaged weights)
- Focal Loss (γ=2.0) for hard example focus
- Live Confusion-Weighted Loss (updates every 3 epochs)
- Hard Negative Mining with sqrt-smoothed weights (25 confused pairs)
- TTA Validation for better model selection
- Metadata Dropout (20%) to prevent habitat shortcuts
- CutMix augmentation
- Sequential B3 + ConvNeXt training in one script
- Full taxon support (TODO: integrate with taxonomy_hierarchy.json)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
from PIL import Image
import os, json, math, random
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
DATA_DIR = './training_data_v2'
BATCH_SIZE = 16
NUM_EPOCHS = 20
SWA_START_EPOCH = 15
FIXRES_EPOCHS = 3  # Extended from 1
FIXRES_SIZE = 384
BASE_SIZE_B3 = 300
BASE_SIZE_CONVNEXT = 224
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
METADATA_DROPOUT = 0.2
FOCAL_GAMMA = 2.0
CONFUSION_UPDATE_INTERVAL = 3  # Update weights every 3 epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Top 25 Confused Pairs from Grandmaster Analysis (with error counts)
CONFUSED_PAIRS = [
    (('elk', 'moose'), 97),
    (('eastern_newt', 'spotted_salamander'), 84),
    (('arctic_fox', 'red_fox'), 78),
    (('california_sea_lion', 'northern_elephant_seal'), 22),
    (('american_marten', 'wolverine'), 19),
    (('brown_anole', 'green_anole'), 19),
    (('california_sea_lion', 'sea_otter'), 18),
    (('american_black_bear', 'grizzly_bear'), 17),
    (('bald_eagle', 'red-tailed_hawk'), 17),
    (('thinhorn_sheep', 'mountain_goat'), 17),
    (('mountain_lion', 'ocelot'), 16),
    (('polar_bear', 'walrus'), 16),
    (('raccoon', 'striped_skunk'), 16),
    (('american_alligator', 'american_crocodile'), 15),
    (('american_black_bear', 'mountain_lion'), 15),
    (('argentine_black_and_white_tegu', 'nile_monitor'), 15),
    (('harbor_seal', 'northern_elephant_seal'), 14),
    (('coyote', 'gray_fox'), 13),
    (('coyote', 'gray_wolf'), 13),
    (('mallard', 'mute_swan'), 13),
    (('bighorn_sheep', 'thinhorn_sheep'), 12),
    (('eastern_cottontail', 'snowshoe_hare'), 12),
    (('mute_swan', 'whooping_crane'), 12),
    (('american_mink', 'north_american_river_otter'), 11),
    (('american_red_squirrel', 'eastern_gray_squirrel'), 11),
]

# Sqrt-smoothed sampling probabilities
pair_weights = np.array([count for _, count in CONFUSED_PAIRS])
smoothed_weights = np.sqrt(pair_weights)
PAIR_PROBS = smoothed_weights / smoothed_weights.sum()

# --- TAXONOMY HIERARCHY ---
# Load taxonomic hierarchy for hierarchical loss weighting
TAXONOMY = json.load(open('taxonomy_hierarchy.json'))
SPECIES_TO_FAMILY = {}
SPECIES_TO_CLASS = {}

for family, species_list in TAXONOMY['family_to_species'].items():
    for species in species_list:
        SPECIES_TO_FAMILY[species] = family if family != "null" else None

for class_name, species_list in TAXONOMY['class_to_species'].items():
    for species in species_list:
        SPECIES_TO_CLASS[species] = class_name if class_name != "null" else None


def get_taxonomic_distance(species_a, species_b):
    """
    Compute taxonomic distance between two species.
    Returns: 0 (same species), 1 (same family), 2 (same class), 3 (different class)
    """
    if species_a == species_b:
        return 0
    
    family_a = SPECIES_TO_FAMILY.get(species_a)
    family_b = SPECIES_TO_FAMILY.get(species_b)
    class_a = SPECIES_TO_CLASS.get(species_a)
    class_b = SPECIES_TO_CLASS.get(species_b)
    
    # Same family (e.g., two canids)
    if family_a and family_b and family_a == family_b:
        return 1
    
    # Same class but different family (e.g., canid vs felid)
    if class_a and class_b and class_a == class_b:
        return 2
    
    # Different class (e.g., mammal vs bird)
    return 3


# --- 1. GEO-AWARE DATASET ---
class WildlifeGeoDataset(Dataset):
    """Loads images + metadata (lat/lng/date) from JSON sidecars."""
    
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        # Robust listing
        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: Directory not found: {root_dir}")
            self.classes = []
        else:
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build hierarchical mappings
        all_families = set()
        all_tax_classes = set()
        for species in self.classes:
            family = SPECIES_TO_FAMILY.get(species, "unknown")
            tax_class = SPECIES_TO_CLASS.get(species, "unknown")
            if family and family != "unknown":
                all_families.add(family)
            if tax_class and tax_class != "unknown":
                all_tax_classes.add(tax_class)
        
        self.families = sorted(list(all_families))
        self.tax_classes = sorted(list(all_tax_classes))
        self.family_to_idx = {f: idx for idx, f in enumerate(self.families)}
        self.class_to_idx_tax = {c: idx for idx, c in enumerate(self.tax_classes)}
        
        # Build sample list
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir): continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, self.class_to_idx[class_name], class_name))
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path, label, class_name = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback for corrupt images
            image = Image.new('RGB', (300, 300))

        if self.transform: 
            image = self.transform(image)
        
        # Load metadata with fallbacks
        lat, lng, month = 0.0, 0.0, 6
        if os.path.exists(json_path):
            try:
                meta = json.load(open(json_path))
                lat = meta.get('lat', 0.0) or 0.0
                lng = meta.get('lng', 0.0) or 0.0
                date_str = meta.get('date', '')
                if date_str:
                    try: 
                        month = datetime.strptime(date_str, '%Y-%m-%d').month
                    except: 
                        pass
            except: 
                pass
        
        # Location noise augmentation (±1° ~70 miles)
        if self.is_train and (lat != 0.0 or lng != 0.0):
            lat += np.random.uniform(-1.0, 1.0)
            lng += np.random.uniform(-1.0, 1.0)
            lat = np.clip(lat, -90.0, 90.0)
            lng = np.clip(lng, -180.0, 180.0)
        
        # Cyclic month encoding
        meta_vector = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32)
        
        # Get hierarchical labels
        family = SPECIES_TO_FAMILY.get(class_name)
        tax_class = SPECIES_TO_CLASS.get(class_name)
        
        # Convert to indices (default to 0 if unknown)
        family_idx = self.family_to_idx.get(family, 0) if family else 0
        class_idx = self.class_to_idx_tax.get(tax_class, 0) if tax_class else 0
        
        return image, meta_vector, label, family_idx, class_idx


# --- 2. HARD NEGATIVE MINING SAMPLER ---
class HardNegativeSampler(Sampler):
    """Ensures each batch contains samples from confused pairs."""
    
    def __init__(self, dataset, batch_size, num_pairs_per_batch=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_pairs = num_pairs_per_batch
        
        # Build class-to-indices mapping
        self.class_indices = defaultdict(list)
        for idx, (_, _, _, class_name) in enumerate(dataset.samples):
            self.class_indices[class_name].append(idx)
        
        # Remove pairs where classes don't exist in dataset
        self.valid_pairs = []
        for (class_a, class_b), count in CONFUSED_PAIRS:
            if class_a in self.class_indices and class_b in self.class_indices:
                self.valid_pairs.append((class_a, class_b))
        
        self.num_batches = max(1, len(dataset) // batch_size)
    
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            
            # Sample confused pairs with sqrt-smoothed probabilities
            if len(self.valid_pairs) > 0:
                n_pairs = min(self.num_pairs, len(self.valid_pairs))
                # Adjust probability vector length if pairs were filtered
                probs = PAIR_PROBS[:len(self.valid_pairs)]
                probs = probs / probs.sum() # Normalize
                
                selected_idx = np.random.choice(len(self.valid_pairs), size=n_pairs, replace=False, p=probs)
                selected_pairs = [self.valid_pairs[i] for i in selected_idx]
                
                # Add 1-2 samples from each confused pair
                for class_a, class_b in selected_pairs:
                    if len(batch) < self.batch_size - 1:
                        if self.class_indices[class_a]:
                            batch.append(random.choice(self.class_indices[class_a]))
                        if self.class_indices[class_b]:
                            batch.append(random.choice(self.class_indices[class_b]))
            
            # Fill rest randomly
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                random_indices = random.sample(range(len(self.dataset)), remaining)
                batch.extend(random_indices)
            
            random.shuffle(batch)
            yield batch  # Yield entire batch for batch_sampler
    
    def __len__(self):
        return self.num_batches  # Number of batches, not total samples


# --- 3. AUGMENTATION HELPERS ---
def mixup_data(x, meta, y, alpha=0.2):
    """MixUp: blend images and metadata."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_meta = lam * meta + (1 - lam) * meta[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_meta, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    """Generate random bbox for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste random patches."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual bbox area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# --- 4. HIERARCHICAL MULTI-TASK LOSS ---
class HierarchicalMultiTaskLoss(nn.Module):
    """
    Multi-task loss combining:
    - Species-level focal loss with hierarchical weighting
    - Family-level cross-entropy
    - Class-level cross-entropy
    """
    
    def __init__(self, class_names, gamma=2.0, label_smoothing=0.1, alpha_family=0.3, alpha_class=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha_family = alpha_family  # Weight for family loss
        self.alpha_class = alpha_class    # Weight for class loss
        self.class_names = class_names
        self.class_weights = None
        
        # Build taxonomic distance matrix for species loss
        self.register_buffer('taxon_weights', self._build_taxon_weights())
    
    def _build_taxon_weights(self):
        """Build NxN matrix of taxonomic distance weights."""
        n = len(self.class_names)
        weights = torch.ones(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    weights[i, j] = 1.0
                else:
                    dist = get_taxonomic_distance(self.class_names[i], self.class_names[j])
                    if dist == 1:  # Same family
                        weights[i, j] = 1.0
                    elif dist == 2:  # Same class, different family
                        weights[i, j] = 2.0
                    elif dist == 3:  # Different class
                        weights[i, j] = 3.0
        
        return weights
    
    def forward(self, predictions, species_targets, family_targets, class_targets):
        """
        predictions: dict with 'species', 'family', 'class' logits
        targets: species_targets, family_targets, class_targets
        """
        species_logits = predictions['species']
        family_logits = predictions['family']
        class_logits = predictions['class']
        
        # Species loss with hierarchical focal weighting
        ce_loss = F.cross_entropy(species_logits, species_targets, weight=self.class_weights, 
                                   label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply hierarchical weighting
        _, preds = torch.max(species_logits, 1)
        # Move taxon_weights to same device as targets
        taxon_weight = self.taxon_weights.to(species_targets.device)[species_targets, preds]
        species_loss = (focal_loss * taxon_weight).mean()
        
        # Family loss (simple cross-entropy)
        family_loss = F.cross_entropy(family_logits, family_targets, label_smoothing=self.label_smoothing)
        
        # Class loss (simple cross-entropy)
        class_loss = F.cross_entropy(class_logits, class_targets, label_smoothing=self.label_smoothing)
        
        # Combined loss
        total_loss = species_loss + self.alpha_family * family_loss + self.alpha_class * class_loss
        
        return total_loss, {
            'species': species_loss.item(),
            'family': family_loss.item(),
            'class': class_loss.item()
        }
    
    def update_weights(self, confusion_matrix, device):
        """Update class weights based on confusion matrix."""
        num_classes = len(confusion_matrix)
        weights = torch.ones(num_classes)
        
        for i in range(num_classes):
            total = confusion_matrix[i].sum()
            if total > 0:
                errors = total - confusion_matrix[i, i]
                weights[i] = min(1.0 + (errors / total) * 0.5, 2.0)
        
        self.class_weights = weights.to(device)
        print(f"  Updated loss weights: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")


# --- 5. LATE FUSION MODEL WITH HIERARCHICAL OUTPUTS ---
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


# --- 6. CONFUSION MATRIX COMPUTATION ---
def compute_confusion_matrix(model, dataloader, num_classes, device):
    """Compute confusion matrix on validation set."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Computing Confusion"):
            imgs, meta, labels, _, _ = batch_data
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)
            outputs = model(imgs, meta)
            species_logits = outputs['species']  # Extract species predictions
            _, preds = torch.max(species_logits, 1)
            
            for true, pred in zip(labels.cpu(), preds.cpu()):
                confusion[true, pred] += 1
    
    return confusion


# --- 7. TTA VALIDATION ---
def validate_with_tta(model, dataloader, device, num_augmentations=5):
    """Validation with Test-Time Augmentation."""
    model.eval()
    correct = 0
    total = 0
    
    tta_transforms = [
        lambda x: x,  # Original
        transforms.functional.hflip,  # Horizontal flip
        lambda x: transforms.functional.rotate(x, 10),
        lambda x: transforms.functional.rotate(x, -10),
        lambda x: transforms.functional.adjust_brightness(x, 1.2),
    ]
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="TTA Validation"):
            imgs, meta, labels, _, _ = batch_data
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)
            
            # Average predictions across augmentations
            ensemble_logits = None
            for aug in tta_transforms:
                aug_imgs = torch.stack([aug(img) for img in imgs]).to(device)
                outputs = model(aug_imgs, meta)
                logits = outputs['species']  # Extract species predictions
                if ensemble_logits is None:
                    ensemble_logits = logits
                else:
                    ensemble_logits += logits
            
            ensemble_logits /= len(tta_transforms)
            _, preds = torch.max(ensemble_logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return correct / total


# --- 8. MAIN TRAINING FUNCTION ---
def train_model(model_type='b3'):
    """Train a single model with all Oogway enhancements."""
    print(f"\n{'='*70}")
    print(f"🐢 OOGWAY TRAINING: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.is_available()}")
    print(f"Enhancements: SWA-Fixed | Focal Loss | Confusion Weights | Hard Neg Mining")
    print(f"              TTA Validation | Metadata Dropout | CutMix | Extended FixRes")
    
    # Data transforms
    base_size = BASE_SIZE_B3 if model_type == 'b3' else BASE_SIZE_CONVNEXT
    train_transform = transforms.Compose([
        transforms.Resize((int(base_size * 1.15), int(base_size * 1.15))),
        transforms.RandomResizedCrop(base_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(base_size * 1.1)),
        transforms.CenterCrop(base_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'train'), train_transform, is_train=True)
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'val'), val_transform, is_train=False)
    
    # Check if empty or mismatched
    if len(train_ds) == 0:
        print("❌ CRITICAL: No training data found!")
        return 0, 0
    
    # Hard Negative Mining sampler
    train_sampler = HardNegativeSampler(train_ds, BATCH_SIZE, num_pairs_per_batch=3)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
    
    num_classes = len(train_ds.classes)
    print(f"✅ {len(train_ds)} train, {len(val_ds)} val | {num_classes} classes")
    print(f"🎯 Hard Negative Mining: {len(CONFUSED_PAIRS)} pairs (sqrt-smoothed)")
    
    # Model setup
    print(f"\n🧠 Initializing {model_type.upper()} Hierarchical Late Fusion Model...")
    num_species = len(train_ds.classes)
    num_families = len(train_ds.families)
    num_tax_classes = len(train_ds.tax_classes)
    model = WildlifeLateFusion(num_species, num_families, num_tax_classes, model_type=model_type).to(DEVICE)
    
   # Optimizer & Schedulers
    base_lr = 1e-4 if model_type == 'convnext' else 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    
    # Hierarchical Multi-Task Loss
    criterion = HierarchicalMultiTaskLoss(train_ds.classes, gamma=FOCAL_GAMMA, label_smoothing=0.1)
    
    # SWA setup (FIXED)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)  # 50x lower than Grandmaster!
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SWA_START_EPOCH)
    
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")
        
        model.train()
        running_loss = 0.0
        
        for batch_data in tqdm(train_loader, desc="Training"):
            imgs, meta, labels, family_labels, class_labels = batch_data
            imgs = imgs.to(DEVICE)
            meta = meta.to(DEVICE) 
            labels = labels.to(DEVICE)
            family_labels = family_labels.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            
            # Apply MixUp or CutMix (50/50 chance)
            if random.random() < 0.5:
                # MixUp (also mixes metadata!)
                imgs, meta, labels_a, labels_b, lam = mixup_data(imgs, meta, labels, MIXUP_ALPHA)
                family_a, family_b = family_labels, family_labels  # For simplicity, use same mix
                class_a, class_b = class_labels, class_labels
            else:
                # CutMix (image only, metadata unchanged)
                imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, CUTMIX_ALPHA)
                family_a, family_b = family_labels, family_labels
                class_a, class_b = class_labels, class_labels
            
            optimizer.zero_grad()
            outputs = model(imgs, meta)
            
            # Compute loss for both mixed samples
            loss_a, _ = criterion(outputs, labels_a, family_a, class_a)
            loss_b, _ = criterion(outputs, labels_b, family_b, class_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                imgs, meta, labels, _, _ = batch_data
                imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs, meta)
                species_logits = outputs['species']  # Extract species predictions
                _, predicted = torch.max(species_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        print(f"Val Acc: {acc:.4f} ({acc*100:.2f}%)")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': train_ds.classes,
                'epoch': epoch,
                'accuracy': acc
            }, f"oogway_{model_type}_best.pth")
            print(f"⭐ New Best! (Epoch {epoch+1})")
        
        # Update confusion weights every 3 epochs
        if (epoch + 1) % CONFUSION_UPDATE_INTERVAL == 0 and epoch < SWA_START_EPOCH:
            print(f"\n🔄 Updating confusion-weighted loss...")
            confusion = compute_confusion_matrix(model, val_loader, num_classes, DEVICE)
            criterion.update_weights(confusion, DEVICE)
        
        # SWA logic (FIXED)
        if (epoch + 1) >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print("🧠 SWA Update (averaging weights)")
        else:
            scheduler.step()
    
    print(f"\n{'='*70}")
    print(f"BEST VALIDATION: {best_acc:.4f} at Epoch {best_epoch+1}")
    print(f"{'='*70}")
    
    # Update BatchNorm for SWA model (CRITICAL FIX)
    print("\n🔄 Updating BatchNorm statistics for SWA model...")
    update_bn(train_loader, swa_model, device=DEVICE)
    
    # Save SWA model (FIXED - this was missing in Grandmaster!)
    torch.save({
        'model_state_dict': swa_model.module.state_dict(),
        'class_names': train_ds.classes,
        'best_accuracy': best_acc
    }, f"oogway_{model_type}_swa.pth")
    print(f"💾 Saved SWA model: oogway_{model_type}_swa.pth")
    
    # FixRes Fine-Tuning (Extended to 3 epochs)
    print(f"\n{'='*70}")
    print(f"📏 STARTING FIXRES FINE-TUNING ({FIXRES_SIZE}px, {FIXRES_EPOCHS} epochs)")
    print(f"{'='*70}")
    
    # Load best model (or SWA model)
    model.load_state_dict(swa_model.module.state_dict())
    
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    
    fixres_transform = transforms.Compose([
        transforms.Resize(FIXRES_SIZE),
        transforms.CenterCrop(FIXRES_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds.transform = fixres_transform
    fixres_loader = DataLoader(train_ds, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=4)
    
    optimizer_fixres = optim.SGD(model.classifier.parameters(), lr=1e-4, momentum=0.9)
    
    for fixres_epoch in range(FIXRES_EPOCHS):
        print(f"\nFixRes Epoch {fixres_epoch+1}/{FIXRES_EPOCHS}")
        model.train()
        for batch_data in tqdm(fixres_loader, desc=f"FixRes E{fixres_epoch+1}"):
            imgs, meta, labels, family_labels, class_labels = batch_data
            imgs = imgs.to(DEVICE)
            meta = meta.to(DEVICE)
            labels = labels.to(DEVICE)
            family_labels = family_labels.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            optimizer_fixres.zero_grad()
            outputs = model(imgs, meta)
            loss, _ = criterion(outputs, labels, family_labels, class_labels)
            loss.backward()
            optimizer_fixres.step()
    
    # Final validation with TTA
    print("\n🔍 Final TTA Validation...")
    val_ds.transform = val_transform
    val_loader_tta = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    tta_acc = validate_with_tta(model, val_loader_tta, DEVICE)
    print(f"✅ TTA Validation Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_ds.classes,
        'best_accuracy': best_acc,
        'tta_accuracy': tta_acc
    }, f"oogway_{model_type}_final.pth")
    
    print(f"\n🏆 OOGWAY {model_type.upper()} TRAINING COMPLETE!")
    print(f"   Best Val Acc: {best_acc:.4f}")
    print(f"   TTA Val Acc: {tta_acc:.4f}")
    print(f"   Saved: oogway_{model_type}_final.pth")
    
    return best_acc, tta_acc


# --- 9. MAIN ENTRY POINT ---
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("🐢 OOGWAY TRAINING - Next Generation Wildlife Classifier")
    print(f"{'='*70}")
    
    results = {}
    
    # Train B3
    try:
        print("\n\n" + "="*70)
        print("TRAINING MODEL 1/2: EfficientNet-B3")
        acc_b3, tta_b3 = train_model('b3')
        results['b3'] = {'acc': acc_b3, 'tta': tta_b3}
    except Exception as e:
        print(f"❌ Error training B3: {e}")
        import traceback
        traceback.print_exc()
        
    # Train ConvNeXt
    # Uncomment if needed, usually we train one at a time to save time
    # print("\n\n" + "="*70)
    # print("TRAINING MODEL 2/2: ConvNeXt-Tiny")
    # acc_cx, tta_cx = train_model('convnext')
    # results['convnext'] = {'acc': acc_cx, 'tta': tta_cx}
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print(json.dumps(results, indent=2))
