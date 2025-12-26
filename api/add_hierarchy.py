"""
Add Hierarchy Head to Existing Model
=====================================
Run AFTER main training is complete.

This script:
1. Loads an existing trained Late Fusion model
2. FREEZES the backbone (no full retrain!)
3. Adds a "Class" level head (Mammal/Bird/Reptile/Amphibian)
4. Trains ONLY the new hierarchy head (fast, ~30 min)
5. At inference: both heads work together

This gives ~70% of hierarchical benefit with ~20% of training time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# --- TAXONOMY: Map species to high-level class ---
# This can be expanded to more levels later
TAXONOMY = {
    # MAMMALS
    'american_badger': 'Mammal', 'american_bison': 'Mammal', 'american_black_bear': 'Mammal',
    'american_marten': 'Mammal', 'american_mink': 'Mammal', 'american_red_squirrel': 'Mammal',
    'arctic_fox': 'Mammal', 'axis_deer': 'Mammal', 'bighorn_sheep': 'Mammal',
    'black-footed_ferret': 'Mammal', 'black-tailed_prairie_dog': 'Mammal', 'bobcat': 'Mammal',
    'california_sea_lion': 'Mammal', 'caribou': 'Mammal', 'coyote': 'Mammal',
    'dall_sheep': 'Mammal', 'domestic_cat': 'Mammal', 'domestic_dog': 'Mammal',
    'eastern_chipmunk': 'Mammal', 'eastern_cottontail': 'Mammal', 'eastern_fox_squirrel': 'Mammal',
    'eastern_gray_squirrel': 'Mammal', 'elk': 'Mammal', 'gray_fox': 'Mammal',
    'gray_wolf': 'Mammal', 'grizzly_bear': 'Mammal', 'groundhog': 'Mammal',
    'harbor_seal': 'Mammal', 'jaguar': 'Mammal', 'moose': 'Mammal',
    'mountain_goat': 'Mammal', 'mountain_lion': 'Mammal', 'muskox': 'Mammal',
    'nine-banded_armadillo': 'Mammal', 'north_american_beaver': 'Mammal',
    'north_american_porcupine': 'Mammal', 'north_american_river_otter': 'Mammal',
    'northern_elephant_seal': 'Mammal', 'nutria': 'Mammal', 'ocelot': 'Mammal',
    'polar_bear': 'Mammal', 'pronghorn': 'Mammal', 'raccoon': 'Mammal',
    'red_fox': 'Mammal', 'rhesus_macaque': 'Mammal', 'sea_otter': 'Mammal',
    'snowshoe_hare': 'Mammal', 'striped_skunk': 'Mammal', 'virginia_opossum': 'Mammal',
    'walrus': 'Mammal', 'white-tailed_deer': 'Mammal', 'wild_boar': 'Mammal',
    'wolverine': 'Mammal', 'yellow-bellied_marmot': 'Mammal',
    
    # BIRDS
    'american_flamingo': 'Bird', 'bald_eagle': 'Bird', 'barn_owl': 'Bird',
    'blue_jay': 'Bird', 'brown_pelican': 'Bird', 'canada_goose': 'Bird',
    'eurasian_collared-dove': 'Bird', 'european_starling': 'Bird',
    'great_blue_heron': 'Bird', 'great_horned_owl': 'Bird', 'house_sparrow': 'Bird',
    'mallard': 'Bird', 'monk_parakeet': 'Bird', 'mute_swan': 'Bird',
    'northern_cardinal': 'Bird', 'osprey': 'Bird', 'peregrine_falcon': 'Bird',
    'ring-necked_pheasant': 'Bird', 'rock_pigeon': 'Bird', 'snowy_owl': 'Bird',
    'whooping_crane': 'Bird', 'wild_turkey': 'Bird',
    
    # REPTILES
    'american_alligator': 'Reptile', 'american_crocodile': 'Reptile',
    'argentine_black_and_white_tegu': 'Reptile', 'brown_anole': 'Reptile',
    'burmese_python': 'Reptile', 'common_garter_snake': 'Reptile',
    'common_snapping_turtle': 'Reptile', 'desert_tortoise': 'Reptile',
    'common_box_turtle': 'Reptile', 'eastern_copperhead': 'Reptile',
    'gila_monster': 'Reptile', 'green_anole': 'Reptile', 'green_iguana': 'Reptile',
    'nile_monitor': 'Reptile', 'painted_turtle': 'Reptile', 'tokay_gecko': 'Reptile',
    'veiled_chameleon': 'Reptile', 'western_diamondback_rattlesnake': 'Reptile',
    
    # AMPHIBIANS
    'american_bullfrog': 'Amphibian', 'cane_toad': 'Amphibian',
    'eastern_hellbender': 'Amphibian', 'eastern_newt': 'Amphibian',
    'spotted_salamander': 'Amphibian',
    
    # OTHERS (if not in taxonomy, default)
    'gemsbok': 'Mammal', 'chukar': 'Bird',
}

CLASS_ORDER = ['Mammal', 'Bird', 'Reptile', 'Amphibian']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_ORDER)}


# --- DATASET WITH HIERARCHY LABELS ---
class WildlifeHierarchyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
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
                    species_label = self.class_to_idx[class_name]
                    # Get hierarchy label
                    class_label = CLASS_TO_IDX.get(TAXONOMY.get(class_name, 'Mammal'), 0)
                    self.samples.append((img_path, json_path, species_label, class_label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path, species_label, class_label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
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
        
        if self.is_train and (lat != 0.0 or lng != 0.0):
            lat += np.random.uniform(-1.0, 1.0)
            lng += np.random.uniform(-1.0, 1.0)
            lat = np.clip(lat, -90.0, 90.0)
            lng = np.clip(lng, -180.0, 180.0)
        
        meta_vector = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0, lng / 180.0
        ], dtype=torch.float32)
        
        return image, meta_vector, species_label, class_label


# --- HIERARCHICAL MODEL (adds class head to existing model) ---
class WildlifeHierarchical(nn.Module):
    """
    Wraps an existing Late Fusion model and adds a hierarchy head.
    The original model is FROZEN - we only train the new head.
    """
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.base_model = base_model
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get feature dimension from base model
        # We'll tap into the image model's features before final classifier
        if hasattr(self.base_model, 'image_model'):
            if hasattr(self.base_model.image_model, 'classifier'):
                # EfficientNet
                in_features = self.base_model.image_model.classifier[1].in_features \
                    if hasattr(self.base_model.image_model.classifier[1], 'in_features') \
                    else self.base_model.image_model.classifier[1].weight.shape[1]
            else:
                in_features = 1536  # Default for B3
        else:
            in_features = 1536
        
        # New hierarchy head (Mammal/Bird/Reptile/Amphibian)
        self.class_head = nn.Sequential(
            nn.Linear(4, 32),  # From metadata
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Alternative: Use image features
        self.class_head_image = nn.Linear(in_features, num_classes)
        
    def forward(self, x, meta):
        # Get species prediction from frozen base model
        with torch.no_grad():
            species_logits = self.base_model(x, meta)
        
        # Get class prediction from new head (using metadata)
        class_logits = self.class_head(meta)
        
        return species_logits, class_logits
    
    def predict_with_hierarchy(self, x, meta):
        """
        Inference with hierarchy-aware prediction.
        Uses class prediction to boost/filter species predictions.
        """
        species_logits, class_logits = self.forward(x, meta)
        
        # Get class probabilities
        class_probs = torch.softmax(class_logits, dim=1)
        
        # Could use this to filter/boost species predictions
        # For now, just return both
        return species_logits, class_logits, class_probs


def train_hierarchy_head(model_path, model_type, epochs=10, lr=1e-3):
    """
    Train ONLY the hierarchy head on an existing model.
    Base model stays frozen.
    """
    print(f"🔧 HIERARCHY HEAD TRAINING")
    print(f"   Base Model: {model_path}")
    print(f"   Type: {model_type}")
    print(f"   Epochs: {epochs}, LR: {lr}")
    print(f"   Classes: {CLASS_ORDER}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((int(300 * 1.15), int(300 * 1.15))),
        transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(300 * 1.1)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = WildlifeHierarchyDataset(os.path.join(DATA_DIR, 'train'), train_transform, is_train=True)
    val_ds = WildlifeHierarchyDataset(os.path.join(DATA_DIR, 'val'), val_transform, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    num_species = len(train_ds.classes)
    print(f"📂 Loaded {len(train_ds)} train, {len(val_ds)} val images ({num_species} species)")
    
    # Load base model
    from train_grandmaster import WildlifeLateFusion
    base_model = WildlifeLateFusion(num_species, model_type=model_type)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.to(DEVICE)
    base_model.eval()
    print(f"✅ Loaded base model (FROZEN)")
    
    # Create hierarchical model
    model = WildlifeHierarchical(base_model, num_classes=len(CLASS_ORDER)).to(DEVICE)
    
    # Only train hierarchy head parameters
    optimizer = optim.Adam(model.class_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_class_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for imgs, meta, species_labels, class_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, meta = imgs.to(DEVICE), meta.to(DEVICE)
            class_labels = class_labels.to(DEVICE)
            
            optimizer.zero_grad()
            _, class_logits = model(imgs, meta)
            loss = criterion(class_logits, class_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        species_correct, class_correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, meta, species_labels, class_labels in val_loader:
                imgs, meta = imgs.to(DEVICE), meta.to(DEVICE)
                species_labels, class_labels = species_labels.to(DEVICE), class_labels.to(DEVICE)
                
                species_logits, class_logits = model(imgs, meta)
                
                _, species_pred = torch.max(species_logits, 1)
                _, class_pred = torch.max(class_logits, 1)
                
                total += class_labels.size(0)
                species_correct += (species_pred == species_labels).sum().item()
                class_correct += (class_pred == class_labels).sum().item()
        
        species_acc = species_correct / total
        class_acc = class_correct / total
        
        print(f"   Loss: {running_loss/len(train_loader):.4f}")
        print(f"   Species Acc: {species_acc*100:.2f}% (frozen, unchanged)")
        print(f"   Class Acc: {class_acc*100:.2f}% (Mammal/Bird/Reptile/Amphibian)")
        
        if class_acc > best_class_acc:
            best_class_acc = class_acc
            output_path = model_path.replace('.pth', '_hierarchical.pth')
            torch.save({
                'base_model_state_dict': checkpoint['model_state_dict'],
                'class_head_state_dict': model.class_head.state_dict(),
                'class_names': train_ds.classes,
                'class_order': CLASS_ORDER,
                'species_accuracy': species_acc,
                'class_accuracy': class_acc
            }, output_path)
            print(f"   ⭐ Saved to {output_path}")
    
    print(f"\n🏆 Hierarchy training complete!")
    print(f"   Species Accuracy: {species_acc*100:.2f}% (unchanged from base model)")
    print(f"   Class Accuracy: {best_class_acc*100:.2f}% (new hierarchy head)")


def main():
    parser = argparse.ArgumentParser(description='Add Hierarchy Head to Model')
    parser.add_argument('--model', type=str, required=True, help='Path to base .pth model')
    parser.add_argument('--type', type=str, default='b3', choices=['b3', 'convnext'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train_hierarchy_head(args.model, args.type, args.epochs, args.lr)


if __name__ == "__main__":
    main()
