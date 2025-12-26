"""
Targeted Fine-Tuning Script (with Focal Loss + Safety Monitoring)
==================================================================
Run AFTER main Grandmaster training is complete on BOTH models.

Safety Features:
1. Per-class accuracy monitoring - warns if any class drops significantly
2. Validates on original (non-augmented) data to detect over-prediction
3. Focal Loss for automatic hard example mining

Usage:
  python targeted_finetune.py --model grandmaster_b3_final.pth --type b3
  python targeted_finetune.py --model grandmaster_convnext_final.pth --type convnext
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from collections import defaultdict

# Config
# Updated to v2 (Oogway era)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data_v2')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- FOCAL LOSS (Auto-focuses on hard examples) ---
class FocalLoss(nn.Module):
    """
    Focal Loss for automatic hard example mining.
    
    - Easy samples (high confidence correct): Very low weight
    - Hard samples (low confidence / confused): High weight
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


# --- DATASET ---
class WildlifeGeoDataset(torch.utils.data.Dataset):
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
                    self.samples.append((img_path, json_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]
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
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32)
        
        return image, meta_vector, label


# --- LATE FUSION MODEL ---
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
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, meta):
        image_logits = self.image_model(x)
        meta_logits = self.meta_mlp(meta)
        return image_logits + self.fusion_weight * meta_logits


def evaluate_per_class(model, dataloader, class_names, device):
    """
    Evaluate model and return per-class accuracy.
    Used to detect over-prediction of specific classes.
    """
    model.eval()
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for imgs, meta, labels in dataloader:
            imgs, meta, labels = imgs.to(device), meta.to(device), labels.to(device)
            outputs = model(imgs, meta)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
    
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        if class_total[idx] > 0:
            per_class_acc[name] = class_correct[idx] / class_total[idx]
        else:
            per_class_acc[name] = 0.0
    
    return per_class_acc


def check_for_overprediction(baseline_acc, current_acc, threshold=0.10):
    """
    Check if any class has dropped significantly (potential over-prediction happening).
    Returns list of classes that dropped by more than threshold.
    """
    warnings = []
    
    for class_name in baseline_acc:
        if class_name in current_acc:
            drop = baseline_acc[class_name] - current_acc[class_name]
            if drop > threshold:
                warnings.append({
                    'class': class_name,
                    'baseline': baseline_acc[class_name],
                    'current': current_acc[class_name],
                    'drop': drop
                })
    
    return warnings


def fine_tune(model_path, model_type, epochs=5, lr=1e-4, gamma=2.0):
    """
    Fine-tune an existing model with Focal Loss and safety monitoring.
    """
    print(f"🔧 TARGETED FINE-TUNING (with Safety Monitoring)")
    print(f"   Model: {model_path}")
    print(f"   Type: {model_type}")
    print(f"   Epochs: {epochs}, LR: {lr}, Gamma: {gamma}")
    print(f"   🛡️ Safety: Per-class monitoring + over-prediction detection")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((int(300 * 1.15), int(300 * 1.15))),
        transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(300 * 1.1)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load data
    train_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'train'), train_transform, is_train=True)
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'val'), val_transform, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    print(f"📂 Loaded {len(train_ds)} train, {len(val_ds)} val images ({num_classes} classes)")
    
    # Load existing model
    model = WildlifeLateFusion(num_classes, model_type=model_type).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded weights from {model_path}")
    
    # ===== SAFETY CHECK 1: Get baseline per-class accuracy =====
    print("\n📊 Computing baseline per-class accuracy...")
    baseline_per_class = evaluate_per_class(model, val_loader, class_names, DEVICE)
    baseline_overall = sum(baseline_per_class.values()) / len(baseline_per_class)
    print(f"   Baseline overall accuracy: {baseline_overall*100:.2f}%")
    
    # Find initially weak classes (for comparison later)
    weak_classes = sorted(baseline_per_class.items(), key=lambda x: x[1])[:10]
    print(f"\n   Bottom 10 classes (these should improve):")
    for name, acc in weak_classes:
        print(f"      {name.replace('_', ' ')}: {acc*100:.1f}%")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = FocalLoss(gamma=gamma, label_smoothing=0.1)
    
    best_acc = baseline_overall
    best_model_state = None
    early_stop = False
    
    print(f"\n🚀 Starting fine-tuning with Focal Loss (γ={gamma})...")
    
    # Fine-tuning loop
    for epoch in range(epochs):
        if early_stop:
            break
            
        model.train()
        running_loss = 0.0
        
        for imgs, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        # ===== SAFETY CHECK 2: Per-class evaluation =====
        current_per_class = evaluate_per_class(model, val_loader, class_names, DEVICE)
        current_overall = sum(current_per_class.values()) / len(current_per_class)
        
        print(f"\n   Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Accuracy={current_overall*100:.2f}%")
        
        # Check for over-prediction (any class dropped by >10%)
        warnings = check_for_overprediction(baseline_per_class, current_per_class, threshold=0.10)
        
        if warnings:
            print(f"\n   ⚠️ OVER-PREDICTION WARNING! These classes dropped significantly:")
            for w in warnings[:5]:
                print(f"      {w['class'].replace('_', ' ')}: {w['baseline']*100:.1f}% → {w['current']*100:.1f}% (↓{w['drop']*100:.1f}%)")
            
            if len(warnings) > 5:
                print(f"   ⚠️ Stopping early to prevent over-fitting!")
                early_stop = True
        
        # Check if weak classes improved
        improved_count = 0
        for name, baseline_acc in weak_classes:
            if current_per_class[name] > baseline_acc + 0.02:  # 2% improvement
                improved_count += 1
        
        print(f"   📈 Weak classes improved: {improved_count}/{len(weak_classes)}")
        
        # Save best model
        if current_overall > best_acc and len(warnings) == 0:
            best_acc = current_overall
            best_model_state = model.state_dict().copy()
            output_path = model_path.replace('.pth', '_finetuned.pth')
            torch.save({
                'model_state_dict': best_model_state,
                'class_names': class_names,
                'accuracy': best_acc,
                'baseline_accuracy': baseline_overall,
                'improved_from': model_path,
                'fine_tuned_with': 'focal_loss',
                'per_class_accuracy': current_per_class
            }, output_path)
            print(f"   ⭐ New Best! Saved to {output_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🏆 FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"   Baseline accuracy: {baseline_overall*100:.2f}%")
    print(f"   Best accuracy:     {best_acc*100:.2f}%")
    print(f"   Improvement:       {(best_acc - baseline_overall)*100:+.2f}%")
    
    if best_model_state:
        # Final per-class check on best model
        model.load_state_dict(best_model_state)
        final_per_class = evaluate_per_class(model, val_loader, class_names, DEVICE)
        
        # Show which weak classes improved
        print(f"\n   Weak class improvements:")
        for name, baseline_acc in weak_classes:
            current = final_per_class[name]
            change = current - baseline_acc
            arrow = "↑" if change > 0 else "↓" if change < 0 else "="
            print(f"      {name.replace('_', ' ')}: {baseline_acc*100:.1f}% → {current*100:.1f}% ({arrow}{abs(change)*100:.1f}%)")
    
    if early_stop:
        print(f"\n   ⚠️ Training stopped early due to over-prediction detection")


def main():
    parser = argparse.ArgumentParser(description='Targeted Fine-Tuning with Safety Monitoring')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--type', type=str, default='b3', choices=['b3', 'convnext'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    args = parser.parse_args()
    
    fine_tune(args.model, args.type, args.epochs, args.lr, args.gamma)


if __name__ == "__main__":
    main()
