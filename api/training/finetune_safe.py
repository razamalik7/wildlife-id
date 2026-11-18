"""
Safe Overnight Fine-Tuning Script
==================================
Ultra-conservative approach: Low LR, no hard negatives, just gentle exposure to refined data.

Usage:
  python finetune_safe.py --model oogway_b3_best.pth --type b3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image
import os, json, random
from tqdm import tqdm
import argparse

# --- CONFIGURATION (SAFE) ---
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Low augmentation
MIXUP_ALPHA = 0.1  # Reduced from 0.2

# --- SIMPLE DATASET ---
class SimpleWildlifeDataset(Dataset):
    def __init__(self, root_dirs, transform, is_train=True):
        self.transform = transform
        self.samples = []
        self.classes = []
        
        all_classes = set()
        for root in root_dirs:
            split = 'train' if is_train else 'val'
            split_path = os.path.join(BASE_DIR, root, split)
            if not os.path.exists(split_path):
                continue
            for cls in os.listdir(split_path):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    all_classes.add(cls)
                    for f in os.listdir(cls_path):
                        if f.endswith('.jpg'):
                            self.samples.append((os.path.join(cls_path, f), cls))
        
        self.classes = sorted(list(all_classes))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, cls = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.class_to_idx[cls]
        return img, label

# --- MODEL ---
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes, model_type='b3'):
        super().__init__()
        if model_type == 'b3':
            self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
        else:  # convnext
            self.backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                self.backbone.classifier[0],  # LayerNorm
                self.backbone.classifier[1],  # Flatten
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

# --- MIXUP ---
def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    idx = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- MAIN ---
def fine_tune(args):
    print(f"🌙 SAFE OVERNIGHT FINE-TUNING")
    print(f"   Model: {args.model}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Epochs: {args.epochs}")
    
    # Transforms
    if args.type == 'b3':
        tf = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tf = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Load datasets
    print("📂 Loading datasets...")
    train_ds = SimpleWildlifeDataset(args.roots, tf, is_train=True)
    val_ds = SimpleWildlifeDataset(args.roots, val_tf, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"✅ Loaded {len(train_ds)} train, {len(val_ds)} val samples ({len(train_ds.classes)} classes)")
    
    # Model
    model = SimpleClassifier(len(train_ds.classes), args.type).to(DEVICE)
    
    # Load pretrained weights
    if os.path.exists(args.model):
        print(f"📦 Loading weights from {args.model}...")
        ckpt = torch.load(args.model, map_location=DEVICE)
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        # Filter and load compatible weights
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            key = k.replace('module.', '')
            # Map old keys to new keys
            if 'image_model.' in key:
                new_key = key.replace('image_model.', 'backbone.')
                if new_key in model_dict and v.shape == model_dict[new_key].shape:
                    pretrained_dict[new_key] = v
            elif key in model_dict and v.shape == model_dict[key].shape:
                pretrained_dict[key] = v
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"   Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
    
    # Optimizer with very low LR
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # SWA
    swa_model = AveragedModel(model)
    swa_start = args.epochs // 2
    
    best_acc = 0.0
    save_path = args.model.replace('.pth', '_safe_finetuned.pth')
    
    print(f"\n🚀 Starting Safe Fine-Tuning ({args.epochs} Epochs)...")
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Mixup with low alpha
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
        # SWA update
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        acc = correct / total
        scheduler.step(acc)
        print(f"   Val Acc: {acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if acc > best_acc:
            best_acc = acc
            state = swa_model.module.state_dict() if epoch >= swa_start else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'accuracy': acc,
                'epoch': epoch,
                'classes': train_ds.classes
            }, save_path)
            print(f"   ⭐ Best Model Saved: {save_path} ({acc*100:.2f}%)")
    
    print(f"\n✅ Training Complete. Best Accuracy: {best_acc*100:.2f}%")
    print(f"   Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--type', default='b3')
    parser.add_argument('--roots', nargs='+', default=['training_data_v2', 'training_data_refined'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)  # Very low LR
    args = parser.parse_args()
    fine_tune(args)
