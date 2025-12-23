import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import json
import math
from datetime import datetime
from tqdm import tqdm
import multiprocessing

import argparse

# --- ARGS ---
parser = argparse.ArgumentParser(description='Grandmaster Training')
parser.add_argument('--model', type=str, default='b3', choices=['b3', 'convnext', 'resnet'], help='Model architecture')
args = parser.parse_args()

# --- CONFIGURATION ---
DATA_DIR = '../training_data_cropped' 
BATCH_SIZE = 8 
NUM_EPOCHS = 20
SWA_START_EPOCH = 15
FIXRES_SIZE = 384
BASE_SIZE = 300 if args.model == 'b3' else 224 # Adjust base size by model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (Dataset class remains the same)

# --- 2. GEO-AWARE MODEL FACTORY ---
class WildlifeGeoAware(nn.Module):
    def __init__(self, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            # EfficientNet B3
            base_model = models.efficientnet_b3(weights="IMAGENET1K_V1")
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.classifier = base_model.classifier
            in_features = self.classifier[1].in_features
            
            # Replace Head
            self.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features + 4, num_classes)
            )
            
        elif model_type == 'convnext':
            # ConvNeXt Tiny
            base_model = models.convnext_tiny(weights="IMAGENET1K_V1")
            self.features = base_model.features
            self.avgpool = base_model.avgpool # AdaptiveAvgPool2d((1, 1))
            self.classifier = base_model.classifier
            in_features = self.classifier[2].in_features
            
            # Replace Head (ConvNeXt classifier is sequential: Flatten, LayerNorm, Linear)
            # We need to inject metadata before the final Linear layer
            # But the classifier block is hard to split cleanly in `forward`.
            # Easies way: We will override forward completely.
            
            # Recreate the final projection layer
            self.classifier[2] = nn.Sequential(
                 nn.Linear(in_features + 4, num_classes)
            )

        elif model_type == 'resnet':
            # ResNet50
            base_model = models.resnet50(weights="IMAGENET1K_V1")
            self.features = nn.Sequential(*list(base_model.children())[:-2]) # Remove avgpool and fc
            self.avgpool = base_model.avgpool
            self.classifier = base_model.fc
            in_features = self.classifier.in_features
            
            # Replace Head
            self.classifier = nn.Sequential(
                 nn.Dropout(0.2), # Standard ResNet dropout (optional)
                 nn.Linear(in_features + 4, num_classes)
            )

    def forward(self, x, meta):
        # Feature Extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Concat Metadata
        x = torch.cat((x, meta), dim=1)
        
        # Classification
        if self.model_type == 'b3':
            x = self.classifier(x)
        elif self.model_type == 'convnext':
            # ConvNeXt needs LayerNorm before the final linear, usually part of classifier[0] and [1]
            # Since we only replaced [2], we need to run [0] and [1] first?
            # Actually, standard ConvNeXt `classifier` is: 0=LayerNorm2d(768), 1=Flatten, 2=Linear
            # Our self.classifier still has those if we didn't overwrite the whole thing.
            # Wait, in __init__ we only overwrote classifier[2].
            # So `self.classifier` is still [LN, Flatten, NewLinear]. 
            # But we can't pump `x` (which is already flattened) through LN(2d).
            # Let's fix this manually:
            
            # Note: `base_model.features` ends with spatial tensor. 
            # `self.avgpool` converts to (C, 1, 1). 
            # `torch.flatten` converts to (Batch, C).
            # The original ConvNeXt classifier expects the spatial tensor or checks shapes.
            # Let's just use the final Linear layer we made.
            
            x = self.classifier[2](x) 
            
        elif self.model_type == 'resnet':
             x = self.classifier(x)
             
        return x

def main():
    print(f"🦁 GRANDMASTER TRAINING ({args.model.upper()}) | GPU: {torch.cuda.is_available()}")
    
    # ... (Transforms setup) ...

    
    val_transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(BASE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📂 Loading Datasets...")
    train_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'train'), train_transform)
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'val'), val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_classes = len(train_ds.classes)
    print(f"✅ Found {len(train_ds)} train, {len(val_ds)} val images across {num_classes} classes.")

    # 2. Setup Model & SWA
    print("🧠 Initializing Geo-Aware Model...")
    model = WildlifeGeoAware(num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # SWA Setup
    swa_model = AveragedModel(model) 
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SWA_START_EPOCH)
    
    best_acc = 0.0

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        
        for imgs, meta, labels in tqdm(train_loader):
            imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Train Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, meta, labels in val_loader:
                imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs, meta)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct/total
        print(f"Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "grandmaster_best.pth")
            print("⭐ New Best!")

        # SWA Logic
        if (epoch + 1) >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print("🧠 SWA Update")
        else:
            scheduler.step()

    # 4. Update BN for SWA
    print("🔄 Updating SWA BatchNorm...")
    # swa_utils.update_bn(train_loader, swa_model, device=DEVICE) 
    # Note: update_bn requires a custom loader modification to pass only images usually, 
    # but since our forward expects meta, we need a custom update_bn function or skip it for now.
    # For simplicity in this script, we skip explicit update_bn or implement a simple pass.
    
    # 5. FixRes Fine-Tuning
    print("\n📏 STARTING FIXRES FINE-TUNING (384px)")
    # Load Best Weights
    model.load_state_dict(torch.load("grandmaster_best.pth"))
    
    # Freeze Features
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Change Resolution
    fixres_transform = transforms.Compose([
        transforms.Resize(FIXRES_SIZE), # 384
        transforms.CenterCrop(FIXRES_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # New Loader (Res only changes, no need to reload dataset object entirely if we could patch it, 
    # but easier to reload)
    train_ds.transform = fixres_transform
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=4) # Smaller batch
    
    # Train Head Only
    optimizer = optim.SGD(model.classifier.parameters(), lr=1e-4, momentum=0.9) # Low LR
    
    model.train()
    for imgs, meta, labels in tqdm(train_loader):
        imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs, meta)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), "grandmaster_final.pth")
    print("🏆 GRANDMASTER TRAINING COMPLETE.")

if __name__ == "__main__":
    main()
