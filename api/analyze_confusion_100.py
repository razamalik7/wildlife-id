"""
Quick Confusion Matrix Analysis for Grandmaster Model
Outputs top 25 confused pairs (bidirectional merged)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os, json, math
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

DATA_DIR = './training_data_cropped'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WildlifeGeoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir): continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    self.samples.append((img_path, json_path, self.class_to_idx[class_name]))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        lat, lng, month = 0.0, 0.0, 6
        if os.path.exists(json_path):
            try:
                meta = json.load(open(json_path))
                lat = meta.get('lat', 0.0) or 0.0
                lng = meta.get('lng', 0.0) or 0.0
                date_str = meta.get('date', '')
                if date_str:
                    try: month = datetime.strptime(date_str, '%Y-%m-%d').month
                    except: pass
            except: pass
        meta_vector = torch.tensor([
            math.sin(2*math.pi*month/12), 
            math.cos(2*math.pi*month/12), 
            lat/90.0, lng/180.0
        ], dtype=torch.float32)
        return image, meta_vector, label

class WildlifeLateFusion(nn.Module):
    def __init__(self, num_classes, model_type='convnext'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights=None)
            in_features = self.image_model.classifier[2].in_features
            self.image_model.classifier[2] = nn.Linear(in_features, num_classes)
        else:
            self.image_model = models.efficientnet_b3(weights=None)
            in_features = self.image_model.classifier[1].in_features
            self.image_model.classifier[1] = nn.Linear(in_features, num_classes)
        self.meta_mlp = nn.Sequential(
            nn.Linear(4,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, meta):
        return self.image_model(x) + self.fusion_weight * self.meta_mlp(meta)

if __name__ == '__main__':
    print("Loading validation data...")
    val_transform = transforms.Compose([
        transforms.Resize(int(300*1.1)), 
        transforms.CenterCrop(300), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR,'val'), val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)  # num_workers=0 for Windows
    num_classes = len(val_ds.classes)
    print(f"Loaded {len(val_ds)} images across {num_classes} classes")

    print("Loading ConvNeXt model...")
    model = WildlifeLateFusion(num_classes, 'convnext').to(DEVICE)
    ckpt = torch.load('grandmaster_convnext_best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("Computing confusion matrix...")
    confusion = defaultdict(int)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, meta, labels in tqdm(val_loader):
            imgs, meta, labels = imgs.to(DEVICE), meta.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs, meta)
            _, preds = torch.max(outputs, 1)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                if true != pred:
                    confusion[(val_ds.classes[true], val_ds.classes[pred])] += 1
                else:
                    correct += 1
                total += 1

    print(f"\nAccuracy: {correct/total*100:.2f}%")

    # Merge bidirectional
    merged = defaultdict(int)
    for (a,b), cnt in confusion.items():
        key = tuple(sorted([a,b]))
        merged[key] += cnt

    # Top 25
    top = sorted(merged.items(), key=lambda x: -x[1])[:25]
    print('\n' + '='*60)
    print('TOP 25 CONFUSED PAIRS (Bidirectional, 100 classes)')
    print('='*60)
    for i, ((a,b), cnt) in enumerate(top, 1):
        print(f'{i:2}. {a} <-> {b}: {cnt}')
