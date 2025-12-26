"""
Confusion Matrix Analyzer
=========================
Run AFTER Grandmaster training to identify worst confusing pairs.

Outputs:
1. Confusion matrix heatmap (PNG)
2. Top confused pairs report (console + JSON)
3. Recommendations for targeted scraping

Usage:
  python analyze_confusion.py --model grandmaster_b3_final.pth --type b3
  python analyze_confusion.py --model grandmaster_convnext_final.pth --type convnext
"""

import torch
import torch.nn as nn
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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Config
DATA_DIR = './training_data_cropped'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- DATASET ---
class WildlifeGeoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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


def analyze_confusion(model_path, model_type, top_n=20):
    """
    Generate comprehensive confusion analysis.
    """
    print(f"🔍 CONFUSION MATRIX ANALYZER")
    print(f"   Model: {model_path}")
    print(f"   Type: {model_type}")
    
    # Transforms
    val_transform = transforms.Compose([
        transforms.Resize(int(300 * 1.1)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_ds = WildlifeGeoDataset(os.path.join(DATA_DIR, 'val'), val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    num_classes = len(val_ds.classes)
    class_names = val_ds.classes
    print(f"📂 Loaded {len(val_ds)} validation images ({num_classes} classes)")
    
    # Load model
    model = WildlifeLateFusion(num_classes, model_type=model_type).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded")
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    print("📊 Running inference...")
    with torch.no_grad():
        for imgs, meta, labels in tqdm(val_loader):
            imgs, meta = imgs.to(DEVICE), meta.to(DEVICE)
            outputs = model(imgs, meta)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1
    
    # Calculate overall accuracy
    correct = np.sum(np.diag(confusion))
    total = np.sum(confusion)
    accuracy = correct / total
    print(f"\n📈 Overall Accuracy: {accuracy*100:.2f}%")
    
    # Find worst confused pairs (excluding diagonal)
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                confusion_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'errors': int(confusion[i, j]),
                    'true_class_total': int(np.sum(confusion[i, :])),
                    'error_rate': confusion[i, j] / np.sum(confusion[i, :]) * 100
                })
    
    # Sort by number of errors
    confusion_pairs.sort(key=lambda x: x['errors'], reverse=True)
    
    # Print top confused pairs
    print(f"\n🔴 TOP {top_n} CONFUSED PAIRS:")
    print("=" * 70)
    print(f"{'True Class':<25} {'→ Predicted':<25} {'Errors':<8} {'Rate':<8}")
    print("-" * 70)
    
    for i, pair in enumerate(confusion_pairs[:top_n]):
        true_name = pair['true_class'].replace('_', ' ')[:24]
        pred_name = pair['predicted_class'].replace('_', ' ')[:24]
        print(f"{true_name:<25} → {pred_name:<25} {pair['errors']:<8} {pair['error_rate']:.1f}%")
    
    # Find classes that need more data (appear frequently in confusion)
    class_confusion_score = defaultdict(int)
    for pair in confusion_pairs:
        class_confusion_score[pair['true_class']] += pair['errors']
        class_confusion_score[pair['predicted_class']] += pair['errors']
    
    # Sort classes by total confusion involvement
    confused_classes = sorted(class_confusion_score.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 CLASSES MOST INVOLVED IN CONFUSION:")
    print("=" * 50)
    print(f"{'Class':<30} {'Total Confusion Score':<20}")
    print("-" * 50)
    for class_name, score in confused_classes[:15]:
        print(f"{class_name.replace('_', ' '):<30} {score:<20}")
    
    # Save detailed report as JSON
    report = {
        'model_path': model_path,
        'model_type': model_type,
        'overall_accuracy': accuracy,
        'total_validation_images': int(total),
        'top_confused_pairs': confusion_pairs[:top_n],
        'classes_needing_more_data': [c[0] for c in confused_classes[:15]],
        'confusion_scores': dict(confused_classes)
    }
    
    report_path = model_path.replace('.pth', '_confusion_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to: {report_path}")
    
    # Generate confusion matrix heatmap
    print("\n📊 Generating confusion matrix heatmap...")
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix for better visualization
    confusion_norm = confusion.astype(float)
    row_sums = confusion_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_norm = confusion_norm / row_sums
    
    sns.heatmap(confusion_norm, 
                xticklabels=[c.replace('_', ' ')[:15] for c in class_names],
                yticklabels=[c.replace('_', ' ')[:15] for c in class_names],
                cmap='YlOrRd',
                vmin=0, vmax=0.3,  # Cap at 30% for better contrast
                cbar_kws={'label': 'Error Rate'})
    
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix - {model_type.upper()} (Acc: {accuracy*100:.1f}%)')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    heatmap_path = model_path.replace('.pth', '_confusion_matrix.png')
    plt.savefig(heatmap_path, dpi=150)
    print(f"📊 Heatmap saved to: {heatmap_path}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("📋 RECOMMENDATIONS FOR TARGETED IMPROVEMENT:")
    print("=" * 70)
    print(f"\n1. SCRAPE MORE DATA for these {min(10, len(confused_classes))} classes:")
    for class_name, _ in confused_classes[:10]:
        print(f"   - {class_name}")
    
    print(f"\n2. Run targeted fine-tuning:")
    print(f"   python targeted_finetune.py --model {model_path} --type {model_type}")
    
    print(f"\n3. Consider merging if visually identical:")
    # Find bidirectional confusion (A→B and B→A)
    bidirectional = []
    for pair in confusion_pairs[:30]:
        reverse = next((p for p in confusion_pairs if 
                        p['true_class'] == pair['predicted_class'] and 
                        p['predicted_class'] == pair['true_class']), None)
        if reverse and pair['errors'] + reverse['errors'] > 10:
            key = tuple(sorted([pair['true_class'], pair['predicted_class']]))
            if key not in [tuple(sorted([b[0], b[1]])) for b in bidirectional]:
                bidirectional.append((pair['true_class'], pair['predicted_class'], 
                                     pair['errors'] + reverse['errors']))
    
    for a, b, total in sorted(bidirectional, key=lambda x: x[2], reverse=True)[:5]:
        print(f"   - {a.replace('_', ' ')} ↔ {b.replace('_', ' ')} ({total} mutual errors)")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze Confusion Matrix')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--type', type=str, default='b3', choices=['b3', 'convnext'])
    parser.add_argument('--top', type=int, default=20, help='Number of top confused pairs to show')
    args = parser.parse_args()
    
    analyze_confusion(args.model, args.type, args.top)


if __name__ == "__main__":
    main()
