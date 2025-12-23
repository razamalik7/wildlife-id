import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse

# Reuse logic from evaluate_matrix but simplified for text output
def get_model(model_path, num_classes, device):
    checkpoint = torch.load(model_path, map_location=device)
    if 'resnet' in model_path.lower():
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    elif 'b3' in model_path.lower():
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    else:
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, num_classes))
        
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_dir = 'training_data'
    img_size = 224
    
    # Validation Data
    data_transforms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    class_names = val_dataset.classes
    
    # Load Model
    # Hardcoding ResNet for this specific check as requested
    model_path = 'wildlife_model_resnet.pth' 
    print(f"Loading {model_path}...")
    model = get_model(model_path, len(class_names), device)
    
    all_preds = []
    all_labels = []
    
    print("Inference...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\n" + "="*40)
    print("TOP 25 CONFUSED PAIRS")
    print("="*40)
    cm = confusion_matrix(all_labels, all_preds)
    np.fill_diagonal(cm, 0)
    indices = np.argsort(cm.flatten())[::-1]
    
    for i in range(25):
        index = indices[i]
        true_idx = index // len(class_names)
        pred_idx = index % len(class_names)
        count = cm[true_idx, pred_idx]
        if count > 0:
            print(f"{class_names[true_idx]} -> {class_names[pred_idx]}: {count}")

if __name__ == "__main__":
    main()
