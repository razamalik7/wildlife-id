import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse

# Config
DATA_DIR = 'training_data'
IMG_SIZE = 224 # Default for ResNet/MobileNet, adjust for B3 if needed (300)

def load_data(data_dir, img_size):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"Error: {val_dir} not found.")
        return None, None

    dataset = datasets.ImageFolder(val_dir, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    return dataloader, dataset.classes

def load_model(model_path, num_classes, device):
    # Try to determine model type from filename or arguments
    # For now, we'll try to load it as a ResNet50 first, then EfficientNet if that fails/is specified
    # Ideally, we should save architecture info in the .pth file.
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Heuristic to guess architecture or use a generic loader
    # This part is tricky without metadata. Let's assume ResNet50 for now based on 'wildlife_model_resnet.pth'
    if 'resnet' in model_path.lower():
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif 'b3' in model_path.lower():
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        # Default fallback to EfficientNet B0 (train_model.py default)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

def generate_confusion_matrix(model, dataloader, class_names, device):
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {accuracy*100:.2f}%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    
    # Print most confused classes
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    indices = np.argsort(cm_no_diag.flatten())[::-1]
    
    print("\nTop 20 Confused Pairs (True -> Predicted):")
    for i in range(20):
        index = indices[i]
        true_idx = index // len(class_names)
        pred_idx = index % len(class_names)
        count = cm_no_diag[true_idx, pred_idx]
        if count > 0:
            print(f"{class_names[true_idx]} -> {class_names[pred_idx]}: {count} errors")
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def main():
    parser = argparse.ArgumentParser(description='Evaluate Wildlife Model')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader, class_names = load_data(DATA_DIR, IMG_SIZE)
    if dataloader is None:
        return

    model = load_model(args.model, len(class_names), device)
    generate_confusion_matrix(model, dataloader, class_names, device)

if __name__ == "__main__":
    main()
