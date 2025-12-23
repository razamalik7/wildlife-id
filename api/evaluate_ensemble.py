import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm
from PIL import Image

# CONFIG
DATA_DIR = 'training_data'
BATCH_SIZE = 16 

# --- A. CUSTOM DATASET FOR DUAL INPUTS (B3 + ConvNeXt) ---
class DualModelDataset(Dataset):
    def __init__(self, root_dir, transform_b3, transform_hero):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform_b3 = transform_b3
        self.transform_hero = transform_hero
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = Image.open(path).convert('RGB')
        
        # 1. Standard Center Crop (No TTA base)
        img_b3 = self.transform_b3(img)
        img_hero = self.transform_hero(img)
        
        # 2. TTA: We return the raw image too if we want to do TTA inside the loop, 
        # but for simplicity/speed in PyTorch, standard TTA usually happens via 
        # FiveCrop/TenCrop inside the transform or handled in the loop. 
        # Let's do a "Poor Man's TTA" (Horizontal Flip) manually in the loop 
        # to keep memory usage low, or use TenCrop if batch size allows.
        
        return img_b3, img_hero, label

# --- B. MODEL LOADERS ---
def get_b3(num_classes, device):
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    
    ckpt = torch.load('wildlife_model_b3_hero.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def get_hero(num_classes, device):
    model = models.convnext_tiny(weights=None)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
    
    ckpt = torch.load('wildlife_model_hero.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🦁 ASSEMBLING THE COUNCIL (Ensemble Evaluation) ON: {device}")
    
    # 1. Verify Models Exist
    if not os.path.exists('wildlife_model_b3_hero.pth') or not os.path.exists('wildlife_model_hero.pth'):
        print("❌ Models not ready yet. Please wait for training to finish.")
        return

    # 2. Setup Transforms (Standard Evaluation)
    # Note: TTA will be applied by running the model twice: once on x, once on flip(x)
    trans_b3 = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    trans_hero = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dir = os.path.join(DATA_DIR, 'val')
    test_ds = DualModelDataset(val_dir, trans_b3, trans_hero)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    class_names = test_ds.classes
    print(f"📂 Loaded {len(test_ds)} validation images.")

    # 3. Load Models
    print("🧠 Loading Models...")
    model_b3 = get_b3(len(class_names), device)
    model_hero = get_hero(len(class_names), device)
    print("✅ Models Loaded.")

    # 4. Evaluation Loop with TTA (2-View: Normal + Flip)
    correct_b3 = 0
    correct_hero = 0
    correct_ensemble = 0
    total = 0
    
    # TTA Flip Transform
    # We will just flip the tensor batch manually
    
    print("🚀 Starting Inference with TTA (2-View)...")
    
    with torch.no_grad():
        for b3_imgs, hero_imgs, labels in tqdm(test_loader):
            b3_imgs = b3_imgs.to(device)
            hero_imgs = hero_imgs.to(device)
            labels = labels.to(device)
            
            # --- VIEW 1: STANDARD ---
            out_b3_1 = torch.nn.functional.softmax(model_b3(b3_imgs), dim=1)
            out_hero_1 = torch.nn.functional.softmax(model_hero(hero_imgs), dim=1)
            
            # --- VIEW 2: FLIPPED (Horizontal Flip) ---
            # torch.flip(tensor, dims) -> dims=3 is width
            out_b3_2 = torch.nn.functional.softmax(model_b3(torch.flip(b3_imgs, [3])), dim=1)
            out_hero_2 = torch.nn.functional.softmax(model_hero(torch.flip(hero_imgs, [3])), dim=1)
            
            # --- AVERAGE VIEWS (TTA) ---
            probs_b3 = (out_b3_1 + out_b3_2) / 2
            probs_hero = (out_hero_1 + out_hero_2) / 2
            
            # --- ENSEMBLE ---
            # You can weight them 0.6/0.4 if one is stronger, but 0.5/0.5 is safe
            probs_ensemble = (probs_b3 + probs_hero) / 2
            
            # --- PREDICTIONS ---
            _, preds_b3 = torch.max(probs_b3, 1)
            _, preds_hero = torch.max(probs_hero, 1)
            _, preds_ensemble = torch.max(probs_ensemble, 1)
            
            total += labels.size(0)
            correct_b3 += (preds_b3 == labels).sum().item()
            correct_hero += (preds_hero == labels).sum().item()
            correct_ensemble += (preds_ensemble == labels).sum().item()

    print("\n🏆 FINAL RESULTS (Validation Set) 🏆")
    print(f"EfficientNet B3 (TTA):  {100 * correct_b3 / total:.2f}%")
    print(f"ConvNeXt Hero (TTA):    {100 * correct_hero / total:.2f}%")
    print(f"🌟 ENSEMBLE (Combined): {100 * correct_ensemble / total:.2f}%")

if __name__ == "__main__":
    main()
