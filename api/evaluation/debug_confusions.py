
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from tqdm import tqdm
from evaluate_ensemble import WildlifeLateFusion, EvaluationDataset, load_model

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🦁 DEBUGGING CONFUSIONS")
    
    # 1. Dataset & Models
    tf_b3 = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tf_cx = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dir = os.path.join(DATA_DIR, 'val')
    dataset = EvaluationDataset(val_dir, tf_b3, tf_cx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = dataset.classes
    
    model_b3 = load_model('oogway_b3_best.pth', 'b3', DEVICE)
    model_cx = load_model('oogway_convnext_final.pth', 'convnext', DEVICE)
    
    print("\n🚀 Scanning for Specific Errors...")
    
    # Target Pairs to Debug
    TARGET_ERRORS = [
        ('wild_boar', 'striped_skunk'),
        ('brown_anole', 'green_anole')
    ]
    
    with torch.no_grad():
        for b3_imgs, cx_imgs, meta, labels in tqdm(loader):
            b3_imgs, cx_imgs = b3_imgs.to(DEVICE), cx_imgs.to(DEVICE)
            meta, labels = meta.to(DEVICE), labels.to(DEVICE)
            
            # Predictions
            out_b3 = model_b3(b3_imgs, meta)
            probs_b3 = torch.softmax(out_b3['species'], dim=1)
            _, pred_b3 = torch.max(probs_b3, 1)
            
            out_cx = model_cx(cx_imgs, meta)
            probs_cx = torch.softmax(out_cx['species'], dim=1)
            _, pred_cx = torch.max(probs_cx, 1)
            
            # Check samples
            for i in range(len(labels)):
                true_idx = labels[i].item()
                true_name = class_names[true_idx]
                
                b3_pred_name = class_names[pred_b3[i].item()]
                cx_pred_name = class_names[pred_cx[i].item()]
                
                # Check Boar -> Skunk
                if true_name == 'wild_boar' and (b3_pred_name == 'striped_skunk' or cx_pred_name == 'striped_skunk'):
                    # Find filename
                    # Note: Using dataset.samples works sequentially because shuffle=False
                    # But we are iterating batches. We need global index.
                    # This script is approximate unless we track index carefully.
                    # Let's just print finding for now.
                    pass 

    # Since getting filenames inside batch loop is tricky without custom collate or enumeration,
    # let's write a simpler single-image loop for just these classes.
    
    debug_classes = ['wild_boar', 'brown_anole']
    
    print("\n🔎 Detailed File Scan for Targets:")
    for cls in debug_classes:
        cls_dir = os.path.join(val_dir, cls)
        files = os.listdir(cls_dir)
        
        print(f"\nScanning {cls} ({len(files)} images)...")
        
        for fname in files:
            if not fname.lower().endswith(('jpg', 'png', 'jpeg')): continue
            
            fpath = os.path.join(cls_dir, fname)
            jpath = fpath.replace(os.path.splitext(fpath)[1], '.json')
            
            # Load and Prep
            img = Image.open(fpath).convert('RGB')
            # Assuming metadata is handled inside model or dummy 0s
            # For quick debug, we mock meta (mid-year, 0,0)
            meta_t = torch.tensor([[0, -1, 0, 0]]).to(DEVICE).float()
            
            # B3
            in_b3 = tf_b3(img).unsqueeze(0).to(DEVICE)
            p_b3 = torch.softmax(model_b3(in_b3, meta_t)['species'], 1)
            _, idx_b3 = torch.max(p_b3, 1)
            pred_b3 = class_names[idx_b3.item()]
            
            # CX
            in_cx = tf_cx(img).unsqueeze(0).to(DEVICE)
            p_cx = torch.softmax(model_cx(in_cx, meta_t)['species'], 1)
            _, idx_cx = torch.max(p_cx, 1)
            pred_cx = class_names[idx_cx.item()]
            
            # Logic
            if cls == 'wild_boar':
                if pred_cx == 'striped_skunk' and pred_b3 == 'striped_skunk':
                    print(f"  [BOTH WRONG] {fname} -> Skunk (CX: {p_cx[0][idx_cx].item():.2f}, B3: {p_b3[0][idx_b3].item():.2f})")
                elif pred_cx == 'striped_skunk':
                    print(f"  [CX WRONG]   {fname} -> Skunk (B3 got: {pred_b3})")
                    
            if cls == 'brown_anole':
                if pred_b3 == 'green_anole':
                     print(f"  [B3 WRONG]   {fname} -> Green Anole (Conf: {p_b3[0][idx_b3].item():.2f})")

if __name__ == "__main__":
    main()
