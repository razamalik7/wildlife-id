
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os, json, math
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Reuse the model/dataset classes from evaluating script to ensure consistency
from evaluate_ensemble import WildlifeLateFusion, EvaluationDataset, load_model

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'training_data_v2')
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_species_to_family():
    # Load taxonomy mapping
    try:
        with open('taxonomy_hierarchy.json', 'r') as f:
            taxonomy = json.load(f)
        
        species_to_family = {}
        for family, species_list in taxonomy['family_to_species'].items():
            for species in species_list:
                species_to_family[species] = family
        return species_to_family
    except:
        return {}

def main():
    print(f"🦁 ANALYZING MODEL STRENGTHS BY FAMILY")
    
    # 1. Load Taxonomy
    species_to_family = get_species_to_family()
    
    # 2. Dataset & Models
    # B3: 300x300
    tf_b3 = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ConvNeXt: 224x224
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
    
    # 3. Track Results
    # Store: {family: {'total': 0, 'b3_corr': 0, 'cx_corr': 0}}
    stats = defaultdict(lambda: {'total': 0, 'b3_corr': 0, 'cx_corr': 0})
    
    print("\n🚀 Running Analysis...")
    with torch.no_grad():
        for b3_imgs, cx_imgs, meta, labels in tqdm(loader):
            b3_imgs, cx_imgs = b3_imgs.to(DEVICE), cx_imgs.to(DEVICE)
            meta, labels = meta.to(DEVICE), labels.to(DEVICE)
            
            # B3 Preds
            out_b3 = model_b3(b3_imgs, meta)
            _, pred_b3 = torch.max(out_b3['species'], 1)
            
            # CX Preds
            out_cx = model_cx(cx_imgs, meta)
            _, pred_cx = torch.max(out_cx['species'], 1)
            
            # Tally per sample
            for i in range(len(labels)):
                label_idx = labels[i].item()
                species_name = class_names[label_idx]
                family = species_to_family.get(species_name, "Unknown")
                
                stats[family]['total'] += 1
                if pred_b3[i] == labels[i]:
                    stats[family]['b3_corr'] += 1
                if pred_cx[i] == labels[i]:
                    stats[family]['cx_corr'] += 1

    # 4. Print Report
    print(f"\n{'='*75}")
    print(f"{'FAMILY':<25} | {'COUNT':<5} | {'B3 ACC':<8} | {'CX ACC':<8} | {'WINNER':<10}")
    print(f"{'-'*75}")
    
    b3_wins = 0
    cx_wins = 0
    ties = 0
    
    sorted_families = sorted(stats.keys())
    
    for family in sorted_families:
        data = stats[family]
        total = data['total']
        if total == 0: continue
        
        b3_acc = data['b3_corr'] / total * 100
        cx_acc = data['cx_corr'] / total * 100
        
        if b3_acc > cx_acc:
            winner = "B3 🏆"
            b3_wins += 1
        elif cx_acc > b3_acc:
            winner = "CX 🚀"
            cx_wins += 1
        else:
            winner = "Tie 🤝"
            ties += 1
            
        print(f"{family:<25} | {total:<5} | {b3_acc:<6.1f}% | {cx_acc:<6.1f}% | {winner:<10}")
        
    print(f"{'='*75}")
    print(f"Summary: ConvNeXt won {cx_wins} families, B3 won {b3_wins} families, {ties} ties.")

if __name__ == "__main__":
    main()
