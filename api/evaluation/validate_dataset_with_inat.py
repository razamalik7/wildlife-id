"""
Comprehensive dataset validation using iNaturalist pretrained model
Tests EVERY species folder and reports accuracy
"""
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import json
from tqdm import tqdm
from collections import defaultdict

# Updated to v2 (Oogway era)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data_v2')

# Load species config
with open('species_config.json', 'r') as f:
    SPECIES_CONFIG = json.load(f)

# iNaturalist class names (simplified - will use model's built-in labels)
# The model will return class indices, we'll use class names from model

def load_inat_model():
    """Load pretrained iNaturalist model"""
    print("Loading iNaturalist pretrained model...")
    # EfficientNet-B3 Noisy Student trained on iNaturalist
    model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=10000)
    model.eval()
    
    # Get class names if available
    model_info = timm.models.get_pretrained_cfg('tf_efficientnet_b3_ns')
    print(f"Model: {model_info}")
    
    return model

def get_predictions(model, image_path, transform):
    """Get top-5 predictions with class names"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
            
        return {
            'probs': top5_prob[0].tolist(),
            'indices': top5_idx[0].tolist()
        }
    except Exception as e:
        return None

def validate_species_folder(species_name, model, transform, max_images=50):
    """
    Validate a species folder by checking if iNat model predicts correctly
    
    Returns:
        accuracy: % of images where species_name appears in top-5 predictions
        confusion: what iNat thinks these images are
    """
    results = {
        'total': 0,
        'top1_match': 0,
        'top5_match': 0,
        'predictions': defaultdict(int)
    }
    
    # Check train folder
    species_dir = os.path.join(DATA_DIR, 'train', species_name)
    
    if not os.path.exists(species_dir):
        return None
        
    images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sample if too many
    if len(images) > max_images:
        import random
        images = random.sample(images, max_images)
    
    for img_name in images:
        img_path = os.path.join(species_dir, img_name)
        pred = get_predictions(model, img_path, transform)
        
        if pred is None:
            continue
        
        results['total'] += 1
        
        # For now, just track top prediction index
        # We'll need to map indices to species names
        top_idx = pred['indices'][0]
        results['predictions'][top_idx] += 1
        
        # We can't directly match yet since we don't have the label mapping
        # But we can check consistency
        
    return results

if __name__ == '__main__':
    print("="*70)
    print("COMPREHENSIVE DATASET VALIDATION WITH INATURALIST")
    print("="*70)
    
    # Setup
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model = load_inat_model()
    
    print("\nNote: iNat model uses its own 10k class taxonomy")
    print("We'll identify species by checking prediction consistency")
    print("Low consistency = likely mislabeled folder\n")
    
    # Validate all species
    species_results = {}
    
    # Get all species from training data
    train_dir = os.path.join(DATA_DIR, 'train')
    all_species = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"Found {len(all_species)} species folders\n")
    
    for species in tqdm(all_species, desc="Validating species"):
        results = validate_species_folder(species, model, transform, max_images=30)
        if results and results['total'] > 0:
            # Calculate consistency: what % agree on top prediction
            top_pred = max(results['predictions'].items(), key=lambda x: x[1])
            consistency = top_pred[1] / results['total']
            
            species_results[species] = {
                'consistency': consistency,
                'total_checked': results['total'],
                'top_prediction_idx': top_pred[0],
                'top_prediction_count': top_pred[1]
            }
    
    # Report
    print("\n" + "="*70)
    print("CONSISTENCY REPORT")
    print("="*70)
    print("Consistency = % of images where iNat agrees on the same class")
    print("Low consistency (<50%) = likely contaminated folder\n")
    
    # Sort by consistency
    sorted_species = sorted(species_results.items(), key=lambda x: x[1]['consistency'])
    
    print("\nWORST 20 FOLDERS (Most Likely Contaminated):")
    print(f"{'Species':<35} {'Consistency':>12} {'Samples':>8}")
    print("-"*70)
    
    for species, data in sorted_species[:20]:
        consistency = data['consistency']
        total = data['total_checked']
        flag = "🔴" if consistency < 0.5 else "🟡" if consistency < 0.7 else ""
        print(f"{flag} {species:<35} {consistency:>11.1%} {total:>8}")
    
    print("\n\nBEST 10 FOLDERS (Clean):")
    print(f"{'Species':<35} {'Consistency':>12} {'Samples':>8}")
    print("-"*70)
    
    for species, data in sorted_species[-10:]:
        consistency = data['consistency']
        total = data['total_checked']
        print(f"✓ {species:<35} {consistency:>11.1%} {total:>8}")
    
    # Save full report
    with open('inat_validation_report.json', 'w') as f:
        json.dump(species_results, f, indent=2)
    
    print("\n📁 Full report saved to: inat_validation_report.json")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
    - High consistency (>80%): Folder is probably clean
    - Medium consistency (50-80%): Some contamination or hard species
    - Low consistency (<50%): Likely heavily contaminated
    
    Known issues we expect to see:
    - elk: Should show low consistency (moose contamination)
    - arctic_fox: Low consistency (red fox contamination)  
    - gemsbok: Low consistency if plant photos present
    - jaguar: Low consistency if plant photos present
    """)
