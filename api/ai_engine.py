
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import math
from datetime import datetime

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"--- 🦁 AI Engine: OOGWAY SMART ENSEMBLE Initializing on {DEVICE} ---")

# --- 1. DEFINE ARCHITECTURE (Must match training exactly) ---
class WildlifeLateFusion(nn.Module):
    """Late Fusion: Image pathway + Metadata pathway with hierarchical outputs."""
    
    def __init__(self, num_species, num_families, num_classes, model_type='b3'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'b3':
            self.image_model = models.efficientnet_b3(weights=None)
            in_features = self.image_model.classifier[1].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        elif model_type == 'convnext':
            self.image_model = models.convnext_tiny(weights=None)
            in_features = self.image_model.classifier[2].in_features
            self.image_model.classifier = self.image_model.classifier[:-1]
            feature_dim = in_features
        
        # Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        
        # Hierarchical prediction heads
        self.species_head = nn.Linear(feature_dim, num_species)
        self.family_head = nn.Linear(feature_dim, num_families)
        self.class_head = nn.Linear(feature_dim, num_classes)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, meta):
        image_features = self.image_model(x)
        meta_features = self.meta_mlp(meta)
        combined_features = image_features + self.fusion_weight * meta_features
        
        return {
            'species': self.species_head(combined_features),
            'family': self.family_head(combined_features),
            'class': self.class_head(combined_features)
        }

# --- 2. LOAD TAXONOMY & MODELS ---

# Load Species Config
SPECIES_CONFIG = {}
SPECIES_LIST_RAW = [] # Keep ordered list for indexing
try:
    config_path = os.path.join(BASE_DIR, 'species_config.json')
    with open(config_path, 'r') as f:
        SPECIES_LIST_RAW = json.load(f)
        for species in SPECIES_LIST_RAW:
            key = species.get('name', '').lower().replace(' ', '_').replace('-', '_')
            SPECIES_CONFIG[key] = species
    print(f"   Loaded taxonomy for {len(SPECIES_CONFIG)} species")
except Exception as e:
    print(f"   Warning: Could not load species_config.json: {e}")

def load_checkpoint(path, model_type):
    if not os.path.exists(path):
        print(f"❌ Model file not found: {path}")
        return None, None
        
    try:
        ckpt = torch.load(path, map_location=DEVICE)
        
        # Extract dimensions
        class_names = ckpt['class_names']
        families = ckpt['families']
        tax_classes = ckpt['tax_classes']
        
        # Init Model
        model = WildlifeLateFusion(len(class_names), len(families), len(tax_classes), model_type)
        
        # Handle state dict (remove 'module.' prefix)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model.to(DEVICE)
        model.eval()
        
        print(f"✅ Loaded {model_type.upper()} from {os.path.basename(path)}")
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading {model_type}: {e}")
        return None, None

# Load Models
print("🧠 Loading Trained Models...")
path_b3 = os.path.join(BASE_DIR, 'oogway_b3_best.pth')
path_cx = os.path.join(BASE_DIR, 'oogway_convnext_final.pth')

model_b3, classes_b3 = load_checkpoint(path_b3, 'b3')
model_cx, classes_cx = load_checkpoint(path_cx, 'convnext')

# Helper: Family lookup for Smart Ensemble
# We need to map class_index -> family_name to check against B3_STRONG_FAMILIES
idx_to_family = {}
if classes_cx:
    for idx, name in enumerate(classes_cx):
        # Look up in config
        clean_name = name.lower().replace('-', '_')
        info = SPECIES_CONFIG.get(clean_name, {})
        family = info.get('taxonomy', {}).get('family', 'Unknown')
        idx_to_family[idx] = family

def idx_to_class_name(idx, classes):
    if idx < len(classes):
        return classes[idx]
    return "unknown"

# --- 3. TRANSFORMS ---
# Standard Eval Transforms (Not TTA, for speed)
# If user wants TTA again, we can re-add it, but 10-crop is slow for real-time.
# We will use CenterCrop which is standard for inference.

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

# --- 4. PREDICTION LOGIC ---

def predict_animal(image_path, lat=None, lng=None, date=None):
    try:
        # Load Image
        img = Image.open(image_path).convert('RGB')
        
        # Prepare Metadata
        if lat is None: lat = 0.0
        if lng is None: lng = 0.0
        if date is None:
            month = datetime.now().month
        else:
            # Parse date string if needed, or assume it's a month int
            try:
                if isinstance(date, str):
                    month = datetime.strptime(date, "%Y-%m-%d").month
                else:
                    month = int(date)
            except:
                month = 6 # Default mid-year
                
        # Create Meta Tensor [sin(month), cos(month), lat, lng]
        meta_tensor = torch.tensor([
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            lat / 90.0,
            lng / 180.0
        ], dtype=torch.float32).unsqueeze(0).to(DEVICE) # Add batch dim
        
        print("\n🔎 --- OOGWAY SESSION STARTED ---")
        
        # --- MODEL 1: B3 ---
        probs_b3 = None
        if model_b3:
            input_b3 = tf_b3(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model_b3(input_b3, meta_tensor)
                probs_b3 = torch.softmax(out['species'], dim=1)
                
        # --- MODEL 2: ConvNeXt ---
        probs_cx = None
        if model_cx:
            input_cx = tf_cx(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model_cx(input_cx, meta_tensor)
                probs_cx = torch.softmax(out['species'], dim=1)
        
        # --- SMART ENSEMBLE VOTING ---
        final_probs = None
        
        if probs_b3 is not None and probs_cx is not None:
            # 1. Get initial predictions
            score_b3, idx_b3 = torch.max(probs_b3, 1)
            score_cx, idx_cx = torch.max(probs_cx, 1)
            
            pred_b3_name = idx_to_class_name(idx_b3.item(), classes_b3)
            pred_cx_name = idx_to_class_name(idx_cx.item(), classes_cx)
            
            # 2. DEFINITIVE EXPERT OVERRIDES (Data-Driven from find_expert_overrides.py)
            # If ConvNeXt predicts X, but B3 predicts Y, trust B3
            # Format: (ConvNeXt_Pred, B3_Pred) -> Trust B3
            EXPERT_OVERRIDES = {
                ('striped_skunk', 'wild_boar'),        # Skunk/Boar Confusion
                ('sea_otter', 'harbor_seal'),          # Otter/Seal Confusion
                ('coyote', 'gray_wolf'),               # Canids
                ('american_alligator', 'american_crocodile'), # Crocs
                ('nile_monitor', 'argentine_black_and_white_tegu'), # Monitors
                ('white-tailed_deer', 'moose'),        # Deer/Moose
                ('california_sea_lion', 'northern_elephant_seal'), # Seals
                ('burmese_python', 'western_diamondback_rattlesnake') # Snake patterns
            }
            
            # 3. CLASS-LEVEL SUPERIORITY (Classes where B3 Acc > ConvNeXt Acc)
            B3_SUPERIOR_CLASSES = {
                'nile_monitor', 'yellow-bellied_marmot', 'harbor_seal', 
                'virginia_opossum', 'thinhorn_sheep', 'spotted_salamander',
                'western_diamondback_rattlesnake', 'mountain_lion', 
                'american_marten', 'common_box_turtle', 'great_horned_owl'
            }
            
            # LOGIC START
            if (pred_cx_name, pred_b3_name) in EXPERT_OVERRIDES:
                print(f"⚖️  EXPERT OVERRIDE: Trusting B3 ({pred_b3_name}) over ConvNeXt ({pred_cx_name}) due to known confusion.")
                final_probs = (0.7 * probs_b3) + (0.3 * probs_cx)
                
            elif pred_b3_name in B3_SUPERIOR_CLASSES:
                print(f"⚖️  Smart Ensemble: B3 is a specialist for {pred_b3_name}. Upweighting B3.")
                final_probs = (0.6 * probs_b3) + (0.4 * probs_cx)
                
            else:
                # Default: Trust ConvNeXt more
                final_probs = (0.4 * probs_b3) + (0.6 * probs_cx)
                
        elif probs_b3 is not None:
            final_probs = probs_b3
        elif probs_cx is not None:
            final_probs = probs_cx
            
        if final_probs is None:
            return {"error": "No models loaded"}

        # --- FORMAT RESULTS ---
        top_probs, top_idxs = torch.topk(final_probs, 3)
        
        candidates = []
        for i in range(3):
            score = top_probs[0][i].item() * 100
            idx = top_idxs[0][i].item()
            
            # Helper to safely get name
            if classes_cx and idx < len(classes_cx):
                raw_name = classes_cx[idx]
            elif classes_b3 and idx < len(classes_b3):
                raw_name = classes_b3[idx]
            else:
                continue
                
            display_name = raw_name.replace('_', ' ').title()
            
            # Look up taxonomy
            species_key = raw_name.lower().replace('-', '_')
            species_info = SPECIES_CONFIG.get(species_key, {})
            taxonomy = species_info.get('taxonomy', {})
            
            candidate = {
                "name": display_name,
                "score": score,
                "scientific_name": species_info.get('scientific_name', ''),
                "taxonomy": {
                    "class": taxonomy.get('class', ''),
                    "order": taxonomy.get('order', ''),
                    "family": taxonomy.get('family', '')
                }
            }
            candidates.append(candidate)
            
        print(f"🏆 Final Decision: {candidates[0]['name']} ({candidates[0]['score']:.1f}%)")
        return {"candidates": candidates}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Test run
    test_img = os.path.join(BASE_DIR, "confusion_matrix.png") # Just a dummy path check
    # predict_animal(test_img)