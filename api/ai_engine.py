import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

print("--- AI Engine Initializing ---")

# 1. SETUP: Force CPU (Matches your training setup)
device = torch.device("cpu")

def get_model(num_classes):
    """
    Rebuilds the exact same model structure used in training.
    """
    # Load the empty MobileNet shell (No weights needed, we load our own)
    model = models.mobilenet_v3_large(weights=None)
    
    # MATCHING ARCHITECTURE (Critical Step)
    # We must rebuild the exact same structure we defined in train_model.py
    # If we trained with Dropout, we must load with Dropout.
    in_features = model.classifier[3].in_features
    
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3), 
        nn.Linear(in_features, num_classes)
    )
    
    return model

# 2. LOAD THE SAVED WEIGHTS
MODEL_PATH = "wildlife_model.pth"
model = None
class_names = []

if os.path.exists(MODEL_PATH):
    print(f"Loading weights from {MODEL_PATH}...")
    try:
        # Load the dictionary we saved (weights + class names)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # 1. Get the list of animals (e.g., ['bear', 'wolf'...])
        class_names = checkpoint['class_names']
        
        # 2. Initialize the model with the correct number of outputs
        model = get_model(len(class_names))
        
        # 3. Fill the shell with your trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. Lock it down for testing (Eval mode)
        model.to(device)
        model.eval()
        print(f"✅ Model Loaded Successfully! Knows {len(class_names)} species.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: Did you change the training architecture without re-training?")
else:
    print(f"⚠️ Warning: {MODEL_PATH} not found.")
    print("The app will run, but AI predictions will fail until training is done.")

# 3. DEFINE THE TRANSFORM (The "Glasses")
# This must match the 'val' transform from your training script
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_animal(image_path):
    """
    The main function called by api/main.py
    """
    if model is None:
        return "AI Not Ready"
        
    try:
        # 1. Load and Preprocess Image
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # 2. Predict (Forward Pass)
        with torch.no_grad():
            outputs = model(img_tensor)
            # Convert raw numbers to probabilities (0% to 100%)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # 3. Interpret Results
        # Get the highest probability
        top_prob, top_idx = torch.max(probabilities, 0)
        
        # Look up the name using the index
        predicted_class = class_names[top_idx.item()]
        confidence = top_prob.item()
        
        # Clean up text: "american_black_bear" -> "American Black Bear"
        clean_name = predicted_class.replace('_', ' ').title()
        
        print(f"AI Prediction: {clean_name} ({confidence:.2%})")
        
        # Optional: Threshold check
        if confidence < 0.4:
            return "Unknown Animal"
            
        return clean_name

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error"