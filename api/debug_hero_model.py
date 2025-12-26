import torch
from torchvision import models
import torch.nn as nn

# Load checkpoint
ckpt = torch.load('wildlife_model_b3_hero.pth', map_location='cpu')

# Create base model
model = models.efficientnet_b3(weights=None)
print("Original classifier:")
print(model.classifier)

# Try to load and see what fails
try:
    model.load_state_dict(ckpt['model_state_dict'])
    print("\nLoaded successfully!")
except Exception as e:
    print(f"\nError: {e}")
    
# Check what keys exist
state = ckpt['model_state_dict']
classifier_keys = [k for k in state.keys() if 'classifier' in k]
print(f"\nClassifier keys in checkpoint: {classifier_keys}")

# Infer structure
print("\nInferred structure:")
print("classifier.0 = Dropout (default)")
print("classifier.1.0 = ? (missing)")
print("classifier.1.1.weight/bias = Linear layer")
print("classifier.1.2.weight/bias = Linear layer")
