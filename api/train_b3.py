import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os

# --- CONFIGURATION ---
DATA_DIR = 'training_data'
SAVE_PATH = 'wildlife_model_b3.pth'
EPOCHS = 20          # Increased to let AdamW work its magic
BATCH_SIZE = 16      # Fits in 6-8GB VRAM
IMG_SIZE = 300       # EfficientNet B3 Native

# --- AUGMENTATION ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((320, 320)),
        # SMART CROP: Zooms in/out to force model to look at details (ears/tails) not just "blob size"
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # SHAPE BIAS: 20% of images are Black & White. Forces it to learn "Shape" not just "Brown".
        transforms.RandomGrayscale(p=0.2), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 TRAINING SUPERCHARGED B3 ON: {device}")

    print(f"Loading Data...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    }
    
    class_names = image_datasets['train'].classes
    
    print("Initializing EfficientNet B3...")
    model = models.efficientnet_b3(weights='DEFAULT')
    
    # UNFREEZE MORE LAYERS (The "Unlock" Strategy)
    # Instead of just the last 3, we unfreeze the last 5 blocks to let it learn fur patterns better.
    for param in model.parameters():
        param.requires_grad = False
    
    # EfficientNet B3 has ~8 blocks. We unfreeze the last half.
    for param in model.features[-5:].parameters():
        param.requires_grad = True
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5), # Higher dropout = harder to memorize
        nn.Linear(num_ftrs, len(class_names))
    )
    
    model = model.to(device)
    
    # TRICK 1: LABEL SMOOTHING
    # This tells the AI: "If it's a Wolf, 0.9 confidence is good enough."
    # It stops the model from becoming overconfident and wrong.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # TRICK 2: ADAMW OPTIMIZER
    # Better at regularization than standard Adam.
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=0.0003, # Slightly higher starting LR
                            weight_decay=0.01)
                           
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    print("Starting Optimized Training...")
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)

    print(f"Saving Supercharged Model to {SAVE_PATH}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)

if __name__ == "__main__":
    train()