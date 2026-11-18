import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import time

# CONFIG
DATA_DIR = 'training_data'
SAVE_PATH = 'wildlife_model.pth'
EPOCHS = 12  # A few more epochs to let the Scheduler work
BATCH_SIZE = 32

# STRONGER AUGMENTATION (Prevents memorization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        # RandomResizedCrop is harder than RandomCrop. It zooms in/out randomly.
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        # ColorJitter prevents it from memorizing "Bears are always dark brown"
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train():
    print("Loading Data...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    
    # num_workers=4 uses 4 CPU cores to prep data ahead of time
# pin_memory=True speeds up the transfer from RAM to VRAM
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True, 
                                              num_workers=4, 
                                              pin_memory=True) 
               for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    print(f"Classes: {len(class_names)}")
    
    print("Initializing EfficientNet B0...")
    model = models.efficientnet_b0(weights='DEFAULT')
    
    # 1. Freeze Base
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. DEEPER UNFREEZING (Unlock last 3 blocks for better texture learning)
    # EfficientNet features are in blocks. We unlock the last few.
    for param in model.features[-3:].parameters():
        param.requires_grad = True
        
    # 3. REPLACE HEAD
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4), # Increased Dropout to 40%
        nn.Linear(num_ftrs, len(class_names))
    )
    
    device = torch.device("cuda")
    model = model.to(device)
    
    print(f"🚀 TRAINING ON DEVICE: {device}") # <--- Add this line
    print(f"GPU Name: {torch.cuda.get_device_name(0)}") # <--- And this one

    criterion = nn.CrossEntropyLoss()
    
    # OPTIMIZER UPGRADE: Added 'weight_decay' (L2 Regularization)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=0.0001, 
                           weight_decay=1e-4)
                           
    # SCHEDULER: If validation loss stops improving for 2 epochs, cut LR by 10x
    # NEW (Fixed)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    print("Starting Professional Training Run...")
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

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

            # Step the scheduler on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)

    print(f"Saving Smart Model to {SAVE_PATH}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)

if __name__ == "__main__":
    train()