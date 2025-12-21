import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# 1. SETUP
DATA_DIR = 'training_data'
SAVE_PATH = 'wildlife_model.pth'
EPOCHS = 15  # Increased to 15 since we have more data/dropout
BATCH_SIZE = 32 # Increased batch size for speed (reduce if memory error)

# 2. LEVEL 1 UPGRADE: STRONGER AUGMENTATION
# We force the AI to look at random cropped parts (head, paw, tail)
# instead of just the whole image. This prevents memorization.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
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
    print("Loading images...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) 
                   for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    print(f"Classes detected: {len(class_names)}")
    
    print("Downloading MobileNet...")
    model = models.mobilenet_v3_large(weights='DEFAULT')
    
    # Freeze the base
    for param in model.parameters():
        param.requires_grad = False
        
    # LEVEL 2 UPGRADE: DROPOUT ARCHITECTURE
    # We add a Dropout layer to randomly 'kill' neurons during training.
    # This forces the brain to be robust and not rely on single features.
    num_ftrs = model.classifier[3].in_features
    
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3), # Drop 30% of connections
        nn.Linear(num_ftrs, len(class_names))
    )
    
    # Force CPU (Since GPU is having issues)
    device = torch.device("cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    
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

    print(f"Training Complete. Saving to {SAVE_PATH}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)

if __name__ == "__main__":
    train()