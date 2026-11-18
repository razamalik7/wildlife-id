import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import numpy as np

# CONFIG
DATA_DIR = 'training_data_cropped'
SAVE_PATH = 'wildlife_model_b3_hero.pth' # The Upgraded B3
EPOCHS = 20 # B3 converges faster
BATCH_SIZE = 16 # B3 is VRAM hungry, 16 is safer even for 5070, but 5070 might handle 32. Let's stick to 24 to be safe.
BATCH_SIZE = 24 
MIXUP_ALPHA = 0.2

# B3 prefers 300x300
IMG_SIZE = 300

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🦸 TRAINING B3 HERO (EfficientNet B3 + MixUp) ON: {device}")

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) 
                   for x in ['train', 'val']}
    
    print("Initializing EfficientNet B3...")
    model = models.efficientnet_b3(weights='DEFAULT')
    
    # Replace Classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    
    model = model.to(device)
    
    # AdamW usually works well
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_acc = 0.0

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
                    if phase == 'train':
                        # MixUp
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, use_cuda=(device.type=='cuda'))
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        
                        loss.backward()
                        optimizer.step()
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        running_corrects += torch.sum(preds == labels.data)

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(image_datasets[phase])
            
            if phase == 'val':
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print("⭐ New Best Model!")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'class_names': class_names
                    }, SAVE_PATH)
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} (MixUp active)')

    print(f"Training Complete. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()
