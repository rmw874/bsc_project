# scripts/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LogbookDataset
from model import UNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

def main():
    # Paths
    train_images_dir = 'data/processed/train/images/'
    train_masks_dir = 'data/processed/train/masks/'
    val_images_dir = 'data/processed/val/images/'
    val_masks_dir = 'data/processed/val/masks/'

    # Hyperparameters
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets and DataLoaders
    train_dataset = LogbookDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=get_transforms()
    )

    val_dataset = LogbookDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        transform=get_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Model, Loss, Optimizer
    model = UNet(n_channels=3).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for images, masks in dataloaders[phase]:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # Adjust learning rate
            if phase == 'val':
                scheduler.step(epoch_loss)

                # Save the model if validation loss has decreased
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), f'models/unet_best.pth')
                    print('Model saved.')

    print('Training complete.')

if __name__ == '__main__':
    main()