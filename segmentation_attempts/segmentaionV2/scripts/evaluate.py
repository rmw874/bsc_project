# scripts/evaluate.py

import os
import torch
from torch.utils.data import DataLoader

from dataset import LogbookDataset
from model import UNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
from sklearn.metrics import f1_score

def get_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

def main():
    # Paths
    val_images_dir = 'data/processed/val/images/'
    val_masks_dir = 'data/processed/val/masks/'
    model_path = 'models/unet_best.pth'
    output_dir = 'outputs/'

    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    val_dataset = LogbookDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        transform=get_transforms()
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load the model
    model = UNet(n_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluation
    f1_scores = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            preds_np = preds.cpu().numpy().flatten()
            masks_np = masks.cpu().numpy().flatten()

            # Compute F1 score
            f1 = f1_score(masks_np, preds_np, zero_division=1)
            f1_scores.append(f1)

            # Save predicted mask
            pred_mask = preds.cpu().numpy()[0, 0]
            pred_mask = (pred_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'prediction_{idx}.png'), pred_mask)

    mean_f1 = np.mean(f1_scores)
    print(f'Mean F1 Score on validation set: {mean_f1:.4f}')

if __name__ == '__main__':
    main()