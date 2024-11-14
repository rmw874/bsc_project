# scripts/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LogbookDataset(Dataset):
    """Custom Dataset for loading logbook images and boundary masks"""

    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Directory with input images.
            masks_dir (str): Directory with boundary masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        assert len(self.images) == len(self.masks), "Number of images and masks should be equal"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # Normalize mask to [0, 1]

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert mask to tensor and add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask