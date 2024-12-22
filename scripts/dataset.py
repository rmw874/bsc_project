import torch
import os
import cv2
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from utils import color2class

def add_rectangle(img):
    width, height = img.shape[1], img.shape[0]
    start_point_right = (math.ceil(0.5*width), 0)
    end_point_right = (width, height)
    color = (255,255,255)
    img = cv2.rectangle(img, start_point_right, end_point_right, color, -1)
        
    start_point_left = (0, 0)
    end_point_left = (math.floor(width*0.08), height)
    img = cv2.rectangle(img, start_point_left, end_point_left, color, -1)

    return img

class PirateLogDataset(Dataset):
    def __init__(self, mask_dir, img_dir, target_size=(3200,2496), num_classes=6, transform=None):
        # Initialize the dataset
        # Load data, apply initial transformations, etc.
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.num_classes = num_classes
        self.transform = transform

        # pre them pre them
        self.data = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.mask_dir))

        self.validate_paths()

    def validate_paths(self):
        """Validate all paths exist before training starts"""
        for img_file, mask_file in zip(self.data, self.labels):
            img_path = os.path.join(self.img_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            assert os.path.exists(img_path), f"Image not found: {img_path}"
            assert os.path.exists(mask_path), f"Mask not found: {mask_path}"
            assert len(self.data) == len(self.labels), "Number of images and masks must be the same"

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # # Retrieve the sample at index 'idx'
        # sample_path = os.path.join(self.img_dir, self.data[idx])
        # label_path = os.path.join(self.mask_dir, self.labels[idx])

        # img = cv2.imread(sample_path)
        # mask = cv2.imread(label_path)

        # # add rect
        # img = add_rectangle(img)
        
        # # Threshold the image and erode to remove noise
        # _, img = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((3, 3), np.uint8)
        # img = cv2.erode(img, kernel, iterations=2)
        # img = img / 255.0
        
        # img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        # img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        
        
        # class_mask = color2class(mask)
        # mask_resized = cv2.resize(class_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        # mask_tensor = torch.tensor(mask_resized, dtype=torch.long)
        
        # return img_tensor, mask_tensor
    

    ## almost working implementation of transform

        sample_path = os.path.join(self.img_dir, self.data[idx])
        label_path = os.path.join(self.mask_dir, self.labels[idx])

        # Read the image and mask
        img = cv2.imread(sample_path)
        mask = cv2.imread(label_path)

        # Add rectangles to img
        img = add_rectangle(img)

        # Threshold and erode image
        _, img = cv2.threshold(img, 8, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)

        # Resize both image and mask before augmentation
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Apply Albumentations transform if specified
        if self.transform:
            transformed = self.transform(image=img_resized, mask=mask_resized)
            img_transformed = transformed['image']
            mask_transformed = transformed['mask']
        else:
            img_transformed = img_resized
            mask_transformed = mask_resized

        # Convert the transformed image to [0,1] range
        img_transformed = img_transformed / 255.0

        # Convert color mask to class indices after augmentation
        mask_transformed = color2class(mask_transformed)

        # Convert to tensors
        img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_transformed, dtype=torch.long)

        return img_tensor, mask_tensor


