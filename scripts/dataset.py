import torch
import os
import cv2
import numpy as np
import math
from torch.utils.data import Dataset
from utils import color2class

def add_rectangle(img):
    width, height = img.shape[1], img.shape[0]
    color = (255,255,255)   
    start_point_left = (0, 0)
    end_point_left = (math.floor(width*0.08), height)
    img = cv2.rectangle(img, start_point_left, end_point_left, color, -1)

    return img

def find_vertical_line_bounds(img):
    """Find the leftmost and rightmost vertical lines in left half of image"""
    vertical = np.copy(img)
    
    # nothing relevant on right-half of image.
    width = img.shape[1] // 2
    vertical = vertical[:, : width]

    # get cols and pick significant out
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vertical_kernel)    
    
    col_profile = np.sum(vertical == 0, axis=0) 
    threshold = np.max(col_profile) * 0.15  # Adjust this threshold if needed
    line_cols = np.where(col_profile > threshold)[0]
    


    # first and last col in left half
    left_bound = max(0, line_cols[0] - 10)  
    right_bound = min(width, line_cols[-1] - 5)  

    return left_bound, right_bound

def preprocess(img, target_size, BS=13, C=12):
    """Complete preprocessing pipeline"""
    img = add_rectangle(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=BS,
        C=C
    )
    
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

    lb, rb = find_vertical_line_bounds(img)
    img[:, :lb] = 255
    img[:, rb:] = 255

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # format for albumentations
    img = np.stack([img, img, img], axis=-1)
    return img.astype(np.uint8)

class PirateLogDataset(Dataset):
    def __init__(self, mask_dir, img_dir, target_size=(3200,2496), num_classes=6, transform=None):
        # Initialize the dataset
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
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_path = os.path.join(self.img_dir, self.data[idx])
        label_path = os.path.join(self.mask_dir, self.labels[idx])
        
        # Read the image and mask
        img = cv2.imread(sample_path)
        mask = cv2.imread(label_path)
        
        # Apply new preprocessing pipeline
        img_processed = preprocess(img, self.target_size)
        
        # Resize mask
        mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply Albumentations transform if specified
        if self.transform:
            transformed = self.transform(image=img_processed, mask=mask_resized)
            img_transformed = transformed['image']
            mask_transformed = transformed['mask']
        else:
            img_transformed = img_processed
            mask_transformed = mask_resized
        
        # Convert color mask to class indices after augmentation
        mask_transformed = color2class(mask_transformed)
        
        # Convert to tensors and normalize image to [0,1] range after augmentations
        img_tensor = torch.from_numpy(img_transformed).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.tensor(mask_transformed, dtype=torch.long)
        
        return img_tensor, mask_tensor