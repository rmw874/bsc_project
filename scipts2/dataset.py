import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from utils import color2class

class PirateLogDataset(Dataset):
    def __init__(self, mask_dir, img_dir, target_size=(3200,2496), num_classes=6):
        # Initialize the dataset
        # Load data, apply initial transformations, etc.
        self.data = img_dir
        self.labels = mask_dir
        self.target_size = target_size
        self.num_classes = num_classes
    

        # sort
        self.data = sorted(os.listdir(self.data))
        self.labels = sorted(os.listdir(self.labels))

        # assert that the number of images and masks are the same
        assert len(self.data) == len(self.labels), "Number of images and masks must be the same"


    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the sample at index 'idx'
        sample_path = os.path.join(self.data, self.data[idx])
        label_path = os.path.join(self.labels, self.labels[idx])

        
         # Read image and mask using cv2
        img = cv2.imread(sample_path)
        mask = cv2.imread(label_path)

        # threshold and erode the img
        _, img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for small erosion
        img = cv2.erode(img, kernel, iterations=2)
        img = img / 255.0

        # resize the image and mask
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)

        class_mask = color2class(mask)
        mask_resized = cv2.resize(class_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_resized_tensor = torch.tensor(mask_resized, dtype=torch.long)
        one_hot_mask_resized = one_hot(mask_resized_tensor, num_classes=6)
        mask_tensor = one_hot_mask_resized.permute(2, 0, 1).float().unsqueeze(0)
        mask_tensor = mask_tensor.permute(0, 2, 3, 1)  # [batch, height, width, num_classes]
        
        return img_tensor, mask_tensor