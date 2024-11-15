# scripts/preprocess_data.py

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

def create_boundary_mask(cell_mask):
    """
    Converts a cell mask into a boundary mask by detecting edges.
    """
    # Assuming cell_mask is a numpy array with RGB values
    gray_mask = cv2.cvtColor(cell_mask, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_mask, threshold1=50, threshold2=150)
    return edges

def get_train_transforms():
    return Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])

def get_val_transforms():
    return Compose([
        A.Resize(512, 512),
    ])

def preprocess_and_save(images_dir, masks_dir, output_images_dir, output_masks_dir, num_augmentations=10):
    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))
    assert len(images) == len(masks), "Number of images and masks should be equal"

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    transform = get_train_transforms()

    for idx in tqdm(range(len(images))):
        img_path = os.path.join(images_dir, images[idx])
        mask_path = os.path.join(masks_dir, masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cell_mask = cv2.imread(mask_path)
        cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2RGB)

        # Create boundary mask
        boundary_mask = create_boundary_mask(cell_mask)

        # Save original image and mask
        cv2.imwrite(os.path.join(output_images_dir, f'image_{idx}_orig.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_masks_dir, f'mask_{idx}_orig.png'), boundary_mask)

        # Generate augmented images and masks
        for i in range(num_augmentations):
            augmented = transform(image=image, mask=boundary_mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

            # Save augmented images and masks
            cv2.imwrite(
                os.path.join(output_images_dir, f'image_{idx}_aug_{i}.png'),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(output_masks_dir, f'mask_{idx}_aug_{i}.png'),
                aug_mask
            )

if __name__ == "__main__":
    # Define paths
    raw_images_dir = 'data/raw/images/'
    raw_masks_dir = 'data/raw/masks/'

    train_images_dir = 'data/processed/train/images/'
    train_masks_dir = 'data/processed/train/masks/'

    val_images_dir = 'data/processed/val/images/'
    val_masks_dir = 'data/processed/val/masks/'

    # Split data into training and validation (using the first image for validation)
    preprocess_and_save(
        images_dir=raw_images_dir,
        masks_dir=raw_masks_dir,
        output_images_dir=train_images_dir,
        output_masks_dir=train_masks_dir,
        num_augmentations=10  # Adjust the number of augmentations as needed
    )

    # For validation data, apply only resizing
    val_transform = get_val_transforms()

    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    # Assuming the first image is used for validation
    val_image_path = os.path.join(raw_images_dir, os.listdir(raw_images_dir)[0])
    val_mask_path = os.path.join(raw_masks_dir, os.listdir(raw_masks_dir)[0])

    image = cv2.imread(val_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cell_mask = cv2.imread(val_mask_path)
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2RGB)

    # Create boundary mask
    boundary_mask = create_boundary_mask(cell_mask)

    augmented = val_transform(image=image, mask=boundary_mask)
    val_image = augmented['image']
    val_mask = augmented['mask']

    cv2.imwrite(os.path.join(val_images_dir, 'image_val.png'), cv2.cvtColor(val_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(val_masks_dir, 'mask_val.png'), val_mask)