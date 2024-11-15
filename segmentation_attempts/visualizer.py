import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the SegmentationDataset class directly in this file
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))  # Sorted list of image files
        self.mask_filenames = sorted(os.listdir(mask_dir))    # Sorted list of mask files
        self.transform_img = transform_img

        # Static color-to-class mapping (35 rows, 5 columns)
        self.color_to_class = self.generate_color_to_class_mapping(35, 5)

    def generate_color_to_class_mapping(self, num_rows, num_columns):
        """Generates a color-to-class mapping where each (R, G, 0) color maps to a class."""
        color_to_class = {}
        class_label = 1
        for row in range(1, num_rows + 1):
            for col in range(1, num_columns + 1):
                rgb_color = (row, col, 0)  # Color (R, G, 0)
                color_to_class[rgb_color] = class_label
                class_label += 1
        return color_to_class

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load the image and corresponding mask based on filenames
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Open the image and mask files
        img = Image.open(image_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("RGB")  # Ensure mask is RGB (remove alpha channel if present)

        # Apply transformations to the image (if any)
        if self.transform_img:
            img = self.transform_img(img)

        # Convert mask colors to class labels
        mask_np = np.array(mask)
        label_array = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)

        # Match mask colors to class labels
        for rgb, class_label in self.color_to_class.items():
            matches = (mask_np == rgb).all(axis=-1)
            label_array[matches] = class_label

        # Resize the label array
        label_array = Image.fromarray(label_array).resize((512, 512), resample=Image.NEAREST)
        label_array = np.array(label_array)

        # Convert label array to tensor
        label_tensor = torch.from_numpy(label_array).long()

        return img, label_tensor, self.image_filenames[idx]  # Return filename for visualization purposes

# Visualization functions
def visualize_image_mask_pair(dataset, idx):
    """Visualize the input image and its corresponding mask side-by-side."""
    img, mask, filename = dataset[idx]  # Get image, mask, and filename from dataset

    # Convert image and mask to numpy arrays for visualization
    img_np = np.transpose(img.numpy(), (1, 2, 0))  # Convert image tensor to HWC format
    mask_np = mask.numpy()

    # Create a subplot to visualize the image and mask side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image: {filename}")
    axes[0].axis("off")

    # Plot the mask
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title(f"Mask for: {filename}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_all_pairs(dataset):
    """Visualize all image-mask pairs in the dataset."""
    for idx in range(len(dataset)):
        visualize_image_mask_pair(dataset, idx)

# Main code to run visualization
if __name__ == "__main__":
    # Set directories for images and masks
    image_dir = 'data/images/'  # Adjust this path to where your images are stored
    mask_dir = 'data/masks/'    # Adjust this path to where your masks are stored

    # Prepare transformations
    transform_img = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

    # Initialize the dataset
    train_dataset = SegmentationDataset(image_dir, mask_dir, transform_img=transform_img)

    # Visualize all image-mask pairs
    visualize_all_pairs(train_dataset)
