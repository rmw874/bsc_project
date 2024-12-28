import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def common_resolution_divisible_by_32_from_folder(folder_path):
    """
    Finds the closest common resolution divisible by 32 for all images in a folder.

    Args:
        folder_path (str): Path to the folder containing images.
    """
    # List all files in the folder and filter for image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")  # Add more as needed
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                   if file.lower().endswith(image_extensions)]
    
    if not image_paths:
        raise ValueError("No images found in the specified folder.")
    
    # Store the dimensions of all images
    dimensions = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read {image_path}. Skipping.")
            continue
        height, width, _ = image.shape
        dimensions.append((width, height))
    
    if not dimensions:
        raise ValueError("No valid images found in the specified folder.")
    
    # Find the smallest width and height across all images
    min_width = min(dim[0] for dim in dimensions)
    min_height = min(dim[1] for dim in dimensions)
    
    # Adjust dimensions to be divisible by 32 while ensuring they remain <= smallest image dimensions
    common_width = ((min_width // 32) * 32)
    common_height = ((min_height // 32) * 32)
    
    # Ensure the common resolution is <= dimensions of all images
    for width, height in dimensions:
        if common_width > width or common_height > height:
            common_width = ((width // 32) * 32)
            common_height = ((height // 32) * 32)

    return common_width, common_height