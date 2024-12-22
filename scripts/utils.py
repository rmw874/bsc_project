import torch
import numpy as np

def color2class(mask):
    if mask is None:
        raise ValueError(f"Failed to read mask at {mask}")
    green_channel = mask[:, :, 1]  # 1 for green since (height, width, channels)
    class_mask = np.zeros_like(green_channel)

    valid_values = [1, 2, 3, 4, 5, 255]

    unique_values = np.unique(green_channel)
    if not np.all(np.isin(unique_values, valid_values)):
        print(f"Warning: Mask contains unexpected green channel values: {unique_values}")

    class_mask[green_channel == 1] = 0  # Year 
    class_mask[green_channel == 2] = 1  # Date
    class_mask[green_channel == 3] = 2  # Latitude
    class_mask[green_channel == 4] = 3  # Longitude
    class_mask[green_channel == 5] = 4  # Water Temperature
    class_mask[green_channel == 255] = 5  # Background
    class_mask[green_channel == 0] = 5  # Also background (maybe?)

    return class_mask