import torch
import numpy as np

def color2class(mask):
    green_channel = mask[:, :, 1] # 1 for green since (hieght, width (R, G, B))
    class_mask = np.zeros_like(green_channel)

    class_mask[green_channel == 1] = 0  # Year 
    class_mask[green_channel == 2] = 1  # Date
    class_mask[green_channel == 3] = 2  # Latitude
    class_mask[green_channel == 4] = 3  # Longitude
    class_mask[green_channel == 5] = 4  # Water Temperature
    class_mask[green_channel == 255] = 5  # Background

    return class_mask