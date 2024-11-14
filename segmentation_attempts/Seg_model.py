import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class SegUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegUNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = self.up_conv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.up_conv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.up_conv(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two conv layers and ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        """Upsampling via ConvTranspose2d followed by a ReLU."""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder (Downsampling)
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder (Upsampling)
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final output layer
        return self.final_conv(d1)


def generate_color_to_class_mapping(num_rows, num_columns):
    """Generates a color-to-class mapping where each (R, G, 0) color maps to a class.
    
    Args:
        num_rows (int): Number of rows.
        num_columns (int): Number of columns.
        
    Returns:
        dict: A mapping of (R, G, B) -> class label.
    """
    color_to_class = {}
    class_label = 1  # Starting from class 1
    
    # Iterate over rows and columns to generate the mapping
    for row in range(1, num_rows + 1):  # Rows 1 to num_rows
        for col in range(1, num_columns + 1):  # Columns 1 to num_columns
            rgb_color = (row, col, 0)  # Set RGB color (row, col, 0)
            color_to_class[rgb_color] = class_label
            class_label += 1  # Increment class label for the next cell
    
    return color_to_class

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform_img=None, transform_mask=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        # Use the generated color-to-class mapping
        self.color_to_class = generate_color_to_class_mapping(35, 5)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        # Convert mask colors to class labels
        mask = np.array(mask)
        label_array = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for rgb, class_label in self.color_to_class.items():
            label_array[(mask == rgb).all(axis=-1)] = class_label
        
        # Apply transforms (resize and convert to tensor)
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            label_tensor = self.transform_mask(Image.fromarray(label_array))
        else:
            label_tensor = torch.from_numpy(label_array).long()
        
        return img, label_tensor

