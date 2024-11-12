import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import os

# Define the U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
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
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        # Final output
        return self.final_conv(d1)


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

        # Apply transformations to image
        if self.transform_img:
            img = self.transform_img(img)

        # Convert mask colors to class labels
        mask = np.array(mask)
        label_array = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for rgb, class_label in self.color_to_class.items():
            matches = (mask == rgb).all(axis=-1)
            label_array[matches] = class_label

        # Resize label_array
        label_array = Image.fromarray(label_array).resize((512, 512), resample=Image.NEAREST)
        label_array = np.array(label_array)

        # Convert to tensor
        label_tensor = torch.from_numpy(label_array).long()

        return img, label_tensor, self.image_filenames[idx]  # Also return the filename

# Metrics
def pixel_accuracy(output, mask):
    _, predicted = torch.max(output, 1)  # Get the predicted class for each pixel
    correct = (predicted == mask).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

def intersection_over_union(output, mask, num_classes):
    _, predicted = torch.max(output, 1)
    iou_per_class = []
    for cls in range(1, num_classes+1):
        pred_inds = predicted == cls
        target_inds = mask == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union > 0:
            iou_per_class.append((intersection / union).item())
    if len(iou_per_class) == 0:
        return 0.0
    return sum(iou_per_class) / len(iou_per_class)

def dice_coefficient(output, mask, num_classes):
    _, predicted = torch.max(output, 1)
    dice_per_class = []
    for cls in range(1, num_classes+1):
        pred_inds = predicted == cls
        target_inds = mask == cls
        intersection = (pred_inds & target_inds).sum().float()
        dice = (2. * intersection) / (pred_inds.sum().float() + target_inds.sum().float() + 1e-6)
        if (pred_inds.sum() + target_inds.sum()) > 0:
            dice_per_class.append(dice.item())
    if len(dice_per_class) == 0:
        return 0.0
    return sum(dice_per_class) / len(dice_per_class)


def train_model(model, train_loader, optimizer, criterion, num_epochs, device, num_classes):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_iou = 0.0
        running_dice = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels, filenames in progress_bar:  # Capture the filenames
            images, labels = images.to(device), labels.to(device)

            # Adjust labels dimensions if necessary
            labels = labels.squeeze(1)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc = pixel_accuracy(outputs, labels)
            iou = intersection_over_union(outputs, labels, num_classes)
            dice = dice_coefficient(outputs, labels, num_classes)

            # Update metrics
            running_loss += loss.item()
            running_accuracy += acc
            running_iou += iou
            running_dice += dice

            # Print the filenames for this batch
            print(f"Processing image: {filenames[0]}")  # Print the filename (batch size is 1)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss / len(train_loader):.4f}',
                'Acc': f'{running_accuracy / len(train_loader):.4f}',
                'IoU': f'{running_iou / len(train_loader):.4f}',
                'Dice': f'{running_dice / len(train_loader):.4f}'
            })
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, '
              f'Acc: {running_accuracy/len(train_loader):.4f}, IoU: {running_iou/len(train_loader):.4f}, '
              f'Dice: {running_dice/len(train_loader):.4f}')



# Prepare transformations
transform_img = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

# Set directories
image_dir = 'data/images/'
mask_dir = 'data/masks/'

# Initialize dataset and dataloader
train_dataset = SegmentationDataset(image_dir, mask_dir, transform_img=transform_img)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# Device setup (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
num_classes = 175  # Assuming 35 rows and 5 columns in your cell layout
model = UNet(in_channels=3, out_channels=num_classes).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Start training
num_epochs = 200
train_model(model, train_loader, optimizer, criterion, num_epochs, device, num_classes)
