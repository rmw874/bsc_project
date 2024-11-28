import torch
import os
import cv2
from scipy.ndimage import distance_transform_edt
from torch.nn.functional import relu, one_hot, cross_entropy
import numpy as np
from model import UNet
from tqdm import tqdm
from dataset import PirateLogDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import gc
import signal
import sys
from contextlib import contextmanager

try:
    if torch.cuda.is_available():
        # Check memory before starting
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        if free_memory < 1e9:  # Less than 1GB free
            raise RuntimeError("Insufficient GPU memory available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except RuntimeError as e:
    print(f"GPU error: {e}")
    device = torch.device("cpu")


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_handler(signum, frame):
    """Handle script termination"""
    print("\nCleaning up and exiting...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)

def train_step_scope():
    """Context manager for training step memory management"""
    try:
        yield
    finally:
        cleanup()

# HYPERPARAMS
BATCH_SIZE = 4
SIGMA = 10
EPOCHS = 200
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200//2, 2496//2)

def create_background_weights(mask, sigma=10.0):
    """
    Create weights for background pixels based on distance to non-background objects.
    
    Args:
        mask: Tensor of shape [batch_size, height, width]
        sigma: Parameter to control the decay of weights with distance.
    
    Returns:
        weights: Tensor of shape [batch_size, height, width] - background weights
    """
    mask_np = mask.cpu().numpy()
    batch_weights = []

    for batch_idx in range(mask_np.shape[0]):
        # Create binary mask of non-background pixels (everything that's not class 5)
        non_background = mask_np[batch_idx] != 5  # Assuming 5 is the background class
        
        # Calculate distance transform
        distances = distance_transform_edt(~non_background)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        batch_weights.append(weights)
    
    # Stack weights and convert back to tensor
    batch_weights = np.stack(batch_weights)
    return torch.tensor(batch_weights, dtype=torch.float32).to(mask.device)


class WeightedSegmentationLoss(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of shape [batch_size, num_classes, height, width]
            target: Tensor of shape [batch_size, height, width] containing class indices
        Returns:
            Cross entropy loss with higher weights for background pixels near class boundaries
        """
        # Get distance-based weights that emphasize background pixels near class boundaries
        background_weights = create_background_weights(target, SIGMA)  # [batch_size, height, width]
        
        # Calculate standard cross entropy loss (per-pixel)
        pixel_losses = cross_entropy(pred, target, reduction='none')  # [batch_size, height, width]
        
        # Apply weights only to background pixels (where target == num_classes-1)
        background_mask = (target == self.num_classes - 1)
        weighted_losses = pixel_losses.clone()
        weighted_losses[background_mask] = pixel_losses[background_mask] * background_weights[background_mask]
        
        return weighted_losses.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# print current dir
print(os.getcwd())
# Load the dataset
dataset = PirateLogDataset(img_dir='data/processed/images', mask_dir='data/processed/masks', target_size=TARGET_SIZE, num_classes=N_CLASSES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


model = UNet(N_CLASSES).to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
criterion = WeightedSegmentationLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scaler = torch.amp.GradScaler('cuda')
torch.cuda.empty_cache()

loss_values = []
best_val_loss = float('inf')  # Track the best validation loss (assuming validation loss is used)

torch.cuda.empty_cache()

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    # Wrap the dataloader with tqdm for batch progress tracking
    for img_tensor, mask_tensor in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
        img_tensor = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(img_tensor)  # [batch, num_classes, height, width]
            loss = criterion(outputs, mask_tensor)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_values.append(loss.item())

        # Visualization
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                # Select the first sample in the batch for visualization
                pred = outputs[0].cpu()  # [num_classes, height, width]
                pred = torch.argmax(pred, dim=0).numpy()  # [height, width]

                # Get background weights for visualization
                weight_map = create_background_weights(mask_tensor[0:1]).cpu().numpy()[0]  # [height, width]
                weight_map = cv2.resize(weight_map, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

                img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()  # [height, width, channels]
                mask_resized = mask_tensor[0].cpu().numpy()  # [height, width]

                plt.figure(figsize=(20, 5))

                plt.subplot(1, 4, 1)
                plt.imshow(img_resized)
                plt.title('Original Image (Resized)')
                
                plt.subplot(1, 4, 2)
                plt.imshow(mask_resized, cmap='tab10') 
                plt.title('Ground Truth (Resized)')
                
                plt.subplot(1, 4, 3)
                plt.imshow(pred, cmap='tab10')
                plt.title(f'Prediction (Epoch {epoch+1})')
                
                plt.subplot(1, 4, 4)
                plt.imshow(weight_map, cmap='hot')
                plt.colorbar()
                plt.title('BG Weights')
                
                plt.suptitle(f'Loss: {loss.item():.4f}', fontsize=24)
                plt.savefig(f'results/epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')
                plt.close()

                del pred, weight_map, img_resized, mask_resized

    # Save the latest model
    torch.save(model.state_dict(), 'latest_model.pth')
    cleanup()

    avg_loss = sum(loss_values) / len(loss_values)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")