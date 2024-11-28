import torch
import os
import cv2
from scipy.ndimage import distance_transform_edt
from torch.nn.functional import relu, one_hot, cross_entropy
import numpy as np
from model import UNet
from dataset import PirateLogDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

# HYPERPARAMS
BATCH_SIZE = 2
SIGMA = 10
EPOCHS = 200
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200//2, 2496//2)

def create_background_weights(mask, sigma=10.0):
    """
    Create weights for background pixels based on distance to non-background objects.
    Supports batched masks.
    
    Args:
        mask: Tensor of shape [batch_size, height, width, num_classes]
        sigma: Parameter to control the decay of weights with distance.
    
    Returns:
        weights: Tensor of shape [batch_size, height, width, num_classes].
    """
    mask_np = mask.cpu().numpy()  # Convert to numpy
    batch_weights = []

    for batch_idx in range(mask_np.shape[0]):
        non_background = np.zeros_like(mask_np[batch_idx, ..., 0])
        for i in range(5):  # Exclude background class (5)
            non_background = np.logical_or(non_background, mask_np[batch_idx, ..., i] > 0)
        
        distances = distance_transform_edt(~non_background)
        weights = np.exp(-distances**2 / (2 * sigma**2))

        class_weights = np.ones_like(mask_np[batch_idx])
        class_weights[..., -1] = weights
        batch_weights.append(class_weights)
    
    batch_weights = np.stack(batch_weights)
    return torch.tensor(batch_weights, dtype=torch.float32).to(mask.device)


class WeightedSegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # pred shape: [batch_size, num_classes, height, width] 
        # target shape: [batch_size, height, width]
        one_hot_target = one_hot(target, num_classes=pred.size(1)).float()
        weights = create_background_weights(one_hot_target, SIGMA)

        loss = cross_entropy(pred, target, reduction='none')  # Expects class indices
        loss = loss * weights[..., -1]  # Apply weights for background class
        return loss.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# print current dir
print(os.getcwd())
# Load the dataset
dataset = PirateLogDataset(img_dir='data/processed/images', mask_dir='data/processed/masks', target_size=TARGET_SIZE, num_classes=N_CLASSES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


model = UNet(N_CLASSES).to(device)
criterion = WeightedSegmentationLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()
torch.cuda.empty_cache()

loss_values = []


for epoch in range(EPOCHS):
  print("Epoch:", epoch+1)
  for img_tensor, mask_tensor in dataloader:

      img_tensor = img_tensor.to(device)
      mask_tensor = mask_tensor.to(device)
      
      model.train()
      optimizer.zero_grad()
      
      # Forward pass with mixed precision
      with torch.cuda.amp.autocast():
          outputs = model(img_tensor)  # [batch, num_classes, height, width]
          loss = criterion(outputs, mask_tensor)
      
      # Backward pass with gradient scaling
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      loss_values.append(loss.item())
      
      # Visualization
      if (epoch + 1) % 1 == 0:
        with torch.no_grad():
          model.eval()
          
          # Select the first sample in the batch for visualization
          pred = outputs[0].cpu()  # [num_classes, height, width]
          pred = torch.argmax(pred, dim=0).numpy()  # [height, width]

          # Create background weights for the first mask in the batch
          weight_map = create_background_weights(mask_tensor[0:1]).cpu().numpy()[0, ..., -1]  # [height, width]
          weight_map = cv2.resize(weight_map, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

          # Prepare the image and mask
          img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()  # [height, width, channels]
          mask_resized = mask_tensor[0].cpu().numpy()  # [height, width]

          plt.figure(figsize=(20, 5))
          
          plt.subplot(1, 4, 1)
          plt.imshow(img_resized)
          plt.title('Original Image (Resized)')
          
          plt.subplot(1, 4, 2)
          plt.imshow(mask_resized, cmap='tab10')  # Use a colormap for class indices
          plt.title('Ground Truth (Resized)')
          
          plt.subplot(1, 4, 3)
          plt.imshow(pred, cmap='tab10')  # Use a colormap for predictions
          plt.title(f'Prediction (Epoch {epoch+1})')
          
          plt.subplot(1, 4, 4)
          plt.imshow(weight_map, cmap='hot')
          plt.colorbar()
          plt.title('BG Weights')
          
          plt.suptitle(f'Loss: {loss.item():.4f}', fontsize=24)
          plt.savefig(f'results/epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')

      torch.save(model.state_dict(), 'overfitted_model_w_penalty.pth')