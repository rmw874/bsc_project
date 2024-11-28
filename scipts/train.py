import torch
import os
import cv2
from scipy.ndimage import distance_transform_edt
from torch.nn.functional import relu, one_hot, cross_entropy
import numpy as np
from model import UNet
from dataset import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# HYPERPARAMS
BATCH_SIZE = 16
SIGMA = 10
EPOCHS = 200
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200, 2496)

def create_background_weights(mask, sigma=10.0):
    """
    Create weights for background pixels based on distance to non-background objects.
    
    Args:
        mask: Tensor of shape [height, width, num_classes]
        sigma: Parameter to control the decay of weights with distance
    
    Returns:
        weights: Tensor of same shape as input with higher weights for background pixels between objects
    """
    mask_np = mask.cpu().numpy()
    non_background = np.zeros_like(mask_np[..., 0])
    for i in range(5):  # excluding background class (5)
        non_background = np.logical_or(non_background, mask_np[..., i] > 0)
    
    distances = distance_transform_edt(~non_background) # how far each pixel is from any non-background object.
    weights = np.exp(-distances**2 / (2 * sigma**2)) # pixels closer get high weights; further get low weights

    
    class_weights = torch.ones_like(mask)
    class_weights[..., -1] = torch.from_numpy(weights).float()  # apply weights to background class
    return class_weights.to(mask.device)


class WeightedSegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        weights = create_background_weights(target.squeeze(0), SIGMA)

        # pred shape: [batch_size, num_classes, height, width]
        # target shape: [batch_size, height, width, num_classes]
        target = target.permute(0, 3, 1, 2) 
        weights = weights.permute(2, 0, 1).unsqueeze(0)
        
        loss = cross_entropy(pred, target, reduction='none')
        loss = loss * weights
        return loss.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Load the dataset
dataset = CustomDataset(image_dir='../data/processed/images', mask_dir='../data/processed/masks', target_size=TARGET_SIZE, num_classes=N_CLASSES)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)


model = UNet(N_CLASSES).to(device)
criterion = WeightedSegmentationLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()
torch.cuda.empty_cache()

loss_values = []


for epoch in range(EPOCHS):
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
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                pred = outputs.squeeze(0).cpu()
                pred = torch.argmax(pred, dim=0).numpy()
                
                weight_map = create_background_weights(mask_tensor.squeeze(0)).cpu().numpy()[..., -1]
                weight_map = cv2.resize(weight_map, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            
                plt.figure(figsize=(20, 5))
                
                plt.subplot(1, 4, 1)
                # tensor to numpy array
                img_resized = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                mask_resized = mask_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

                plt.imshow(img_resized)
                plt.title('Original Image (Resized)')
                
                plt.subplot(1, 4, 2)
                plt.imshow(mask_resized)
                plt.title('Ground Truth (Resized)')
                
                plt.subplot(1, 4, 3)
                plt.imshow(pred)
                plt.title(f'Prediction (Epoch {epoch+1})')
                
                plt.subplot(1, 4, 4)
                plt.imshow(weight_map, cmap='hot')
                plt.colorbar()
                plt.title('BG Weights')
                
                plt.suptitle(f'Loss: {loss.item():.4f}', fontsize=24)
                plt.savefig(f'../results/epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')

    torch.save(model.state_dict(), 'overfitted_model_w_penalty.pth')