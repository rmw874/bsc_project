from model import UNet
from loss import WeightedSegmentationLoss
from dataset import PirateLogDataset
import torch
import os
import cv2
from torch.nn.functional import relu, one_hot, cross_entropy
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
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


# HYPERPARAMS
BATCH_SIZE = 2
SIGMA = 5
EPOCHS = 200
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200//2, 2496//2)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Force synchronous error reporting (helpful for debugging)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.75'  # Adjust memory allocation settings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


# Load the dataset
dataset = PirateLogDataset(img_dir='data/processed/images', mask_dir='data/processed/masks', target_size=TARGET_SIZE, num_classes=N_CLASSES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

model = UNet(N_CLASSES).to(device)
model = nn.DataParallel(model, device_ids=[0, 1])
criterion = WeightedSegmentationLoss(num_classes=N_CLASSES, sigma=SIGMA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

torch.cuda.empty_cache()

loss_values = []

torch.cuda.empty_cache()
scaler = torch.amp.GradScaler('cuda')

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    # Wrap the dataloader with tqdm for batch progress tracking
    for img_tensor, mask_tensor in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
        print(f"Loading batch...")
        img_tensor = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        print(f"Batch loaded, starting forward pass...")
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(img_tensor)  # [batch, num_classes, height, width]
        print(f"Forward pass complete, starting backward pass...")
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
                weight_map = criterion.create_background_weights(mask_tensor[0:1]).cpu().numpy()[0]  # [height, width]
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
                plt.savefig(f'results/dice/epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')
                plt.close()

                del pred, weight_map, img_resized, mask_resized

    # Save the latest model
    torch.save(model.state_dict(), 'results/dice/latest_model_dice.pth')
    cleanup()

    avg_loss = sum(loss_values) / len(loss_values)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

#fix this part vro
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), loss_values, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()