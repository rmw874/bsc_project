import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from data_preprocessing import load_images_and_masks
from model_utils import UNet
import datetime

def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
    print(message)

# HYPERPARAMS
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
torch.cuda.empty_cache()

CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LOG_FILE = "training_log.txt"

raw_image_dir = "data/raw/images"
raw_mask_dir = "data/raw/masks"
images, masks = load_images_and_masks(raw_image_dir, raw_mask_dir)

# train_images, train_masks = images[:80], masks[:80]
train_images, train_masks = images, masks

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_masks))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Train dataset size: {len(train_dataset)}")

model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

log_message("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:  # Log progress every 10 batches
            log_message(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    train_loss /= len(train_loader)
    log_message(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {train_loss:.4f}")

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        log_message(f"Checkpoint saved at {checkpoint_path}")

# Save final model
final_model_path = "models/best_model.pth"
torch.save(model.state_dict(), final_model_path)
log_message(f"Final model saved at {final_model_path}")

log_message("Training completed.")
