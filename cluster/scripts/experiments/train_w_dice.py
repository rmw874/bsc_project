from model import UNet
from loss import WeightedSegmentationLoss
from dataset import PirateLogDataset
import torch
import os
import cv2
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import signal
import sys
from contextlib import contextmanager
import torch.multiprocessing as mp


# HYPERPARAMS
BATCH_SIZE = 2
SIGMA = 5
EPOCHS = 200
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200//2, 2496//2)

def setup_device():
    """Setup and verify GPU device"""
    try:
        if torch.cuda.is_available():
            # Check memory before starting
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if free_memory < 1e9:  # Less than 1GB free
                raise RuntimeError("Insufficient GPU memory available")
            return torch.device("cuda")
        return torch.device("cpu")
    except RuntimeError as e:
        print(f"GPU error: {e}")
        return torch.device("cpu")

def setup_environment():
    """Setup CUDA environment variables"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.75'
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)


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

@contextmanager
def train_step_context():
    """Context manager for safe training step execution"""
    try:
        yield
    except Exception as e:
        print(f"Error in training step: {e}")
        cleanup()
    finally:
        cleanup()

def visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss, criterion):
    """Visualize training progress"""
    with torch.no_grad():
        pred = outputs[0].cpu()
        pred = torch.argmax(pred, dim=0).numpy()

        weight_map = criterion.create_background_weights(mask_tensor[0:1]).cpu().numpy()[0]
        weight_map = cv2.resize(weight_map, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        mask_resized = mask_tensor[0].cpu().numpy()

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
        
        plt.suptitle(f'Loss: {loss:.4f}', fontsize=24)
        plt.savefig(f'results/dice/epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')
        plt.close()

def train_epoch(epoch, model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    
    with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
        for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
            with train_step_context():
                # Move data to device
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)
                
                # Forward pass with automatic mixed precision
                with autocast('cuda'):
                    outputs = model(img_tensor)
                    loss = criterion(outputs, mask_tensor)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                
                # Visualization
                if (epoch + 1) % 5 == 0 and batch_idx == 0:
                    visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss_value, criterion)
                
                # Clean up
                del outputs, loss
    
    return epoch_losses

def worker_init_fn(worker_id):
    print(f"Initializing worker {worker_id}")
    device_id = worker_id % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.manual_seed(worker_id)

def main():
    # Setup environment and device
    setup_environment()
    device = setup_device()
    
    # Initialize dataset and dataloader
    dataset = PirateLogDataset(
        img_dir='data/processed/images',
        mask_dir='data/processed/masks',
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Number of workers
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,  # Set to False for debugging
        prefetch_factor=None
    )
    
    # Initialize model and training components
    model = UNet(N_CLASSES).to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    criterion = WeightedSegmentationLoss(num_classes=N_CLASSES, sigma=SIGMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler('cuda')
    
    # Training loop
    all_losses = []
    for epoch in range(EPOCHS):
        epoch_losses = train_epoch(epoch, model, dataloader, criterion, optimizer, scaler, device)
        all_losses.extend(epoch_losses)
        
        # Save model
        torch.save(model.state_dict(), 'results/dice/latest_model_dice.pth')
        
        # Print metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    
    # Plot final loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_losses)), all_losses, label='Training Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('results/dice/loss_curve.png')
    plt.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()