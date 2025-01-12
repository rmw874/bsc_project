from loss import TverskyLoss, FocalLoss, CrossEntropyLoss, WeightedCompositeLoss
from dataset import PirateLogDataset
import torch
import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import gc
import signal
import sys
from contextlib import contextmanager
import albumentations as A
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from visualization import visualize_batch_with_boundaries
from metrics import (
    calculate_confusion_metrics,
    calculate_mean_iou,
    calculate_boundary_metrics,
    calculate_instance_separation,

)
from config import (
    EPOCHS, LEARNING_RATE, N_CLASSES, TARGET_SIZE, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
    RESULTS_DIR, BEST_MODEL_PATH, TVERSKY_PARAMS, DATALOADER_PARAMS, TRAIN_VIZ_FREQ, CUDA_LAUNCH_BLOCKING,
    PYTORCH_CUDA_ALLOC_CONF
)
from validation import validate_epoch

# Define Albumentations transforms
albumentations_transform = A.Compose([
    A.Affine(
        rotate=(-2.5, 0),
        p=0.6,
        mode=cv2.BORDER_CONSTANT,
        cval=(255, 255, 255),
        interpolation=cv2.INTER_NEAREST,
    ),
    A.Affine(
        shear=(-1.5, 1.5),
        p=0.6,
        mode=cv2.BORDER_CONSTANT,
        cval=(255, 255, 255),
        interpolation=cv2.INTER_NEAREST,
    ),
    A.Affine(
        translate_percent={"x": (-0.025, 0.025)},
        p=0.6,
        mode=cv2.BORDER_CONSTANT,
        cval=(255, 255, 255),
        interpolation=cv2.INTER_NEAREST,
    ),
    A.ElasticTransform(
        alpha=25.0,
        sigma=10,
        p=0.3,
        border_mode=cv2.BORDER_CONSTANT,
        value=1,
    ),
    A.GridDistortion(
        num_steps=3,
        distort_limit=0.2,
        p=0.3,
        border_mode=cv2.BORDER_CONSTANT,
        value=1,
    ),
])

def setup_device():
    """Initialize and set up the training device."""
    try:
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if free_memory < 1e9:
                raise RuntimeError("Insufficient GPU memory available")
            return torch.device("cuda")
        return torch.device("cpu")
    except RuntimeError as e:
        print(f"GPU error: {e}")
        return torch.device("cpu")

def setup_environment():
    """Set up the training environment and signal handlers."""
    os.environ["CUDA_LAUNCH_BLOCKING"] = CUDA_LAUNCH_BLOCKING
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

def cleanup():
    """Clean up GPU memory and garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_handler():
    """Handle cleanup when the script is interrupted."""
    print("\nCleaning up and exiting...")
    cleanup()
    sys.exit(0)

@contextmanager
def train_step_context():
    """Context manager for safe training step execution."""
    try:
        yield
    except Exception as e:
        print(f"Error in training step: {e}")
        cleanup()
    finally:
        cleanup()

def train_epoch(epoch, model, dataloader, criterion, optimizer, scaler, device, num_classes):
    """Train the model for one epoch."""
    model.train()
    epoch_losses = []
    mean_ious = []
    boundary_f1_scores = []
    instance_sep_scores = []

    with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
        for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
            with train_step_context():
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)

                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    outputs = model(img_tensor)["out"]
                    loss = criterion(outputs, mask_tensor)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_value = loss.item()
                epoch_losses.append(loss_value)

                pred = torch.argmax(outputs, dim=1)

                results = calculate_confusion_metrics(num_classes, pred, mask_tensor)
                mean_iou_value = calculate_mean_iou(results).item()
                
                boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, num_classes)
                instance_sep_score = calculate_instance_separation(outputs, mask_tensor, num_classes)

                # Store all metrics
                mean_ious.append(mean_iou_value)
                boundary_f1_scores.append(boundary_metrics["boundary_f1"])
                instance_sep_scores.append(instance_sep_score)

                pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "mean_iou": f"{mean_iou_value:.4f}",
                    "boundary_f1": f'{boundary_metrics["boundary_f1"]:.4f}',
                    "instance_sep": f"{instance_sep_score:.4f}"
                })

                # Visualization
                if (epoch + 1) % TRAIN_VIZ_FREQ == 0 and batch_idx == 0:
                    try:
                        visualize_batch_with_boundaries(
                            epoch,
                            img_tensor,
                            mask_tensor,
                            outputs,
                            loss_value,
                            num_classes,
                            index=0,
                        )
                        print("Visualization completed successfully")
                    except Exception as e:
                        print(f"Visualization failed: {str(e)}")
                        print("Debug info - tensor devices:")
                        print(f"Image: {img_tensor.device}")
                        print(f"Mask: {mask_tensor.device}")
                        print(f"Outputs: {outputs.device}")

                del outputs, loss

    return epoch_losses, {
        "mean_iou": sum(mean_ious) / (len(mean_ious) + 1e-7),
        "boundary_f1": sum(boundary_f1_scores) / (len(boundary_f1_scores) + 1e-7),
        "instance_sep": sum(instance_sep_scores) / (len(instance_sep_scores) + 1e-7)
    }

def plot_training_metrics(train_losses, val_losses, val_ious, boundary_f1s, instance_seps):
    """Plot and save training metrics with proper tensor handling."""
    
    # Convert GPU tensors to CPU numpy arrays if needed
    def to_numpy(data):
        if isinstance(data, (list, tuple)):
            return [to_numpy(x) for x in data]
        if torch.is_tensor(data):
            return data.cpu().numpy()
        return data
    
    # Convert all metrics to CPU/numpy
    boundary_f1s = to_numpy(boundary_f1s)
    instance_seps = to_numpy(instance_seps)
    train_losses = to_numpy(train_losses)
    val_losses = to_numpy(val_losses)
    val_ious = to_numpy(val_ious)
    
    # Create epoch numbers
    epochs = list(range(1, len(train_losses) + 1))
    
    # 1. Normalized losses and accuracy
    train_losses_norm = [loss / max(train_losses) for loss in train_losses]
    val_losses_norm = [loss / max(val_losses) for loss in val_losses]
    
    data = pd.DataFrame({
        "Epoch": epochs * 3,
        "Value": train_losses_norm + val_losses_norm + val_ious,
        "Metric": ["Train Loss"] * len(epochs) + 
                 ["Validation Loss"] * len(epochs) + 
                 ["Validation Mean IOU"] * len(epochs)
    })

    # Plot normalized metrics
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_metrics_normalized.png")
    plt.close()

    # 2. Raw losses
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss', linewidth=2.5)
    plt.plot(epochs, val_losses, 'r-', marker='o', label='Validation Loss', linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Raw Loss Values", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/raw_losses.png")
    plt.close()

    # 3. Boundary metrics
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, boundary_f1s, 'g-', marker='o', label='Boundary F1', linewidth=2.5)
    plt.plot(epochs, instance_seps, 'm-', marker='o', label='Instance Separation', linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Boundary and Instance Separation Metrics", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/boundary_metrics.png")
    plt.close()

def main():
    """Main training function."""
    torch.cuda.set_device(0)
    device = setup_device()
    torch.cuda.empty_cache()
    setup_environment()

    # Initialize datasets
    train_dataset = PirateLogDataset(
        img_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES,
        transform=albumentations_transform,
    )

    val_dataset = PirateLogDataset(
        img_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, **DATALOADER_PARAMS, shuffle=True)
    val_loader = DataLoader(val_dataset, **DATALOADER_PARAMS, shuffle=False)

    # Initialize model
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, N_CLASSES, kernel_size=1)
    model = model.to(device)

    # Initialize loss function with parameters from config
    # criterion = BoundaryAwareTverskyLoss(
    #     num_classes=N_CLASSES,
    #     from_logits=True,
    #     class_weights=CLASS_WEIGHTS,
    #     **TVERSKY_PARAMS
    # )

    # criterion = TverskyLoss(
    # num_classes=N_CLASSES,
    # **TVERSKY_PARAMS,
    # from_logits=True
    # )

    criterion = WeightedCompositeLoss()

    # Initialize optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = GradScaler("cuda")

    # Training loop variables
    train_losses = []
    val_losses = []
    val_ious = []
    boundary_f1s = []
    instance_seps = []
    val_boundary_f1s = []
    val_instance_seps = []
    best_instance_sep = float('-inf')
    best_epoch = -1

    for epoch in range(EPOCHS):
        # Training phase
        epoch_losses, metrics = train_epoch(
            epoch, model, train_loader, criterion, optimizer, scaler, device, N_CLASSES
        )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        boundary_f1s.append(metrics["boundary_f1"])
        instance_seps.append(metrics["instance_sep"])

        print(f"Epoch {epoch + 1} Metrics:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Boundary F1: {metrics['boundary_f1']:.4f}")
        print(f"  Instance Separation: {metrics['instance_sep']:.4f}")

        val_loss, val_metrics = validate_epoch(
            epoch, model, val_loader, criterion, device, N_CLASSES  # Correct - using val_loader
        )
        val_loss, val_metrics = validate_epoch(epoch, model, val_loader, criterion, device, N_CLASSES)
        val_boundary_f1s.append(val_metrics['boundary_f1'])
        val_instance_seps.append(val_metrics['instance_sep'])
        val_losses.append(val_loss)
        val_ious.append(val_metrics["mean_iou"])

        # Save model if loss improves
        current_instance_sep = val_metrics['instance_sep']
        if current_instance_sep > best_instance_sep:
            best_instance_sep = current_instance_sep
            best_epoch = epoch
            print(f"Best epoch: {best_epoch}")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model with instance separation score: {best_instance_sep:.4f}")

    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, val_ious, val_boundary_f1s, val_instance_seps)



if __name__ == "__main__":
    main()