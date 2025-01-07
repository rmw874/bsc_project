from scripts.old_files.model import UNet
from loss import loss_weight_scheduler, BoundaryAwareTverskyLoss
from dataset import PirateLogDataset
import torch
import os
import cv2
import pandas as pd
import seaborn as sns
from kornia.filters import sobel
import torch.nn.functional as F

from metrics import calculate_boundary_metrics, calculate_instance_separation, calculate_iou_per_class, plot_metric_heatmap, calculate_confusion_metrics, calculate_mean_iou, calculate_mean_dice, calculate_mean_pixel_accuracy, calculate_mean_precision, calculate_mean_recall, calculate_accuracy
from validation import validate_epoch

from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
import signal
import sys
from contextlib import contextmanager
import albumentations as A

# HYPERPARAMS
BATCH_SIZE = 2
EPOCHS = 1000
LEARNING_RATE = 1e-4
N_CLASSES = 6
ALPHA = 0.3
BETA = 0.7
BOUNDARY_WEIGHT = 5.0
TARGET_SIZE = (3200//1, 2496//1)
RUN_NAME = 'big_bat_run1_a3b7'

# Define Albumentations transforms
albumentations_transform = A.Compose([
    A.Affine(
        rotate=(-5, 0),
        p=0.6,
        mode=cv2.BORDER_CONSTANT,
        cval=(255, 255, 255),
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST
    ),
    A.Affine(
        shear=(-5, 5),
        p=0.6,
        mode=cv2.BORDER_CONSTANT,
        cval=(255, 255, 255),
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST
    ),
    A.Affine(
        translate_percent={"x": (-0.1, 0.4), "y": (-0.1, 0.1)},
        p=0.6,
        mode=cv2.BORDER_CONSTANT,
        cval=(255, 255, 255),
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST
    ),
    # Adding Elastic Deformation
    A.ElasticTransform(
        alpha=1.0,  # Controls the intensity of deformation (higher = more deformation)
        sigma=10,   # Controls the smoothness of the deformation (lower = sharper)
        p=0.3,      # Probability of applying the transformation
    ),
    # Adding Random Contrast Variation
    A.AutoContrast(
        p=0.3,             # Probability of applying the transformation
    )
], additional_targets={'mask': 'mask'})



def setup_device():
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.75'
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_handler(signum, frame):
    print("\nCleaning up and exiting...")
    cleanup()
    sys.exit(0)

@contextmanager
def train_step_context():
    try:
        yield
    except Exception as e:
        print(f"Error in training step: {e}")
        cleanup()
    finally:
        cleanup()

def visualize_batch_with_boundaries(epoch, img_tensor, mask_tensor, outputs, loss, criterion, N_CLASSES):
    """
    Visualize training progress including boundary detection
    
    Args:
        epoch: Current training epoch
        img_tensor: Input image tensor [B, C, H, W]
        mask_tensor: Ground truth mask tensor [B, H, W]
        outputs: Model predictions [B, C, H, W]
        loss: Current loss value
        criterion: Loss function object
        N_CLASSES: Number of classes
    """
    with torch.no_grad():
        # Get predictions
        pred = outputs[0].cpu()
        pred = torch.argmax(pred, dim=0).numpy()

        # Get original image and mask
        img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        mask_resized = mask_tensor[0].cpu().numpy()

        # Calculate metrics
        results = calculate_confusion_metrics(N_CLASSES, outputs, mask_tensor)
        mean_iou = calculate_mean_iou(results).item()
        accuracy = calculate_accuracy(results).item()
        mean_precision = calculate_mean_precision(results).item()
        mean_recall = calculate_mean_recall(results).item()
        
        # Calculate boundary metrics
        boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, N_CLASSES)
        
        # Generate boundary maps
        pred_softmax = F.softmax(outputs, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)
        
        # Convert to one-hot
        pred_onehot = F.one_hot(pred_labels, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        target_onehot = F.one_hot(mask_tensor, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        
        # Compute boundaries for visualization
        combined_pred_boundary = torch.zeros_like(pred_labels[0].float())
        combined_target_boundary = torch.zeros_like(mask_tensor[0].float())
        
        for c in range(N_CLASSES):
            pred_edges = torch.abs(sobel(pred_onehot[:, c:c+1]))
            target_edges = torch.abs(sobel(target_onehot[:, c:c+1]))
            
            combined_pred_boundary += (pred_edges[0, 0] > 0.5).float()
            combined_target_boundary += (target_edges[0, 0] > 0.5).float()
        
        # Create visualization
        plt.figure(figsize=(20, 15))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img_resized)
        plt.title('Preprocessed Image')
        plt.axis('off')

        # Ground truth mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask_resized, cmap='tab10')
        plt.title('Ground Truth')
        plt.axis('off')

        # Prediction mask
        plt.subplot(2, 3, 3)
        plt.imshow(pred, cmap='tab10')
        plt.title(f'Prediction (Epoch {epoch+1})')
        plt.axis('off')

        # Predicted boundaries
        plt.subplot(2, 3, 4)
        plt.imshow(combined_pred_boundary.cpu().numpy(), cmap='gray')
        plt.title('Predicted Boundaries')
        plt.axis('off')

        # Ground truth boundaries
        plt.subplot(2, 3, 5)
        plt.imshow(combined_target_boundary.cpu().numpy(), cmap='gray')
        plt.title('Ground Truth Boundaries')
        plt.axis('off')

        # IoU Heatmap
        plt.subplot(2, 3, 6)
        iou_scores = calculate_iou_per_class(results)
        class_labels = ['Year', 'Date', 'Latitude', 'Longitude', 'Temperature', 'Background']
        plot_metric_heatmap(iou_scores, "IoU", class_labels)

        # Add metrics as suptitle
        plt.suptitle(
            f'Loss: {loss:.4f} | Mean IoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f}\n'
            f'Boundary F1: {boundary_metrics["boundary_f1"]:.4f} | '
            f'Boundary Precision: {boundary_metrics["boundary_precision"]:.4f} | '
            f'Boundary Recall: {boundary_metrics["boundary_recall"]:.4f}\n'
            f'Mean Precision: {mean_precision:.4f} | '
            f'Mean Recall: {mean_recall:.4f}',
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'results/{RUN_NAME}/train_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')
        plt.close()
        print("Saving figure...")


def visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss, criterion):
    """Visualize training progress"""
    with torch.no_grad():
        pred = outputs[0].cpu()
        pred = torch.argmax(pred, dim=0).numpy()

        img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        mask_resized = mask_tensor[0].cpu().numpy()

        plt.close('all')

        # Calculate metrics
        results = calculate_confusion_metrics(N_CLASSES, outputs, mask_tensor)
        mean_iou = calculate_mean_iou(results).item()
        accuracy = calculate_accuracy(results).item()
        mean_precision = calculate_mean_precision(results).item()
        mean_recall = calculate_mean_recall(results).item()
        iou_scores = calculate_iou_per_class(results)

        class_labels = ['Year', 'Date', 'Latitude', 'Longitude', 'Temperature', 'Background'] 

        plt.figure(figsize=(20, 15))

        # Image visualization
        plt.subplot(2, 3, 1)
        plt.imshow(img_resized)
        plt.title('Preprocessed Image (Resized)')

        # Ground Truth Mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask_resized, cmap='tab10')
        plt.title('Ground Truth (Resized)')

        # Prediction Mask
        plt.subplot(2, 3, 3)
        plt.imshow(pred, cmap='tab10')
        plt.title(f'Prediction (Epoch {epoch+1})')

        # IoU Heatmap
        plt.subplot(2, 1, 2)
        plot_metric_heatmap(iou_scores, "IoU", class_labels)

        # Add overall metrics as a title
        plt.suptitle(
            f'Loss: {loss:.4f} | Mean IoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f} | '
            f'Mean Precision: {mean_precision:.4f} | Mean Recall: {mean_recall:.4f}',
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f'results/{RUN_NAME}/train_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')
        plt.close()

def train_epoch(epoch, model, dataloader, criterion, optimizer, scaler, device, num_classes, total_epochs):
    """
    Train the model for one epoch, dynamically adjusting loss weights using loss_weight_scheduler.
    
    Args:
        epoch: Current epoch number
        model: The segmentation model
        dataloader: Training data loader
        criterion: Loss function (WeightedSegmentationLoss)
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision training
        device: Computing device (e.g., 'cuda')
        num_classes: Number of classes in the segmentation task
        total_epochs: Total number of training epochs (used for dynamic loss weighting)
    
    Returns:
        epoch_losses: List of loss values for the epoch
        mean_ious, mean_dices, mean_pixel_accuracies: Metrics for the epoch
    """
    model.train()
    epoch_losses = []
    mean_ious = []
    mean_dices = []
    mean_pixel_accuracies = []
    # Get dynamic weights for Focal Loss and Dice Loss
    lambda_focal, lambda_dice = loss_weight_scheduler(epoch, total_epochs)

    with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
        for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
            with train_step_context():
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs = model(img_tensor)
                    
                    # Pass dynamic weights to the loss function
                    loss = criterion(outputs, mask_tensor)
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Log loss value
                loss_value = loss.item()
                epoch_losses.append(loss_value)

                # Calculate metrics
                pred = torch.argmax(outputs, dim=1)  # Convert logits to class predictions
                results = calculate_confusion_metrics(num_classes, pred, mask_tensor)

                mean_iou_value = calculate_mean_iou(results).item()
                mean_dice_value = calculate_mean_dice(results).item()
                mean_pixel_accuracy_value = calculate_mean_pixel_accuracy(results).item()

                mean_ious.append(mean_iou_value)
                mean_dices.append(mean_dice_value)
                mean_pixel_accuracies.append(mean_pixel_accuracy_value)

                boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, N_CLASSES)
                instance_sep_score = calculate_instance_separation(outputs, mask_tensor, N_CLASSES)
                # Update progress bar with metrics
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'mean_iou': f'{mean_iou_value:.4f}',
                    'boundary_f1': f'{boundary_metrics["boundary_f1"]:.4f}',
                    'instance_sep': f'{instance_sep_score:.4f}'
                })

                # Optional visualization
                if (epoch + 1) % 10 == 0 and batch_idx == 0:
                    visualize_batch_with_boundaries(epoch, img_tensor, mask_tensor, outputs, loss_value, criterion, 6)
                    # visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss_value, criterion)
                del outputs, loss

    return epoch_losses, {
        'mean_iou': sum(mean_ious) / (len(mean_ious)+ 1e-7),
        'mean_dice': sum(mean_dices) / (len(mean_dices)+ 1e-7),
        'mean_pixel_accuracy': sum(mean_pixel_accuracies) / (len(mean_pixel_accuracies)+ 1e-7),
    }

def main():
    torch.cuda.set_device(0)
    device = setup_device()
    torch.cuda.empty_cache()
    setup_environment()
    
    train_dataset = PirateLogDataset(
        img_dir='data/processed/train/images',
        mask_dir='data/processed/train/masks',
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES,
        transform=albumentations_transform
        
    )

    val_dataset = PirateLogDataset(
        img_dir='data/processed/val/images',
        mask_dir='data/processed/val/masks',
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


    results_dir = f"results/{RUN_NAME}"
    os.makedirs(results_dir, exist_ok=True)
    
    model = UNet(N_CLASSES).to(device)
    # model = SparseUNET(N_CLASSES).to(device)
    # model = ResNetUNET(N_CLASSES).to(device)
    
    # focal dice combo loss
    # criterion = WeightedSegmentationLoss(num_classes=N_CLASSES)
    
    # tversky loss
    # criterion = WeightedSegmentationLoss(
    #     num_classes=6, 
    #     use_tversky=True, 
    #     tversky_alpha=ALPHA, 
    #     tversky_beta=BETA, 
    #     from_logits=True
    # )

    criterion = BoundaryAwareTverskyLoss(
    num_classes=6,
    alpha=ALPHA,
    beta=BETA,
    boundary_weight=BOUNDARY_WEIGHT,
    from_logits=True
)
    
    # adamw optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-5
    )
   
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', 
    #     factor=0.9, 
    #     patience=25,
    #     threshold=0.01, 
    #     verbose=True
    # )
    
    scaler = GradScaler('cuda')
    
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_losses, metrics = train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, device, N_CLASSES, EPOCHS)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        train_losses.append(avg_loss)

        
        mean_iou_m = metrics['mean_iou']
        mean_dice_m = metrics['mean_dice']
        mean_pixel_accuracy_m = metrics['mean_pixel_accuracy']

        print(f"Epoch {epoch + 1} Metrics:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Mean IoU: {mean_iou_m:.4f}")
        print(f"  Mean Dice: {mean_dice_m:.4f}")
        print(f"  Mean Pixel Accuracy: {mean_pixel_accuracy_m:.4f}")

        val_loss, val_metrics = validate_epoch(epoch, model, val_loader, criterion, device, N_CLASSES, EPOCHS)
        val_losses.append(val_loss)
        val_accs.append(val_metrics['mean_iou'])

        print(f"Validation Metrics for Epoch {epoch + 1}:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Mean IoU: {val_metrics['mean_iou']:.4f}")
        print(f"  Mean Dice: {val_metrics['mean_dice']:.4f}")
        print(f"  Mean Pixel Accuracy: {val_metrics['mean_pixel_accuracy']:.4f}")

         # Step the scheduler
         #scheduler.step(val_loss)

    # Save model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'results/{RUN_NAME}/best_model.pth')
        print(f"Saved best model for epoch {epoch + 1}.")  
        


    # normalize train and val losses
    train_losses_norm = [loss / max(train_losses) for loss in train_losses]
    val_losses_norm = [loss / max(val_losses) for loss in val_losses]

    # Save training and validation losses
    epochs = list(range(1, len(train_losses) + 1))  # Create epoch numbers
    data = pd.DataFrame({
    "Epoch": epochs * 3,  # Repeated for three metrics
    "Value": train_losses_norm + val_losses_norm + val_accs,  # Concatenated values
    "Metric": (["Train Loss"] * len(epochs) + 
               ["Validation Loss"] * len(epochs) + 
               ["Validation Mean IOU"] * len(epochs))  # Concatenated labels
    })
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/{RUN_NAME}/training_metrics_norm_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png")
    plt.close()

    # Save training and validation losses
    epochs = list(range(1, len(train_losses) + 1))  # Create epoch numbers
    data = pd.DataFrame({
    "Epoch": epochs * 2,  # Repeated for three metrics
    "Value": train_losses + val_losses,  # Concatenated values
    "Metric": (["Train Loss"] * len(epochs) + 
               ["Validation Loss"] * len(epochs))  # Concatenated labels
    })
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/{RUN_NAME}/training_metrics_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png")
    plt.close()

     # Save training and validation losses
    epochs = list(range(1, len(train_losses) + 1))  # Create epoch numbers
    data = pd.DataFrame({
    "Epoch": epochs,  # Repeated for three metrics
    "Value": val_accs,  # Concatenated values
    "Metric": (["Validation Mean IOU"] * len(epochs))  # Concatenated labels
    })
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Mean IOU", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/{RUN_NAME}/Mean_IOU_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png")
    plt.close()

    

if __name__ == "__main__":
    main()