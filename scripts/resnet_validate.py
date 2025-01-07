import torch
from tqdm import tqdm
from metrics import (
    calculate_boundary_metrics,
    calculate_confusion_metrics,
    calculate_instance_separation,
    calculate_mean_iou,
    calculate_accuracy,
    calculate_mean_precision,
    calculate_mean_recall,
)
from visualization import visualize_batch_with_boundaries
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, N_CLASSES, TARGET_SIZE,
    VAL_IMG_DIR, VAL_MASK_DIR, DATALOADER_PARAMS,
    VAL_VIZ_FREQ, RESULTS_DIR
)
from dataset import PirateLogDataset
from torch.utils.data import DataLoader

def validate_epoch(epoch, model, dataloader, criterion, device, num_classes):
    """
    Validate the model for one epoch.
    
    This function performs a complete validation pass, computing various metrics
    including IoU, boundary accuracy, and instance separation. It also generates
    visualizations at specified intervals to track the model's performance.
    
    Args:
        epoch: Current epoch number
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Computing device (CPU/GPU)
        num_classes: Number of segmentation classes
    
    Returns:
        tuple: Average validation loss and dictionary of computed metrics
    """
    model.eval()
    epoch_losses = []
    mean_ious = []
    boundary_f1_scores = []
    instance_sep_scores = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Validating", unit="batch") as pbar:
            for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
                # Move data to appropriate device
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)

                # Get model predictions
                outputs = model(img_tensor)["out"]
                loss = criterion(outputs, mask_tensor)

                # Handle potential NaN losses
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue

                loss_value = loss.item()
                epoch_losses.append(loss_value)

                # Calculate various metrics
                pred = torch.argmax(outputs, dim=1)
                results = calculate_confusion_metrics(num_classes, pred, mask_tensor)
                mean_iou_value = calculate_mean_iou(results).item()
                mean_ious.append(mean_iou_value)

                # Calculate boundary and instance separation metrics
                boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, num_classes)
                instance_sep_score = calculate_instance_separation(outputs, mask_tensor, num_classes)

                boundary_f1_scores.append(boundary_metrics["boundary_f1"])
                instance_sep_scores.append(instance_sep_score)

                # Update progress bar with current metrics
                pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "mean_iou": f"{mean_iou_value:.4f}",
                    "boundary_f1": f'{boundary_metrics["boundary_f1"]:.4f}',
                    "instance_sep": f"{instance_sep_score:.4f}",
                })

                # Generate visualizations at specified frequency
                if (epoch + 1) % VAL_VIZ_FREQ == 0 and batch_idx == 0:
                    visualize_batch_with_boundaries(
                        epoch,
                        img_tensor,
                        mask_tensor,
                        outputs,
                        loss_value,
                        num_classes,
                        index=0,
                        training=False,
                    )

    # Compute average metrics
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    metrics = {
        "mean_iou": sum(mean_ious) / len(mean_ious),
        "boundary_f1": sum(boundary_f1_scores) / len(boundary_f1_scores),
        "instance_sep": sum(instance_sep_scores) / len(instance_sep_scores),
    }

    print(f"\nValidation Epoch {epoch + 1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Boundary F1: {metrics['boundary_f1']:.4f}")
    print(f"  Instance Separation: {metrics['instance_sep']:.4f}")

    return avg_loss, metrics

def setup_validation_data():
    """
    Set up the validation dataset and dataloader.
    
    This function creates the validation dataset and its corresponding dataloader
    using the configuration parameters defined in the config file.
    
    Returns:
        DataLoader: Configured validation data loader
    """
    val_dataset = PirateLogDataset(
        img_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES,
    )
    
    val_loader = DataLoader(
        val_dataset,
        **DATALOADER_PARAMS,
        shuffle=False  # No shuffling for validation
    )
    
    return val_loader

def validate_model(model, criterion, device):
    """
    Perform a complete validation of the model.
    
    This function runs the validation process for one complete pass through
    the validation dataset, computing and returning relevant metrics.
    
    Args:
        model: The neural network model
        criterion: Loss function
        device: Computing device (CPU/GPU)
        
    Returns:
        tuple: Validation loss and metrics for the entire validation set
    """
    val_loader = setup_validation_data()
    model.eval()
    
    # Run validation for one epoch
    val_loss, metrics = validate_epoch(
        0,  # epoch 0 since this is a single validation run
        model,
        val_loader,
        criterion,
        device,
        N_CLASSES
    )
    
    return val_loss, metrics