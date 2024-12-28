from loss import BoundaryAwareTverskyLoss
from dataset import PirateLogDataset
import torch
import os
import cv2
import pandas as pd
import seaborn as sns
from kornia.filters import sobel
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


from metrics import (
    calculate_boundary_metrics,
    calculate_instance_separation,
    calculate_iou_per_class,
    plot_metric_heatmap,
    calculate_confusion_metrics,
    calculate_mean_iou,
    calculate_mean_precision,
    calculate_mean_recall,
    calculate_accuracy,
)
from resnet_validate import validate_epoch

from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import gc
import signal
import sys
from contextlib import contextmanager
import albumentations as A

# HYPERPARAMS
BATCH_SIZE = 8
EPOCHS = 500
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200 // 2, 2496 // 2)
RUN_NAME = "deeplab_2"

# TVERSKY
ALPHA = 0.3
BETA = 0.7
BOUNDARY_WEIGHT = 20.0  # Increased from 10
VERTICAL_CONSISTENCY_WEIGHT = 3.0  # Increased from 1.0
SIGMA = 3.0  # Reduced from 5.0 for sharper boundaries

class_weights = [
    1.5,  # Year
    1.5,  # Date
    2.0,  # Latitude - increased
    1.5,  # Longitude
    1.5,  # Temperature
    2.5,  # Background - significantly increased
]
# Define Albumentations transforms
albumentations_transform = A.Compose(
    [
        # Rotation - low bcs text and column-lines are always very vertical
        A.Affine(
            rotate=(-2.5, 0),
            p=0.6,
            mode=cv2.BORDER_CONSTANT,
            cval=(255, 255, 255),
            interpolation=cv2.INTER_NEAREST,
        ),
        # Shear
        A.Affine(
            shear=(-1.5, 1.5),
            p=0.6,
            mode=cv2.BORDER_CONSTANT,
            cval=(255, 255, 255),
            interpolation=cv2.INTER_NEAREST,
        ),
        # Translation
        A.Affine(
            translate_percent={"x": (-0.025, 0.025)},
            p=0.6,
            mode=cv2.BORDER_CONSTANT,
            cval=(255, 255, 255),
            interpolation=cv2.INTER_NEAREST,
        ),
        # Elastic Transform
        A.ElasticTransform(
            alpha=25.0,
            sigma=10,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=1,
        ),
        # Grid Distortion
        A.GridDistortion(
            num_steps=3,
            distort_limit=0.2,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=1,
        ),
    ],
    additional_targets={"mask": "mask"},
)


def setup_device():
    try:
        if torch.cuda.is_available():
            free_memory = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            if free_memory < 1e9:
                raise RuntimeError("Insufficient GPU memory available")
            return torch.device("cuda")
        return torch.device("cpu")
    except RuntimeError as e:
        print(f"GPU error: {e}")
        return torch.device("cpu")


def setup_environment():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.75"
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


def visualize_batch_with_boundaries(
    epoch, img_tensor, mask_tensor, outputs, loss, criterion, N_CLASSES
):
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
        pred_onehot = (
            F.one_hot(pred_labels, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        )
        target_onehot = (
            F.one_hot(mask_tensor, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        )

        # Compute boundaries for visualization
        combined_pred_boundary = torch.zeros_like(pred_labels[0].float())
        combined_target_boundary = torch.zeros_like(mask_tensor[0].float())

        for c in range(N_CLASSES):
            pred_edges = torch.abs(sobel(pred_onehot[:, c : c + 1]))
            target_edges = torch.abs(sobel(target_onehot[:, c : c + 1]))

            combined_pred_boundary += (pred_edges[0, 0] > 0.5).float()
            combined_target_boundary += (target_edges[0, 0] > 0.5).float()

        # Create visualization
        plt.figure(figsize=(20, 15))

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img_resized)
        plt.title("Preprocessed Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask_resized, cmap="tab10")
        plt.title("Ground Truth")
        plt.axis("off")

        # Prediction mask
        plt.subplot(2, 3, 3)
        plt.imshow(pred, cmap="tab10")
        plt.title(f"Prediction (Epoch {epoch+1})")
        plt.axis("off")

        # Predicted boundaries
        plt.subplot(2, 3, 4)
        plt.imshow(combined_pred_boundary.cpu().numpy(), cmap="gray")
        plt.title("Predicted Boundaries")
        plt.axis("off")

        # Ground truth boundaries
        plt.subplot(2, 3, 5)
        plt.imshow(combined_target_boundary.cpu().numpy(), cmap="gray")
        plt.title("Ground Truth Boundaries")
        plt.axis("off")

        # IoU Heatmap
        plt.subplot(2, 3, 6)
        iou_scores = calculate_iou_per_class(results)
        class_labels = [
            "Year",
            "Date",
            "Latitude",
            "Longitude",
            "Temperature",
            "Background",
        ]
        plot_metric_heatmap(iou_scores, "IoU", class_labels)

        # Add metrics as suptitle
        plt.suptitle(
            f'Loss: {loss:.4f} | Mean IoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f}\n'
            f'Boundary F1: {boundary_metrics["boundary_f1"]:.4f} | '
            f'Boundary Precision: {boundary_metrics["boundary_precision"]:.4f} | '
            f'Boundary Recall: {boundary_metrics["boundary_recall"]:.4f}\n'
            f'Mean Precision: {mean_precision:.4f} | '
            f'Mean Recall: {mean_recall:.4f}',
            fontsize=16,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            f"results/{RUN_NAME}/train_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png"
        )
        plt.close()
        print("Saving figure...")


def visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss, criterion):
    """Visualize training progress"""
    with torch.no_grad():
        pred = outputs[0].cpu()
        pred = torch.argmax(pred, dim=0).numpy()

        img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        mask_resized = mask_tensor[0].cpu().numpy()

        plt.close("all")

        # Calculate metrics
        results = calculate_confusion_metrics(N_CLASSES, outputs, mask_tensor)
        mean_iou = calculate_mean_iou(results).item()
        accuracy = calculate_accuracy(results).item()
        mean_precision = calculate_mean_precision(results).item()
        mean_recall = calculate_mean_recall(results).item()
        iou_scores = calculate_iou_per_class(results)

        class_labels = [
            "Year",
            "Date",
            "Latitude",
            "Longitude",
            "Temperature",
            "Background",
        ]

        plt.figure(figsize=(20, 15))

        # Image visualization
        plt.subplot(2, 3, 1)
        plt.imshow(img_resized)
        plt.title("Preprocessed Image (Resized)")

        # Ground Truth Mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask_resized, cmap="tab10")
        plt.title("Ground Truth (Resized)")

        # Prediction Mask
        plt.subplot(2, 3, 3)
        plt.imshow(pred, cmap="tab10")
        plt.title(f"Prediction (Epoch {epoch+1})")

        # IoU Heatmap
        plt.subplot(2, 1, 2)
        plot_metric_heatmap(iou_scores, "IoU", class_labels)

        # Add overall metrics as a title
        plt.suptitle(
            f"Loss: {loss:.4f} | Mean IoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f} | "
            f"Mean Precision: {mean_precision:.4f} | Mean Recall: {mean_recall:.4f}",
            fontsize=16,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(
            f"results/{RUN_NAME}/train_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png"
        )
        plt.close()


def train_epoch(
    epoch, model, dataloader, criterion, optimizer, scaler, device, num_classes
):
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
    boundary_f1_scores = []
    instance_sep_scores = []

    with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar:
        for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
            with train_step_context():
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)

                with autocast("cuda"):
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
                mean_ious.append(mean_iou_value)

                boundary_metrics = calculate_boundary_metrics(
                    outputs, mask_tensor, N_CLASSES
                )
                instance_sep_score = calculate_instance_separation(
                    outputs, mask_tensor, N_CLASSES
                )

                boundary_f1_scores.append(boundary_metrics["boundary_f1"])
                instance_sep_scores.append(instance_sep_score)

                pbar.set_postfix(
                    {
                        "loss": f"{loss_value:.4f}",
                        "mean_iou": f"{mean_iou_value:.4f}",
                        "boundary_f1": f'{boundary_metrics["boundary_f1"]:.4f}',
                        "instance_sep": f"{instance_sep_score:.4f}",
                    }
                )

                if (epoch + 1) % 5 == 0 and batch_idx == 0:
                    visualize_batch_with_boundaries(
                        epoch,
                        img_tensor,
                        mask_tensor,
                        outputs,
                        loss_value,
                        criterion,
                        6,
                    )
                del outputs, loss

    return epoch_losses, {
        "mean_iou": sum(mean_ious) / (len(mean_ious) + 1e-7),
        "boundary_f1": sum(boundary_f1_scores) / (len(boundary_f1_scores) + 1e-7),
        "instance_sep": sum(instance_sep_scores) / (len(instance_sep_scores) + 1e-7),
    }


def main():
    torch.cuda.set_device(0)
    device = setup_device()
    torch.cuda.empty_cache()
    setup_environment()

    train_dataset = PirateLogDataset(
        img_dir="data/processed/train/images",
        mask_dir="data/processed/train/masks",
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES,
        transform=albumentations_transform,
    )

    val_dataset = PirateLogDataset(
        img_dir="data/processed/val/images",
        mask_dir="data/processed/val/masks",
        target_size=TARGET_SIZE,
        num_classes=N_CLASSES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    results_dir = f"results/{RUN_NAME}"
    os.makedirs(results_dir, exist_ok=True)

    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = nn.Conv2d(256, N_CLASSES, kernel_size=1)
    model = model.to(device)

    criterion = BoundaryAwareTverskyLoss(
        num_classes=N_CLASSES,
        alpha=ALPHA,
        beta=BETA,
        boundary_weight=BOUNDARY_WEIGHT,
        vertical_consistency_weight=VERTICAL_CONSISTENCY_WEIGHT,
        from_logits=True,
        class_weights=class_weights,
        sigma=SIGMA,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )

    scaler = GradScaler("cuda")

    train_losses = []
    val_losses = []
    val_accs = []
    boundary_f1s = []
    instance_seps = []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        epoch_losses, metrics = train_epoch(
            epoch, model, train_loader, criterion, optimizer, scaler, device, N_CLASSES
        )
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        train_losses.append(avg_loss)
        mean_iou_m = metrics["mean_iou"]
        boundary_f1s.append(metrics["boundary_f1"])
        instance_seps.append(metrics["instance_sep"])

        print(f"Epoch {epoch + 1} Metrics:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Mean IoU: {mean_iou_m:.4f}")
        print(f"  Boundary F1: {metrics['boundary_f1']:.4f}")
        print(f"  Instance Separation: {metrics['instance_sep']:.4f}")

        val_loss, val_metrics = validate_epoch(
            epoch, model, val_loader, criterion, device, N_CLASSES
        )
        val_losses.append(val_loss)
        val_accs.append(val_metrics["mean_iou"])

        print(f"Validation Metrics for Epoch {epoch + 1}:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Mean IoU: {val_metrics['mean_iou']:.4f}")
        print(f"  Boundary F1: {val_metrics['boundary_f1']:.4f}")
        print(f"  Instance Separation: {val_metrics['instance_sep']:.4f}")

    # Save model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"results/{RUN_NAME}/best_model.pth")
        print(f"Saved best model for epoch {epoch + 1}.")

    # normalize train and val losses
    train_losses_norm = [loss / max(train_losses) for loss in train_losses]
    val_losses_norm = [loss / max(val_losses) for loss in val_losses]

    # Save training and validation losses
    epochs = list(range(1, len(train_losses) + 1))  # Create epoch numbers
    data = pd.DataFrame(
        {
            "Epoch": epochs * 3,  # Repeated for three metrics
            "Value": train_losses_norm
            + val_losses_norm
            + val_accs,  # Concatenated values
            "Metric": (
                ["Train Loss"] * len(epochs)
                + ["Validation Loss"] * len(epochs)
                + ["Validation Mean IOU"] * len(epochs)
            ),  # Concatenated labels
        }
    )

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        f"results/{RUN_NAME}/training_metrics_norm_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png"
    )
    plt.close()

    # Save training and validation losses
    epochs = list(range(1, len(train_losses) + 1))  # Create epoch numbers
    data = pd.DataFrame(
        {
            "Epoch": epochs * 2,  # Repeated for three metrics
            "Value": train_losses + val_losses,  # Concatenated values
            "Metric": (
                ["Train Loss"] * len(epochs) + ["Validation Loss"] * len(epochs)
            ),  # Concatenated labels
        }
    )

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        f"results/{RUN_NAME}/training_metrics_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png"
    )
    plt.close()

    # Save training and validation losses
    epochs = list(range(1, len(train_losses) + 1))  # Create epoch numbers
    data = pd.DataFrame(
        {
            "Epoch": epochs,  # Repeated for three metrics
            "Value": val_accs,  # Concatenated values
            "Metric": (["Validation Mean IOU"] * len(epochs)),  # Concatenated labels
        }
    )

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Mean IOU", fontsize=14)
    plt.title("Training and Validation Metrics", fontsize=16, fontweight="bold")
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/{RUN_NAME}/Mean_IOU_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png")
    plt.close()

    # boundaries
    epochs = list(range(1, len(boundary_f1s) + 1))
    data = pd.DataFrame(
        {
            "Epoch": epochs * 2,
            "Value": boundary_f1s + instance_seps,
            "Metric": (
                ["Boundary F1"] * len(epochs) + ["Instance Separation"] * len(epochs)
            ),
        }
    )

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=data, x="Epoch", y="Value", hue="Metric", marker="o", linewidth=2.5
    )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title(
        "Boundary and Instance Separation Metrics", fontsize=16, fontweight="bold"
    )
    plt.legend(title="Metric", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        f"results/{RUN_NAME}/boundary_metrics_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png"
    )
    plt.close()


if __name__ == "__main__":
    main()
