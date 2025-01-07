import torch
from tqdm import tqdm
from metrics import (
    calculate_boundary_metrics,
    calculate_confusion_metrics,
    calculate_instance_separation,
    plot_metric_heatmap,
    calculate_iou_per_class,
    calculate_accuracy,
    calculate_mean_precision,
    calculate_mean_recall,
    calculate_mean_iou,
)
from scripts.post_process import apply_post_processing
import matplotlib.pyplot as plt
from kornia.filters import sobel
import torch.nn.functional as F

# HYPERPARAMS
BATCH_SIZE = 8
EPOCHS = 500
LEARNING_RATE = 2e-4
N_CLASSES = 6
TARGET_SIZE = (3200 // 2, 2496 // 2)
RUN_NAME = "deeplab_2"


def visualize_batch_with_boundaries(
    epoch, img_tensor, mask_tensor, outputs, loss, N_CLASSES, index=0, post_process=False
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
        index: number of image to visualize
        post_process: If True, apply post-processing to outputs before visualization
    """
    if post_process:
        outputs = apply_post_processing(outputs)
    with torch.no_grad():
        # Get predictions
        pred = outputs[index].cpu()
        pred = torch.argmax(pred, dim=0).numpy()

        img_resized = img_tensor[index].permute(1, 2, 0).cpu().numpy()
        mask_resized = mask_tensor[index].cpu().numpy()

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
            f"results/{RUN_NAME}/validation_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png"
        )
        plt.close()
        print("Saving figure...")


def validate_epoch(epoch, model, dataloader, criterion, device, num_classes):
    model.eval()
    epoch_losses = []
    mean_ious = []
    boundary_f1_scores = []  # New
    instance_sep_scores = []  # New

    with torch.no_grad():
        with tqdm(dataloader, desc="Validating", unit="batch") as pbar:
            for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)

                outputs = model(img_tensor)["out"]
                loss = criterion(outputs, mask_tensor)
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue

                print(f"Validation loss for batch {batch_idx}: {loss.item()}")

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

                # Store new metrics
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

                if (epoch + 1) % 10 == 0 and batch_idx == 0:
                    index_to_visualize = 0
                    visualize_batch_with_boundaries(
                        epoch,
                        img_tensor,
                        mask_tensor,
                        outputs,
                        loss_value,
                        6,
                        index=index_to_visualize,
                    )

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    metrics = {
        "mean_iou": sum(mean_ious) / len(mean_ious),
        "boundary_f1": sum(boundary_f1_scores) / len(boundary_f1_scores),  # New
        "instance_sep": sum(instance_sep_scores) / len(instance_sep_scores),  # New
    }

    return avg_loss, metrics
