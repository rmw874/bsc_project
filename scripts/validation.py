import torch
from tqdm import tqdm # For progress bar
from metrics import calculate_tp_fp_fn, calculate_mean_iou, calculate_mean_dice, calculate_mean_pixel_accuracy, plot_metric_heatmap, calculate_dice_per_class, calculate_iou_per_class, calculate_accuracy, calculate_mean_precision, calculate_mean_recall, calculate_mean_iou
from loss import loss_weight_scheduler
import cv2
import matplotlib.pyplot as plt

# HYPERPARAMS
BATCH_SIZE = 2
EPOCHS = 200
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200//2, 2496//2)
RUN_NAME = 'run29_sprse16_tversky_moredata'


# def visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss, criterion, index=0):
#     """Visualize training progress"""
#     with torch.no_grad():
#         pred = outputs[index].cpu()
#         pred = torch.argmax(pred, dim=0).numpy()

#         img_resized = img_tensor[index].permute(1, 2, 0).cpu().numpy()
#         mask_resized = mask_tensor[index].cpu().numpy()

#         ### HEATMAP
#         result = calculate_tp_fp_fn(N_CLASSES, outputs, mask_tensor)
#         dice_scores = calculate_dice_per_class(result)
#         iou_scores = calculate_iou_per_class(result)
#         class_labels = ['Year', 'Date', 'Longitude', 'Latitude','Temperature','Background']

#         plt.figure(figsize=(20, 10))  

#         plt.subplot(2, 2, 1)
#         plt.imshow(img_resized)
#         plt.title('Preprocessed Image (Resized)')

#         plt.subplot(2, 2, 2)
#         plot_metric_heatmap(iou_scores, "IoU", class_labels)
        
#         plt.subplot(2, 2, 3)
#         plt.imshow(mask_resized, cmap='tab10')
#         plt.title('Ground Truth (Resized)')
        
#         plt.subplot(2, 2, 4)
#         plt.imshow(pred, cmap='tab10')
#         plt.title(f'Prediction (Epoch {epoch+1})')
        
#         plt.suptitle(f'Loss: {loss:.4f}', fontsize=24)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.savefig(f'results/{RUN_NAME}/val_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png')
#         plt.close()

def visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss, criterion):
    """Visualize training progress"""
    with torch.no_grad():
        pred = outputs[0].cpu()
        pred = torch.argmax(pred, dim=0).numpy()

        img_resized = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        mask_resized = mask_tensor[0].cpu().numpy()

        # Calculate metrics
        results = calculate_tp_fp_fn(N_CLASSES, outputs, mask_tensor)
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


def validate_epoch(epoch, model, dataloader, criterion, device, num_classes, total_epochs):
    # Get dynamic weights for Focal Loss and Dice Loss
    lambda_focal, lambda_dice = loss_weight_scheduler(epoch, total_epochs)

    model.eval()
    epoch_losses = []
    mean_ious = []
    mean_dices = []
    mean_pixel_accuracies = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Validating", unit="batch") as pbar:
            for batch_idx, (img_tensor, mask_tensor) in enumerate(pbar):
                img_tensor = img_tensor.to(device, non_blocking=True)
                mask_tensor = mask_tensor.to(device, non_blocking=True)
                
                outputs = model(img_tensor)
                loss = criterion(outputs, mask_tensor, lambda_focal=lambda_focal, lambda_dice=lambda_dice)
                
                # Log loss value
                loss_value = loss.item()
                epoch_losses.append(loss_value)

                pred = torch.argmax(outputs, dim=1)  # Convert logits to class predictions
                results = calculate_tp_fp_fn(num_classes, pred, mask_tensor)

                mean_iou_value = calculate_mean_iou(results).item()
                mean_dice_value = calculate_mean_dice(results).item()
                mean_pixel_accuracy_value = calculate_mean_pixel_accuracy(results).item()

                mean_ious.append(mean_iou_value)
                mean_dices.append(mean_dice_value)
                mean_pixel_accuracies.append(mean_pixel_accuracy_value)

                pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'mean_iou': f'{mean_iou_value:.4f}',
                    'mean_dice': f'{mean_dice_value:.4f}',
                    'mean_pixel_acc': f'{mean_pixel_accuracy_value:.4f}',
                })

                # Optional visualization: Choose a specific batch to visualize
                if (epoch + 1) % 5 == 0 and batch_idx == 0:  # Visualize the second batch
                    index_to_visualize = 0  # Or any index within the batch
                    visualize_batch(epoch, img_tensor, mask_tensor, outputs, loss_value, criterion, index=index_to_visualize)

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    metrics = {
        'mean_iou': sum(mean_ious) / len(mean_ious),
        'mean_dice': sum(mean_dices) / len(mean_dices),
        'mean_pixel_accuracy': sum(mean_pixel_accuracies) / len(mean_pixel_accuracies),
    }

    return avg_loss, metrics