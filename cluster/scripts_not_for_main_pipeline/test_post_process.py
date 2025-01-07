import torch
import matplotlib.pyplot as plt
from post_process import post_process_predictions, remove_small_regions, create_bounding_boxes, separate_tall_regions, erode_regions
from dataset import PirateLogDataset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def load_model_and_data():
    """Load pretrained model and a sample from the validation dataset"""
    # Initialize model same way as training
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model = model.to('cuda')
    
    # Load the trained weights
    model.load_state_dict(torch.load('results/final_model_maybe/best_model.pth'))
    model.eval()
    
    # Load a validation sample
    val_dataset = PirateLogDataset(
        img_dir="data/processed/val/images",
        mask_dir="data/processed/val/masks",
        target_size=(3200 // 2, 2496 // 2),
        num_classes=6
    )
    
    return model, val_dataset

def visualize_processing_steps(image, gt_mask, logits):
    """Visualize each step of the post-processing pipeline"""
    # Move tensors to CPU
    image = image.cpu()
    gt_mask = gt_mask.cpu()
    logits = logits.cpu()
    
    # Get original prediction
    pred_classes = torch.argmax(logits, dim=1)[0]
    
    # Apply post-processing steps
    cleaned = remove_small_regions(pred_classes)
    separated = separate_tall_regions(cleaned)
    eroded = erode_regions(separated)
    final = create_bounding_boxes(eroded)
    
    # Create visualization
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(3, 2, 1)
    plt.imshow(image[0].permute(1, 2, 0))
    plt.title("Original Image")
    plt.axis('off')
    
    # Ground truth
    plt.subplot(3, 2, 2)
    plt.imshow(gt_mask[0], cmap='tab10')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # After small region removal
    plt.subplot(3, 2, 3)
    plt.imshow(cleaned, cmap='tab10')
    plt.title("After Small Region Removal")
    plt.axis('off')
    
    # After row separation
    plt.subplot(3, 2, 4)
    plt.imshow(separated, cmap='tab10')
    plt.title("After Row Separation")
    plt.axis('off')
    
    # After erosion
    plt.subplot(3, 2, 5)
    plt.imshow(eroded, cmap='tab10')
    plt.title("After Erosion")
    plt.axis('off')
    
    # Final result with bounding boxes
    plt.subplot(3, 2, 6)
    plt.imshow(final, cmap='tab10')
    plt.title("Final Post-processed")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('post_processing_test_final.png')
    plt.close()

def main():
    # Load model and data
    model, dataset = load_model_and_data()
    
    # Get a sample
    image, gt_mask = dataset[2]
    image = image.unsqueeze(0).cuda()
    
    # Get model prediction
    with torch.no_grad():
        logits = model(image)['out']
    
    # Visualize results
    visualize_processing_steps(image, gt_mask.unsqueeze(0), logits)
    
    # Print some statistics
    pred_classes = torch.argmax(logits, dim=1)[0].cpu()
    final = post_process_predictions(logits.cpu())
    
    # Calculate number of changed pixels
    changes = (final[0] != pred_classes).sum().item()
    total_pixels = pred_classes.numel()
    
    print(f"Post-processing modified {changes} pixels ({(changes/total_pixels)*100:.2f}% of image)")
    print("Results saved as post_processing_test.png")

if __name__ == "__main__":
    main()