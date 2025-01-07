import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import preprocess
from post_process import remove_small_regions, separate_tall_regions, erode_regions, create_bounding_boxes

def load_test_images(bottom_dir, top_dir, num_bottom=5, num_top=5):
    """Load test images from bottom and top page directories"""
    bottom_files = [f for f in os.listdir(bottom_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    top_files = [f for f in os.listdir(top_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(bottom_files)
    random.shuffle(top_files)

    bottom_chosen = [os.path.join(bottom_dir, f) for f in bottom_files[:num_bottom]]
    top_chosen = [os.path.join(top_dir, f) for f in top_files[:num_top]]

    return bottom_chosen + top_chosen + ["data/mgr_testers/8013620831-0070.jpg-t.jpg"]

def post_process_mask(logits):
    """Apply the complete post-processing pipeline to model outputs"""
    pred_classes = torch.argmax(logits, dim=1)  
    
    if pred_classes.dim() == 3:
        processed_list = []
        for i in range(pred_classes.shape[0]):
            single = pred_classes[i]
            cleaned = remove_small_regions(single, min_region_size=3000)
            separated = separate_tall_regions(cleaned, expected_row_height=50)
            eroded = erode_regions(separated)
            final = create_bounding_boxes(eroded)
            processed_list.append(final)
        return torch.stack(processed_list)
    else:
        cleaned = remove_small_regions(pred_classes, min_region_size=3000)
        separated = separate_tall_regions(cleaned, expected_row_height=50)
        eroded = erode_regions(separated)
        final = create_bounding_boxes(eroded)
        return final

def extract_row_col_bboxes(final_mask, num_classes=6, row_threshold=30):
    """
    Extract bounding boxes and organize them by row and column position using y-centers.
    
    Args:
        final_mask: Segmentation mask
        num_classes: Number of classes in the segmentation
        row_threshold: Minimum vertical distance to consider as a new row
        
    Returns:
        list of tuples: (row_num, col_num, x1, y1, x2, y2)
    """
    from skimage.measure import label, regionprops

    if isinstance(final_mask, torch.Tensor):
        final_mask = final_mask.cpu().numpy()

    # Store regions with their positions and centers
    regions = []
    
    # Process each class (column)
    for class_id in range(5):  # Excluding background class
        mask = (final_mask == class_id).astype(np.uint8)
        labeled = label(mask)
        
        for region in regionprops(labeled):
            y_min, x_min, y_max, x_max = region.bbox
            y_center = (y_max + y_min) / 2  # Calculate y-center
            
            # Store y_center for sorting and row assignment
            regions.append((y_center, class_id, x_min, y_min, x_max, y_max))
    
    # Sort regions by y-center position
    regions.sort()
    
    # Assign row numbers based on y-center clustering
    row_col_boxes = []
    current_row = 0
    if regions:
        # Initialize with first region's y-center
        current_y_center = regions[0][0]
        
        for y_center, col_num, x_min, y_min, x_max, y_max in regions:
            # If there's a significant gap between y-centers, increment row number
            if abs(y_center - current_y_center) > row_threshold:
                current_row += 1
                current_y_center = y_center
                
            row_col_boxes.append((current_row, col_num, x_min, y_min, x_max, y_max))
    
    # Sort final results by row then column for consistent ordering
    row_col_boxes.sort(key=lambda x: (x[0], x[1]))
    
    return row_col_boxes

def save_row_col_crops(original_pil, bboxes, out_dir, base_filename):
    """
    Save crops using row_column naming scheme:
    out_dir/base_filename/row_X_column_Y.png
    """
    photo_folder = os.path.join(out_dir, base_filename)
    os.makedirs(photo_folder, exist_ok=True)

    for row_num, col_num, x1, y1, x2, y2 in bboxes:
        crop = original_pil.crop((x1, y1, x2, y2))
        crop_path = os.path.join(photo_folder, f"row_{row_num}_column_{col_num}.png")
        crop.save(crop_path)
        print(f"Saved: {crop_path}")

def visualize_row_col_boxes(image_path, bboxes, out_path):
    """Draw bounding boxes with row-column labels on the image"""
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print(f"Could not open {image_path}")
        return
    
    for row_num, col_num, x1, y1, x2, y2 in bboxes:
        color = colors[(col_num - 1) % len(colors)]
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, 2)
        label = f"R{row_num}C{col_num}"
        cv2.putText(bgr_img, label, (x1, max(y1-5, 0)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.imwrite(out_path, bgr_img)
    print(f"Saved visualization: {out_path}")

def main():
    random.seed(42)
    
    # Configure paths
    # bottom_dir = "data/processed/Mathiesen-single-pages/testing/bottom"
    # top_dir = "data/processed/Mathiesen-single-pages/testing/top"
    bottom_dir = "data/processed/Mathiesen-single-pages/temp"
    top_dir = "data/processed/Mathiesen-single-pages/temp"
    out_dir = "results/vanilla-row_col_crops"
    os.makedirs(out_dir, exist_ok=True)

    # Load and prepare model
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model.load_state_dict(torch.load('results/resnet_tversky_vanilla_lower_batch/best_model.pth'))
    model.cuda()
    model.eval()

    # Process test images
    test_paths = load_test_images(bottom_dir, top_dir)
    
    for img_path in test_paths:
        print(f"\nProcessing: {img_path}")
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load and preprocess image
        original_pil = Image.open(img_path).convert('RGB')
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            continue
            
        preprocessed = preprocess(bgr_img, target_size=(1600, 1248))
        h_orig, w_orig = bgr_img.shape[:2]
        h_resized, w_resized = preprocessed.shape[:2]

        # Prepare tensor for model
        img_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # Run inference and post-processing
        with torch.no_grad():
            logits = model(img_tensor)['out']
        final_mask = post_process_mask(logits)
        
        if final_mask.dim() == 3:
            final_mask = final_mask[0]

        # Extract and scale bounding boxes
        bboxes = extract_row_col_bboxes(final_mask)
        scaled_bboxes = []
        for row_num, col_num, x1, y1, x2, y2 in bboxes:
            x1_orig = int(x1 * (w_orig / w_resized))
            y1_orig = int(y1 * (h_orig / h_resized))
            x2_orig = int(x2 * (w_orig / w_resized))
            y2_orig = int(y2 * (h_orig / h_resized))
            scaled_bboxes.append((row_num, col_num, x1_orig, y1_orig, x2_orig, y2_orig))

        # Save visualization and crops
        debug_vis_path = os.path.join(out_dir, f"{base_filename}_debug.png")
        visualize_row_col_boxes(img_path, scaled_bboxes, debug_vis_path)
        save_row_col_crops(original_pil, scaled_bboxes, out_dir, base_filename)

if __name__ == "__main__":
    main()