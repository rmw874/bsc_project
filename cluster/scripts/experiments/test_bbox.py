import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50

# Import your custom dataset code if needed
# from dataset import PirateLogDataset  # If you want to reuse your dataset logic
from dataset import preprocess  # We'll use your "preprocess()" function directly

# Import your post-processing functions
from post_process import (
    remove_small_regions,
    separate_tall_regions,
    erode_regions,
    create_bounding_boxes
)

###################################
# 1. HELPER FUNCTIONS
###################################

def load_images_from_dirs(bottom_dir, top_dir, num_bottom=5, num_top=5):
    """
    Return a list of 'num_bottom' images from bottom_dir
    plus 'num_top' images from top_dir.
    """
    bottom_files = [
        os.path.join(bottom_dir, f)
        for f in os.listdir(bottom_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    top_files = [
        os.path.join(top_dir, f)
        for f in os.listdir(top_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    random.shuffle(bottom_files)
    random.shuffle(top_files)

    bottom_chosen = bottom_files[:num_bottom]
    top_chosen = top_files[:num_top]

    return bottom_chosen + top_chosen


def post_process_mask(logits):
    """
    Apply your pipeline:
      - Argmax
      - remove_small_regions (lower min_region_size if needed)
      - separate_tall_regions
      - erode_regions
      - create_bounding_boxes
    """
    # 1) Argmax to get predicted classes
    pred_classes = torch.argmax(logits, dim=1)  # shape: [B, H, W] if B>1

    # If batch dimension present:
    if pred_classes.dim() == 3:
        processed_list = []
        for i in range(pred_classes.shape[0]):
            single = pred_classes[i]
            cleaned = remove_small_regions(single, min_region_size=3000)
            separated = separate_tall_regions(cleaned, expected_row_height=60)
            eroded = erode_regions(separated)
            final = create_bounding_boxes(eroded)
            processed_list.append(final)
        return torch.stack(processed_list)
    else:
        # Single image
        cleaned = remove_small_regions(pred_classes, min_region_size=3000)
        separated = separate_tall_regions(cleaned, expected_row_height=60)
        eroded = erode_regions(separated)
        final = create_bounding_boxes(eroded)
    return final


def extract_bboxes_from_final(final_mask, num_classes=6):
    """
    Use regionprops to find bounding boxes for each class (0..4 if 5=background).
    Returns a list of tuples: (class_id, x1, y1, x2, y2).
    """
    from skimage.measure import label, regionprops

    if isinstance(final_mask, torch.Tensor):
        final_mask = final_mask.cpu().numpy()

    bboxes = []
    # We skip background == 5. Adjust if your background is 0 or something else.
    # According to your code, classes are 0..4, background=5.
    for class_id in range(0, 5):
        mask = (final_mask == class_id).astype(np.uint8)
        if mask.sum() < 1:
            continue

        labeled = label(mask)
        for region in regionprops(labeled):
            # region.bbox = (y_min, x_min, y_max, x_max)
            y_min, x_min, y_max, x_max = region.bbox

            # If region is extremely small, skip it:
            # (this could also be handled with remove_small_regions)
            if region.area < 20:
                continue

            bboxes.append((class_id, x_min, y_min, x_max, y_max))

    return bboxes

def save_pred_mask(mask_tensor, out_path="pred_mask_debug.png"):
    """
    Save the predicted mask (H,W) as a pseudo-color image.
    """
    import matplotlib.pyplot as plt
    
    mask_np = mask_tensor.cpu().numpy()
    # If shape is (1, H, W), remove dimension:
    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_np, cmap='tab10')
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()


def visualize_bboxes_on_image(image_path, bboxes, out_path="debug_bboxes.png"):
    """
    Draw bounding boxes on the original image (using OpenCV) in different colors by class.
    """
    # Colors for each class_id, up to 6. Adjust or add more if you have more classes
    class_colors = {
        0: (255, 0, 0),    # Blue
        1: (0, 255, 0),    # Green
        2: (0, 0, 255),    # Red
        3: (255, 255, 0),  # Cyan
        4: (255, 0, 255),  # Magenta
        5: (0, 255, 255)   # Yellow
    }
    
    # Read the original image in BGR
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print(f"Could not open {image_path} for visualization.")
        return
    
    for (class_id, x1, y1, x2, y2) in bboxes:
        color = class_colors.get(class_id, (255, 255, 255))  # default white
        # Draw rectangle (thickness=2)
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, 2)
        # Put class label text
        cv2.putText(bgr_img, f"Class {class_id}", (x1, max(y1-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.imwrite(out_path, bgr_img)
    print(f"Debug bounding boxes saved: {out_path}")




def save_crops(original_pil, bboxes, out_dir, base_filename):
    """
    We want: cropped_bboxes/<base_filename>/class_<class_id>/crop_i.png
    so each photo has its own folder, then classes inside.
    """
    # e.g. out_dir = "cropped_bboxes"
    photo_folder = os.path.join(out_dir, base_filename)
    os.makedirs(photo_folder, exist_ok=True)

    for i, (class_id, x1, y1, x2, y2) in enumerate(bboxes):
        class_folder = os.path.join(photo_folder, f"class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)

        crop = original_pil.crop((x1, y1, x2, y2))
        crop_path = os.path.join(class_folder, f"crop_{i}.png")
        crop.save(crop_path)
        print(f"Saved crop: {crop_path}")


###################################
# 2. MAIN INFERENCE SCRIPT
###################################

def main():
    random.seed(42)

    # 1) Pick test images
    bottom_dir = "/home/rmw874/piratbog/data/processed/Mathiesen-single-pages/testing/bottom"
    top_dir = "/home/rmw874/piratbog/data/processed/Mathiesen-single-pages/testing/top"
    #bottom_dir ="/home/rmw874/piratbog/data/raw/Mathiesen-single-pages"
    #top_dir="/home/rmw874/piratbog/data/raw/Mathiesen-single-pages"
    test_paths = load_images_from_dirs(bottom_dir, top_dir, num_bottom=5, num_top=5)

    # 2) Load model
    num_classes = 6
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.load_state_dict(torch.load('results/deeplab_5_bugged/best_model.pth'))
    model.cuda()
    model.eval()

    # 3) Output directory
    out_dir = "results/cropped_bboxes"
    os.makedirs(out_dir, exist_ok=True)

    for img_path in test_paths:
        print(f"\nProcessing: {img_path}")

        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Read original for final cropping
        original_pil = Image.open(img_path).convert('RGB')

        # 4) Preprocess (the same as in PirateLogDataset)
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            print(f"Could not read {img_path}. Skipping.")
            continue

        # Use your 'preprocess()' function from dataset.py
        # Example target size from your training script: (1600, 1248)
        preprocessed = preprocess(bgr_img, target_size=(1600, 1248))
        preprocessed = preprocess(bgr_img, target_size=(1600, 1248))
        h_orig, w_orig = bgr_img.shape[:2]
        h_resized, w_resized = preprocessed.shape[:2]

        # Convert to Tensor, scale to [0..1], move to GPU
        img_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # 5) Run inference
        with torch.no_grad():
            logits = model(img_tensor)['out']  # shape: [1, num_classes, H, W]

        # 6) Post-process to get final labeled mask
        final_mask = post_process_mask(logits)  # shape: [1, H, W] or just (H, W)
        #save_pred_mask(final_mask, f"{base_filename}_debug_mask.png")


        # If batch dimension is present, do final_mask = final_mask[0]
        if final_mask.dim() == 3:
            final_mask = final_mask[0]

        # 7) Extract bounding boxes
        bboxes_resized = extract_bboxes_from_final(final_mask, num_classes=6)

        # Scale bounding boxes back to original size
        scaled_bboxes = []
        for (class_id, x1, y1, x2, y2) in bboxes_resized:
            x1_orig = int(x1 * (w_orig / w_resized))
            y1_orig = int(y1 * (h_orig / h_resized))
            x2_orig = int(x2 * (w_orig / w_resized))
            y2_orig = int(y2 * (h_orig / h_resized))

            scaled_bboxes.append((class_id, x1_orig, y1_orig, x2_orig, y2_orig))

        # Debug: Visualize bounding boxes on the original
        debug_bbox_path = os.path.join(out_dir, f"{base_filename}_bboxes_dbg.png")
        # Now visualize or crop the original image
        visualize_bboxes_on_image(img_path, scaled_bboxes, debug_bbox_path)
        save_crops(original_pil, scaled_bboxes, out_dir, base_filename)

        if not scaled_bboxes:
            print("No bounding boxes found. Check thresholds or classes.")
            continue

        # 8) Crop and save
        save_crops(original_pil, scaled_bboxes, out_dir, base_filename)

    print("\nDone processing images.")


if __name__ == "__main__":
    main()
