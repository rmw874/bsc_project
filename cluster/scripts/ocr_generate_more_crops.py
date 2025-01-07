import os
import json
import random
from PIL import Image
import torch
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import preprocess
from post_process import post_process_mask, extract_row_col_bboxes

def get_next_crop_number(output_dir):
    """Find the next available crop number based on existing files"""
    existing_files = [f for f in os.listdir(output_dir) 
                     if f.startswith('crop_') and f.endswith('.png')]
    if not existing_files:
        return 0
    
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    return max(numbers) + 1

def save_training_crops(original_pil, bboxes, out_dir, source_filename, counter, max_crops, 
                       target_column=1, min_height=20, max_height=200):
    """
    Save crops with sequential naming and metadata, filtering for specific column
    Returns: Dictionary of saved crops and their metadata, and updated counter
    """
    crops_info = {}
    
    for row_num, col_num, x1, y1, x2, y2 in bboxes:
        # Skip if not the target column
        if col_num != target_column + 1:  # +1 because column numbers are 1-based
            continue
            
        if counter >= max_crops:
            return crops_info, counter
            
        # Calculate height
        height = y2 - y1
        
        # Skip if too small or too large
        if height < min_height or height > max_height:
            continue
            
        # Extract crop
        crop = original_pil.crop((x1, y1, x2, y2))
        
        # Skip if width is too small relative to height (likely noise)
        width = x2 - x1
        if width < height * 0.5:
            continue
            
        # Generate filename
        crop_filename = f"crop_{counter:04d}.png"
        crop_path = os.path.join(out_dir, crop_filename)
        
        # Save crop
        crop.save(crop_path)
        
        # Store metadata
        crops_info[crop_filename] = {
            "source_image": source_filename,
            "row": row_num,
            "column": col_num,
            "bbox": [x1, y1, x2, y2],
            "size": crop.size,
            "height": height,
            "width": width
        }
        
        counter += 1
        
    return crops_info, counter

def main():
    # Configure paths
    data_dir = "data/processed/Mathiesen-single-pages/training/bottom"  # Your training images directory
    output_dir = "data/ocr_training_crops"    # Where crops are saved
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing metadata if any
    metadata_path = os.path.join(output_dir, "crop_metadata.json")
    existing_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
    
    # Get the next available crop number
    start_counter = get_next_crop_number(output_dir)
    
    # Set maximum number of additional crops
    MAX_NEW_CROPS = 500
    TARGET_COLUMN = 1  # 0=Year, 1=Date, 2=Lat, 3=Long, 4=Temp
    
    print(f"Starting from crop number: {start_counter}")
    print(f"Targeting column: {TARGET_COLUMN + 1}")
    
    # Load model
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model.load_state_dict(torch.load('results/resnet_tversky_vanilla_lower_batch/best_model.pth'))
    model.cuda()
    model.eval()
    
    # Get all image paths and shuffle them
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    
    all_crops_info = existing_metadata.copy()
    crop_counter = start_counter
    target_counter = start_counter + MAX_NEW_CROPS
    
    for img_file in image_files:
        if crop_counter >= target_counter:
            break
            
        img_path = os.path.join(data_dir, img_file)
        print(f"\nProcessing: {img_path}")
        
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

        # Shuffle bboxes to get random selection from each image
        random.shuffle(scaled_bboxes)
        
        # Save crops and get their info
        crops_info, crop_counter = save_training_crops(
            original_pil, scaled_bboxes, output_dir, img_file, 
            crop_counter, target_counter, TARGET_COLUMN
        )
        all_crops_info.update(crops_info)
        
        print(f"Generated {len(crops_info)} new crops from {img_file}")
        print(f"Progress: {crop_counter - start_counter}/{MAX_NEW_CROPS}")
    
    # Save all metadata
    with open(metadata_path, 'w') as f:
        json.dump(all_crops_info, f, indent=2)
    
    print(f"\nFinished generating {crop_counter - start_counter} new crops")
    print(f"Total crops: {crop_counter}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()