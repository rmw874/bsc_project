import os
import json
import random
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import preprocess
from test_bbox_alt import post_process_mask, extract_row_col_bboxes

def save_training_crops(original_pil, bboxes, out_dir, source_filename, counter, max_crops):
    """
    Save crops with sequential naming and metadata, up to max_crops limit
    Returns: Dictionary of saved crops and their metadata, and updated counter
    """
    crops_info = {}
    
    for row_num, col_num, x1, y1, x2, y2 in bboxes:
        if counter >= max_crops:
            return crops_info, counter
            
        # Extract crop
        crop = original_pil.crop((x1, y1, x2, y2))
        
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
            "size": crop.size
        }
        
        counter += 1
        
    return crops_info, counter

def main():
    # Configure paths
    data_dir = "data/processed/Mathiesen-single-pages/training/top"
    output_dir = "data/ocr_training_crops"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set maximum number of crops
    MAX_CROPS = 500
    
    # Load model
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model.load_state_dict(torch.load('results/resnet_tversky_vanilla_lower_batch/best_model.pth'))
    model.cuda()
    model.eval()
    
    # Get all image paths and shuffle them
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)  # Randomize image order
    
    all_crops_info = {}
    crop_counter = 0
    
    for img_file in image_files:
        if crop_counter >= MAX_CROPS:
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
            crop_counter, MAX_CROPS
        )
        all_crops_info.update(crops_info)
        
        print(f"Generated {len(crops_info)} crops from {img_file}")
        print(f"Total crops so far: {crop_counter}/{MAX_CROPS}")
    
    # Save all metadata
    metadata_path = os.path.join(output_dir, "crop_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(all_crops_info, f, indent=2)
    
    print(f"\nFinished generating {crop_counter} crops")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()