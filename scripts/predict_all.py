import torch
import numpy as np
from model_utils import UNet
import os
import cv2

def load_image(image_path, img_size=(1024, 1024)):
    """
    Load and preprocess a single image
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    
    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0
    
    # Transpose from (H, W, C) to (C, H, W)
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_expanded = np.expand_dims(img, axis=0)
    
    return img, img_expanded

def predict_mask(model, image_tensor, device):
    """
    Predict mask for a single image using the trained model
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
        return prediction.cpu().numpy()

def save_segments(image, mask, output_dir):
    """
    Save segmented regions as separate images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert mask to binary uint8
    mask_binary = (mask * 255).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask_binary)
    
    # Save each segment
    for label in range(1, num_labels):
        segment_mask = (labels == label).astype(np.uint8)
        segment = image.copy()
        segment = segment.transpose(1, 2, 0)  # From (3, 1024, 1024) to (1024, 1024, 3)
        segment[segment_mask == 0] = 0
        cv2.imwrite(os.path.join(output_dir, f'segment_{label}.png'), segment)

def process_directory(model, input_dir, output_base_dir, device):
    """
    Process all images in a directory
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            print(f"Processing {filename}")
            
            # Create output directory for this image
            image_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_base_dir, image_name)
            
            # Process image
            image_path = os.path.join(input_dir, filename)
            original_image, preprocessed_image = load_image(image_path)
            
            # Convert to torch tensor
            image_tensor = torch.from_numpy(preprocessed_image).float()
            
            # Predict and save segments
            predicted_mask = predict_mask(model, image_tensor, device)
            predicted_mask = predicted_mask[0, 0]
            
            save_segments(original_image, predicted_mask, output_dir)
            print(f"Saved segments to {output_dir}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # Directory containing new images to process
    input_dir = "data/raw/Mathiesen-single-pages"  # Update this path
    output_base_dir = "results/predictions"
    
    # Process all images
    process_directory(model, input_dir, output_base_dir, device)
    
    print(f"Segmentation complete. Results saved in {output_base_dir}")

if __name__ == "__main__":
    main()