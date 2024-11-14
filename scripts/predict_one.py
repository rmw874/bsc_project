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
        prediction = torch.sigmoid(prediction)  # Apply sigmoid to get probabilities
        prediction = (prediction > 0.5).float()  # Threshold at 0.5
        return prediction.cpu().numpy()

def save_segments(image, mask, output_dir):
    """
    Save segmented regions as separate images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert mask to binary uint8
    mask_binary = (mask * 255).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask_binary)
    
    # Save each segment
    for label in range(1, num_labels):  # Skip background (0)
        # Create a mask for this segment
        segment_mask = (labels == label).astype(np.uint8)
        
        # Apply mask to original image
        segment = image.copy()
        # Print shapes for debugging
        # Transpose segment to (H, W, C)
        segment = segment.transpose(1, 2, 0)  # From (3, 1024, 1024) to (1024, 1024, 3)

        # Print new shape for verification
        print(f"Transposed segment shape: {segment.shape}")

        # Expand the mask
        segment_mask_expanded = np.repeat(segment_mask[:, :, np.newaxis], 3, axis=2)  # Shape (1024, 1024, 3)

        # Apply the mask
        segment[segment_mask_expanded == 0] = 0


        
        # Save segment
        cv2.imwrite(os.path.join(output_dir, f'segment_{label}.png'), segment)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # Path to your new image
    image_path = "data/raw/Mathiesen-single-pages/8013620831-0033.jpg-t.jpg"  # Update this path
    print(f"Processing image: {image_path}")
    
    # Load and preprocess the image
    original_image, preprocessed_image = load_image(image_path)
    
    # Convert numpy array to torch tensor
    image_tensor = torch.from_numpy(preprocessed_image).float()
    
    # Print shapes for debugging
    print(f"Input tensor shape: {image_tensor.shape}")
    
    # Make prediction
    predicted_mask = predict_mask(model, image_tensor, device)
    predicted_mask = predicted_mask[0, 0]  # Remove batch and channel dimensions
    
    # Create output directory
    output_dir = "results/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the predicted segments
    save_segments(original_image, predicted_mask, output_dir)
    
    print(f"Segmentation complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()