# scripts/inference.py

import os
import torch
import cv2
import numpy as np

from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

def main():
    # Paths
    model_path = 'models/unet_best.pth'
    input_images_dir = 'path_to_new_images/'  # Replace with the path to your new images
    output_dir = 'outputs/'
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = UNet(n_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Transformation
    transform = get_transforms()

    # Process each image
    images = sorted(os.listdir(input_images_dir))
    for img_name in images:
        img_path = os.path.join(input_images_dir, img_name)
        image = cv2.imread(img_path)
        original_size = (image.shape[1], image.shape[0])  # Width, Height
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            pred_mask = probs.cpu().numpy()[0, 0]

        # Resize prediction to original size
        pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
        pred_binary = (pred_mask_resized > 0.5).astype(np.uint8)

        # Post-processing to detect lines and cells
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            pred_binary,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        line_img = np.zeros_like(pred_binary)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)

        # Find contours of cells
        contours, _ = cv2.findContours(255 - line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cell_coords = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cell_coords.append((x, y, w, h))

        # Sort cells and assign positions
        cell_coords_sorted = sorted(cell_coords, key=lambda c: (c[1], c[0]))  # Sort by y, then x

        # Save or process individual cells
        original_image = cv2.imread(img_path)
        for idx, (x, y, w, h) in enumerate(cell_coords_sorted):
            cell_image = original_image[y:y+h, x:x+w]
            cell_output_path = os.path.join(output_dir, f'cell_{idx}_{img_name}')
            cv2.imwrite(cell_output_path, cell_image)
            print(f'Saved cell {idx} from {img_name}.')

        # Optionally, save the boundary mask
        pred_mask_uint8 = (pred_mask_resized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f'boundary_{img_name}'), pred_mask_uint8)

    print('Inference complete.')

if __name__ == '__main__':
    main()