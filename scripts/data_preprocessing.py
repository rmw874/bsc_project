import os
import cv2  # You may need to install this with `pip install opencv-python`
import numpy as np
from skimage.measure import label, regionprops


def load_images_and_masks(image_dir, mask_dir, img_size=(1024, 1024)):
    images = []
    masks = []
    
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Load and resize the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize the image
        images.append(img)
        
        # Load and resize the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0  # Normalize the mask to [0, 1]
        masks.append(mask)
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    masks = np.expand_dims(masks, axis=-1)  # Add a channel dimension for masks

    return images, masks

def load_image(image_dir, img_size=(1024,1024)):
    img = cv2.imread(image_dir)
    img = cv2.resize(img, img_size)
    img = img/255 #normailze pixel values
    img_expanded = np.expand_dims(img, axis=0)
    return img, img_expanded

def save_segments(image, mask, output_dir):
    # Label connected components in the mask
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    os.makedirs(output_dir, exist_ok=True)

    for i, region in enumerate(regions):
        # Create a blank image with the same shape as the original
        segment = np.zeros_like(image)
        
        # Mask the image with the region's coordinates
        minr, minc, maxr, maxc = region.bbox
        segment[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]

        # Save each segment as a new image
        cv2.imwrite(f"{output_dir}/segment_{i+1}.png", segment)