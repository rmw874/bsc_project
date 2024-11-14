import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from skimage.measure import label, regionprops
from data_preprocessing import load_image, save_segments
from model_utils import unet_model


# Load the trained model
model = load_model('models/best_model.h5')

# Path to the input image
image_path = 'data/raw/Mathiesen-single-pages/8013620831-0030.jpg-t.jpg'

# Preprocess the image to match the model input
input_image, original_image = load_image(image_path)

# Predict the segmentation mask
mask = model.predict(input_image)[0]  # Assuming the model returns (1, height, width, channels)
mask = np.argmax(mask, axis=-1)  # Convert to the final class prediction (argmax if multi-class)

# Save the individual segments as images
output_dir = 'data\processed\segments'
save_segments(original_image, mask, output_dir)

print(f"Segmentation complete. Segments saved in {output_dir}/")