import os
from PIL import Image
import imagehash

# Path to the root directory
root_dir = r'results/predictions'

# Dictionary to store image hashes and their file paths
image_hashes = {}

# Traverse through subdirectories
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            file_path = os.path.join(subdir, file)
            
            try:
                # Open the image and compute its hash
                with Image.open(file_path) as img:
                    hash_value = imagehash.average_hash(img)
                    
                    # Check if the hash already exists in the dictionary
                    if hash_value in image_hashes:
                        print(f"Duplicate or similar image found: {file_path} is similar to {image_hashes[hash_value]}")
                    else:
                        image_hashes[hash_value] = file_path
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

print("Comparison complete.")