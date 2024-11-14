import tensorflow as tf
from model_utils import unet_model
from data_preprocessing import load_images_and_masks

# Load the images and masks
image_dir = 'data/raw/images'
mask_dir = 'data/raw/masks'
img_dimensions = 512
train_images, train_masks = load_images_and_masks(image_dir, mask_dir, img_size=(img_dimensions, img_dimensions))

# Initialize and compile the UNet model
model = unet_model(input_shape=(img_dimensions, img_dimensions, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_masks,
    epochs=100,
    batch_size=4,
    # validation_split=0.2,
    callbacks=[tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)]
)
