# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:20:39 2024

@author: Vincent
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import time

#%%
print("\033[H\033[J"); time.sleep(0.1) # clear the ipython console before starting the code.
plt.close('all')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#%% INPUTS:

# Path to the trained model
model_path = "checkpoints/best_model.h5"

# Path to the folder containing test images
test_image_dir = "data/test/Set5"
save_im = False
output_dir = "data/output_images"
scaling = 3


#%% Function:
def load_and_preprocess_images(image_path, scaling_factor=3):
    """
    Load an image and preprocess it for inference.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Optional. Resize to target size (height, width).
    Returns:
        np.ndarray: Preprocessed image array.
    """
    orig_image = Image.open(image_path).convert('L')  # Convert to grayscale
    height, width = orig_image.size
    height_down, width_down = height // scaling_factor, width // scaling_factor
    image_downsized = orig_image.resize((height_down, width_down), Image.BICUBIC) # Bicubic downsampling with PIL
    image_up_bicubic = image_downsized.resize((height, width), Image.BICUBIC) # Bicubic downsampling with PIL
    image_downsized = np.expand_dims(image_downsized, axis=(0, -1))  # Add batch and channel dimensions for the TF model
    image_downsized = np.array(image_downsized) / 255.0  # Normalize to [0, 1]

    image_up_bicubic = np.array(image_up_bicubic) / 255.0  # Normalize to [0, 1]

    orig_image = np.array(orig_image) / 255.0  # Normalize to [0, 1]

    return image_downsized, image_up_bicubic, orig_image


def modify_model_for_dynamic_input(model_path):
    """
    Modify the trained model to accept dynamic input sizes.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Create a new input layer with dynamic shape
    new_input = tf.keras.layers.Input(shape=(None, None, 1))
    x = new_input

    # Rebuild the model layer by layer
    for layer in model.layers[1:]:  # Skip the original Input layer
        x = layer(x)

    # Create the new model
    model_dynamic = tf.keras.models.Model(inputs=new_input, outputs=x)
    return model_dynamic


#%% Load the trained model
# model = modify_model_for_dynamic_input(model_path)
model = tf.keras.models.load_model(model_path, compile=False)


#%%
if save_im == True:
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

# Loop over test images
for filename in os.listdir(test_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        input_path = os.path.join(test_image_dir, filename)
        output_path = os.path.join(output_dir, f"sr_{filename}")

        print(f"Processing: {filename}")
        
        # Preprocess the image
        input_image, bicubic_img, orig_image = load_and_preprocess_images(input_path, scaling_factor=scaling)

        # Run inference on a single image using the trained model
        sr_image = np.squeeze(model.predict(input_image, verbose=0))
        
        # Generate a image by bicubic interpolatio:
        input_image  = np.squeeze(input_image) # after being passed in the model, dimensions of 1 can be squeezed
        
        plt.figure()
        plt.subplot(2,2,1), plt.imshow(orig_image, cmap='gray'), plt.title('Original image')
        plt.subplot(2,2,2), plt.imshow(input_image, cmap='gray'), plt.title('Input downsampled (%X) image' % scaling)
        plt.subplot(2,2,3), plt.imshow(bicubic_img, cmap='gray'), plt.title('bicubic interpolation (%X)' % scaling)
        plt.subplot(2,2,4), plt.imshow(sr_image, cmap='gray'), plt.title('SR image')
        plt.show()

        # Save the super-resolved image
        if save_im == True:
            # Postprocess the result
            sr_image_post = np.squeeze(sr_image)  # Remove batch and channel dimensions
            sr_image_post = np.clip(sr_image_post * 255.0, 0, 255).astype(np.uint8)

            output_image = Image.fromarray(sr_image_post)
            output_image.save(output_path)
            print(f"Saved: {output_path}")






