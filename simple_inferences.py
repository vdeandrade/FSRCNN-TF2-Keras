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
model_path = "checkpoints/ResBlock_v2_div2k_lr1e-6_20241206-1659.h5"

# Path to the folder containing test images
test_image_dir = "data/test/tmp"
save_im = False
output_dir = "data/output_images"
scaling = 3


#%% Function:
def load_and_preprocess_images(image_path, scaling_factor=3):
    orig_image = Image.open(image_path).convert('L')  # Convert to grayscale
    width, height = orig_image.size
    # Adjust dimensions to be divisible by the scaling factor
    new_width = (width // scaling_factor) * scaling_factor
    new_height = (height // scaling_factor) * scaling_factor
    orig_image = orig_image.crop((0, 0, new_width, new_height)) # Crop and rename the original image
    orig_image = np.array(orig_image) / 255.0  # Normalize to [0, 1]

    return orig_image


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
        orig_image = load_and_preprocess_images(input_path, scaling_factor=scaling)
        width, height = orig_image.shape
        
        orig_image = orig_image[np.newaxis, :,:, np.newaxis]
        # Run inference on a single image using the trained model
        sr_image = np.squeeze(model.predict(orig_image, verbose=0))
        
        # Generate a image by bicubic interpolatio:
        orig_image_pil = Image.fromarray((orig_image[0, :, :, 0] * 255).astype(np.uint8))         # Convert the original image to a PIL Image for resizing
        bicubic_img = orig_image_pil.resize((int(width*scaling), int(height*scaling)), Image.BICUBIC)
        bicubic_img = np.array(bicubic_img) / 255.0  # Normalize to [0, 1]

        orig_image = np.squeeze(orig_image)

    
        #%% Plot results:
        plt.figure(figsize=(8, 8))
        plt.subplot(1,3,1), plt.imshow(orig_image, cmap='gray'), plt.title('Orig img'), plt.axis('off')
        plt.subplot(1,3,2), plt.imshow(bicubic_img, cmap='gray'), plt.title("Bicubic interp img"), plt.axis('off')
        plt.subplot(1,3,3, sharex=plt.gca(), sharey=plt.gca()), plt.imshow(sr_image, cmap='gray'), plt.title("SR img "), plt.axis('off')
        plt.tight_layout(), plt.axis('off'), plt.show()

        # Save the super-resolved image
        if save_im == True:
            # Postprocess the result
            sr_image_post = np.squeeze(sr_image)  # Remove batch and channel dimensions
            sr_image_post = np.clip(sr_image_post * 255.0, 0, 255).astype(np.uint8)

            output_image = Image.fromarray(sr_image_post)
            output_image.save(output_path)
            print(f"Saved: {output_path}")






