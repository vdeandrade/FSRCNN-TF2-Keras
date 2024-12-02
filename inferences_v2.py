# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:35:39 2024

@author: Vincent
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os



#%% Initial code setup:
print("\033[H\033[J"); time.sleep(0.1) # clear the ipython console before starting the code.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#%% Input:
model_path = "checkpoints/best_model_real_im.h5"
test_image_dir = "data/test/Set5"
output_dir = "data/output_images"
scaling = 3  # Scaling factor
save_images = True  # Whether to save the output images


#%% Functions
def load_and_preprocess_image(image_path):
    """
    Load an image and preprocess it for inference.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=(0, -1))  # Add batch and channel dimensions

def postprocess_image(image_array):
    """
    Convert model output to a displayable and savable format.
    """
    image_array = np.squeeze(image_array)  # Remove batch and channel dimensions
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    return image_array

# Downscale function
def downscale_image(image, scale_factor):
    """
    Downscale an image using TensorFlow's resize method.
    """
    height, width = image.shape[:2]
    new_height, new_width = height // scale_factor, width // scale_factor
    downscaled = tf.image.resize(image, (new_height, new_width), method="bicubic")
    return downscaled.numpy()


def modify_model_for_dynamic_input(model_path):
    """
    Modify a pre-trained model to accept dynamic input sizes while preserving trained parameters.
    """
    # Load the trained model
    original_model = tf.keras.models.load_model(model_path, compile=False)

    # Define a new input layer with a dynamic shape
    input_layer = tf.keras.layers.Input(shape=(None, None, 1))  # Dynamic input shape
    x = input_layer

    for layer in original_model.layers[1:]:  # Skip the original input layer
        if isinstance(layer, tf.keras.layers.PReLU):
            # Get the weights from the original PReLU layer
            weights = layer.get_weights()
            
            # Reshape weights to ensure compatibility
            alpha = weights[0]  # Alpha parameter
            reshaped_alpha = alpha.mean(axis=(0, 1), keepdims=True)  # Reduce spatial dimensions
            
            # Create a new PReLU layer with shared axes
            new_prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
            x = new_prelu(x)  # Apply PReLU to the current tensor
            
            # Set weights for the new PReLU layer
            new_prelu.set_weights([reshaped_alpha])  # Set reshaped alpha as weights
        else:
            x = layer(x)

    # Rebuild the model with the dynamic input layer
    dynamic_model = tf.keras.Model(inputs=input_layer, outputs=x)

    return dynamic_model


def run_inference(model, input_image):
    """
    Run inference on a single image using the trained model.
    """
    return model.predict(input_image, verbose=0)

def save_image(image_array, output_path):
    """
    Save an image to disk.
    """
    output_image = Image.fromarray(image_array)
    output_image.save(output_path)

def main():
    # Modify the model for dynamic input sizes
    model = modify_model_for_dynamic_input(model_path)

    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(test_image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(test_image_dir, filename)
            output_path = os.path.join(output_dir, f"sr_{filename}")

            print(f"Processing: {filename}")

            # Load and preprocess the input image
            input_image = load_and_preprocess_image(input_path)

            # Downscale the image
            # downscaled_image = downscale_image(original_image[0, ..., 0], scaling)
            
            # Run inference
            sr_image = run_inference(model, input_image)

            # Generate bicubic interpolation for comparison
            original_image = Image.open(input_path).convert('L')
            im_bicubic = original_image.resize(
                (original_image.width * scaling, original_image.height * scaling),
                Image.BICUBIC
            )

            # Postprocess and visualize
            sr_image_post = postprocess_image(sr_image)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title("Input Image")
            plt.subplot(1, 3, 2)
            plt.imshow(im_bicubic, cmap='gray')
            plt.title("Bicubic Interpolation")
            plt.subplot(1, 3, 3)
            plt.imshow(sr_image_post, cmap='gray')
            plt.title("Super-Resolved Image")
            plt.show()

            # Save super-resolved image
            if save_images:
                save_image(sr_image_post, output_path)
                print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
