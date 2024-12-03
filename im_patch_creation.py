# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:47:14 2024

@author: Vincent
"""
import os
from PIL import Image
import numpy as np

#%% INPUT:
# Get the patch size and input/output folders from the user
input_folder = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\data\train" # path to the folder with images
patch_size = 36 # patch size (n x n)
output_folder = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\data\train_patches" # path to the output folder storing the patches

input_folder = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\data\test\Set5_Set14_some_urban100_general100" # path to the folder with images
patch_size = 36 # patch size (n x n)
output_folder = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\data\test\val_patches" # path to the output folder storing the patches


#%% Functions
def cut_image_into_patches(image_path, patch_size, output_folder):
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Get the base name of the image to create output file names
    base_name = os.path.basename(image_path)
    image_name, _ = os.path.splitext(base_name)

    # Loop through the image and extract patches
    patch_count = 0
    for i in range(0, img_width, patch_size):
        for j in range(0, img_height, patch_size):
            # Define the bounding box for the patch
            box = (i, j, min(i + patch_size, img_width), min(j + patch_size, img_height))
            patch = img.crop(box)

            # Save the patch as a new image
            patch_file_name = f"{image_name}_patch_{patch_count}.png"
            patch.save(os.path.join(output_folder, patch_file_name))
            patch_count += 1

def process_images(input_folder, patch_size, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            cut_image_into_patches(image_path, patch_size, output_folder)


#%% Process the images and save the patches
process_images(input_folder, patch_size, output_folder)
print(f"All patches saved to {output_folder}")
