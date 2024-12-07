# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:39:35 2024

@author: Vincent
"""
# -*- coding: utf-8 -*-
"""
Modified script to crop a specified number of patches from the middle of each image.
"""
import os
from PIL import Image


#%% INPUT:
input_folder = r"E:\vdeandrade\Stanford_project\Models_from_github\FSRCNN-TF2-Keras\data\DIV2K_train_HR" # path to the folder with images
patch_size = 36 # patch size (n x n)
output_folder = r"E:\vdeandrade\Stanford_project/DIV2K_train_HR_patch" # path to the output folder storing the patches
num_patches = 625  # Number of middle patches to extract

input_folder = r"E:\vdeandrade\Stanford_project\Models_from_github\FSRCNN-TF2-Keras\data\DIV2K_valid_HR" # path to the folder with images
patch_size = 36 # patch size (n x n)
output_folder = r"E:\vdeandrade\Stanford_project/DIV2K_valid_HR_patch" # path to the output folder storing the patches
num_patches = 625  # Number of middle patches to extract


#%% Functions
def cut_middle_patches(image_path, patch_size, num_patches, output_folder):
    """
    Extract a specified number of patches centered in the middle of the image.
    """
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate the center of the image
    center_x, center_y = img_width // 2, img_height // 2

    # Get the base name of the image to create output file names
    base_name = os.path.basename(image_path)
    image_name, _ = os.path.splitext(base_name)

    # Calculate patch offsets
    offsets = []
    half_num_patches = num_patches // 2
    for i in range(-half_num_patches, half_num_patches + 1):
        for j in range(-half_num_patches, half_num_patches + 1):
            if len(offsets) < num_patches:
                offsets.append((i * patch_size, j * patch_size))
            else:
                break

    # Extract and save patches
    patch_count = 0
    for dx, dy in offsets:
        left = max(center_x + dx - patch_size // 2, 0)
        upper = max(center_y + dy - patch_size // 2, 0)
        right = min(left + patch_size, img_width)
        lower = min(upper + patch_size, img_height)

        # Ensure the crop box is valid
        if right > left and lower > upper:
            patch = img.crop((left, upper, right, lower))
            
            # Save the patch
            patch_file_name = f"{image_name}_patch_{patch_count}.png"
            patch.save(os.path.join(output_folder, patch_file_name))
            patch_count += 1

def process_images(input_folder, patch_size, num_patches, output_folder):
    """
    Process all images in the input folder and extract middle patches.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            cut_middle_patches(image_path, patch_size, num_patches, output_folder)

#%% Process the images and save the patches
process_images(input_folder, patch_size, num_patches, output_folder)
print(f"All middle patches saved to {output_folder}")
