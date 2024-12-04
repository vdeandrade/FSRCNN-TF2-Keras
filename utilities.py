# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:21:27 2024

@author: Vincent
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import time

#%% Convert bmp to png
file_path = r"D:\vdeandrade\General100\archive"
file_list = os.listdir(file_path)
for i in file_list:
    source_files = os.path.join(file_path, i)
    img = Image.open(source_files)
    file_name = i[:-4]+'.png'
    file_name_saved = os.path.join(file_path, file_name)
    img.save(file_name_saved)
    # print(file_name_saved)


#%%
# Path to the saved model
model_path = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\checkpoints\best_model_dyn_fast_learning.h5"
model_path = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\checkpoints\ResBlock_L2_model_patch_lr1e-3_20241202-2328.h5"
model_path = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN-TF2-Keras\checkpoints\ResBlock_v2_patch_lr1e-4_20241203-1423.h5"

# Define the custom PSNR metric function
def psnr_metric(y_true, y_pred):
    max_pixel = 1.0  # Assuming normalized images in the range [0, 1]
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)

# Load the model
model = tf.keras.models.load_model(model_path, custom_objects={'psnr_metric': psnr_metric})
# Plot the model summary
print("Model Summary:")
model.summary()


#%% Displaying Basic Information:
# You can loop through the dataset and print out details about each element,
# such as shape and dtype:
# for element in train_dataset.take(5):  # Limit the output to the first 5 elements
#     lr_image, hr_image = element
#     print("Low-Resolution Image Shape:", lr_image.shape)
#     print("Low-Resolution Image Type:", lr_image.dtype)
#     print("High-Resolution Image Shape:", hr_image.shape)
#     print("High-Resolution Image Type:", hr_image.dtype)
#     print("------")

#%%
# for element in test_dataset.take(5):  # Limit the output to the first 5 elements
#     lr_image, hr_image = element
#     print("Low-Resolution Image Shape:", lr_image.shape)
#     print("Low-Resolution Image Type:", lr_image.dtype)
#     print("High-Resolution Image Shape:", hr_image.shape)
#     print("High-Resolution Image Type:", hr_image.dtype)
#     print("------")


#%%
# Exploring Dataset Elements:
#----------------------------
# Print Element Types and Shapes: You can print the types and shapes of the elements
# in the dataset. This will give you an idea of what kind of data is stored
# in the dataset and how it's structured.

def inspect_dataset(dataset, num_elements=5):
    for element in dataset.take(num_elements):
        if isinstance(element, (tuple, list)):
            for i, el in enumerate(element):
                print(f"Element {i + 1}:")
                print(f"  Type: {type(el)}")
                print(f"  Shape: {el.shape}")
                print(f"  DType: {el.dtype}")
                print("------")
        else:
            print("Single Element:")
            print(f"  Type: {type(element)}")
            print(f"  Shape: {element.shape}")
            print(f"  DType: {element.dtype}")
            print("------")

# Call the function to inspect the dataset
# inspect_dataset(dataset)


#%%
# Convert Tensor to Numpy for Further Inspection: To get more detailed insights,
# you can convert TensorFlow tensors to NumPy arrays and inspect their content.
def inspect_dataset_content(dataset, num_elements=5):
    for element in dataset.take(num_elements):
        if isinstance(element, (tuple, list)):
            for i, el in enumerate(element):
                el_np = el.numpy()
                print(f"Element {i + 1} content (first few values): {el_np.flat[:5]}")
        else:
            element_np = element.numpy()
            print(f"Single Element content (first few values): {element_np.flat[:5]}")

# Call the function to inspect the content of the dataset
# inspect_dataset_content(dataset)

# Summarize Dataset: Create a summary function to provide a comprehensive overview
# of the dataset, including the shapes and types of the first few elements.
def summarize_dataset(dataset, num_elements=5):
    print("Dataset Summary:")
    for element in dataset.take(num_elements):
        if isinstance(element, (tuple, list)):
            for i, el in enumerate(element):
                print(f"Element {i + 1}:")
                print(f"  Type: {type(el)}")
                print(f"  Shape: {el.shape}")
                print(f"  DType: {el.dtype}")
                print(f"  Content (first few values): {el.numpy().flat[:5]}")
                print("------")
        else:
            print("Single Element:")
            print(f"  Type: {type(element)}")
            print(f"  Shape: {element.shape}")
            print(f"  DType: {element.dtype}")
            print(f"  Content (first few values): {element.numpy().flat[:5]}")
            print("------")

# Call the function to summarize the dataset
# summarize_dataset(dataset)

