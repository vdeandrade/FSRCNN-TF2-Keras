#!/usr/bin/env python
"""
    This new version used shared_axes with PReLU activation layer to share the learned parameters.
    It enables the model to accept any image size as an input without the need to adjust its dimensions.

    Use tensorboard --logdir logs/fit to monitor the training or analyzed the logs
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.image import psnr
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Input, Conv2DTranspose, PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from keras.metrics import MeanMetricWrapper
from keras.backend import clear_session
from numpy import ceil
import matplotlib.pyplot as plt
import glob, time, os, logging, datetime
import pandas as pd


#%% USER INPUT:
#------------------------------------------------------------------------------------
train_image_paths = glob.glob(r"D:\vdeandrade\Deep_learning\training_data\test_patches/*.png") # 281 real images cutted in 36x36 pix patches
test_image_paths = glob.glob(r"D:\vdeandrade\Deep_learning\training_data\train_patches/*.png") # 36x36 pix patches
# train_image_paths = glob.glob("./data/train/*.png") # 281 real images
# test_image_paths = glob.glob("./data/test/Set5_Set14_some_urban100_general100/*.png")

model_fname = "dyn_model_small_patch_lr1e-4" # date is added later to avoid erasing models by mistake
model_name = "FSRCNN" # Name for TF. The scaling factor is added to the name later
hr_img_size = (258, 258) # CHOOSE THE TARGET SIZE FOR ITS DIVISION WITH THE SCALING FACTOR TO RETURN AN EVEN NUMBER
hr_img_size = (36, 36) # CHOOSE THE TARGET SIZE FOR ITS DIVISION WITH THE SCALING FACTOR TO RETURN AN EVEN NUMBER
batch_size = 446 # 64
learning_rate = 0.0001
epochs = 100
save_every_n_epochs = 200
scaling = 3
aug_factor = 6 # will add to the training data X times the amount of training data
# Load pre-trained weights
load_former_model = True
pretrained_weights_path = './checkpoints/dyn_model_small_patch_lr1e-4.h5'
early_stop = False
#------------------------------------------------------------------------------------


#%% Initial code setup:
print("\033[H\033[J"); time.sleep(0.1) # clear the ipython console before starting the code.
clear_session()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Add environment variables to handle OpenMP conflicts
logging.basicConfig(level=logging.INFO)


#%% FUNCTIONS:
#-------------
# Load an image and preprocess it
def load_image(filepath):
    """
    Loads an image, decodes it to grayscale, and normalizes pixel values to [0, 1].
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=1)  # Convert to grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
    img.set_shape([None, None, 1])    # Set static shape if known (e.g., for consistent pipeline behavior)
    return img

def generate_lr(hr_image, scale=2):
    """
    Generate a low-resolution version of the given high-resolution image.
    Args:
        hr_image: A high-resolution image tensor.
        scale: the downscaling factor of the high-resolution image
    Returns:
        lr_image: low-resolution image tensor.
    """
    hr_height, hr_width = tf.shape(hr_image)[0], tf.shape(hr_image)[1] # tf.shape dynamically computes the shape
    lr_height, lr_width = hr_height // scale, hr_width // scale
    lr_image = tf.image.resize(hr_image, [lr_height, lr_width], method='bicubic')
    return lr_image

# Preprocessing function for tf.data.Dataset
def preprocess(filepath, scale, target_size=(256, 256)):
    """
    Preprocesses an image by generating its low-resolution version.
    Args:
        filepath: A TensorFlow string tensor representing the file path.
        scale: The scaling factor for low-resolution generation.
    Returns:
        A tuple (lr_image, hr_image) containing the low-resolution and high-resolution images.
    """
    # Load the HR image
    hr_image = load_image(filepath)
    
    # Ensure the HR image has size matching hr_img_size input
    # if too large, image cropped, if too small, image padded
    hr_image = crop_and_pad_images(hr_image, target_size)  # Only crop/pad HR image
    
    # Generate the LR image by downscaling
    lr_image = generate_lr(hr_image, scale)
    
    return lr_image, hr_image


#%% SCALING TRAINING IMAGES:
#---------------------------
# Training images are of various size. These functions ensure their size
# become constant thanks to cropping of padding before starting the training
def crop_and_pad_images(hr_image, target_size=(256, 256)):
    """
    Crops images larger than target_size and pads images smaller than target_size.
    Cropping removes borders equally from all sides. 
    Padding adds zero-padding to make up the difference in size.

    Args:
        hr_image: High-resolution image tensor that potenttially needs to be cropped and padded
        target_size: Tuple (height, width) specifying the target dimensions.

    Returns:
        Cropped and padded HR image tensors.
    """
    def crop_or_pad(image, target_height, target_width):
        # Get current height and width
        current_height, current_width = tf.shape(image)[0], tf.shape(image)[1]
    
        # Determine offsets for cropping (if needed)
        offset_height = tf.maximum((current_height - target_height) // 2, 0)
        offset_width = tf.maximum((current_width - target_width) // 2, 0)
    
        # Crop the image if dimensions are larger than the target
        cropped_height = tf.minimum(current_height, target_height)
        cropped_width = tf.minimum(current_width, target_width)
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=cropped_height,
            target_width=cropped_width)
    
        # Pad the image if dimensions are smaller than the target
        image = tf.image.pad_to_bounding_box(
                image,
                offset_height=0,
                offset_width=0,
                target_height=target_height,
                target_width=target_width)
    
        return image

    # Apply cropping and padding to both LR and HR images
    hr_image = crop_or_pad(hr_image, target_size[0], target_size[1])

    return hr_image

#%% DATA AUGMENTATION
#------------------
def augment_image(lr_image, hr_image):
    """
    Consistently augments the LR and HR images using ImageDataGenerator.
    Args:
        lr_image: Low-resolution image (NumPy array).
        hr_image: High-resolution image (NumPy array).
    Returns:
        Augmented LR and HR images.
    """
    transform = False
    if tf.random.uniform([]) > 0.5:                        # Random horizontal flip
        transform = True
        lr_image = tf.image.flip_left_right(lr_image)
        hr_image = tf.image.flip_left_right(hr_image)
    # if tf.random.uniform([]) > 0.5:                        # Random vertical flip
    #     lr_image = tf.image.flip_up_down(lr_image)
    #     hr_image = tf.image.flip_up_down(hr_image)
    if tf.random.uniform([]) > 0.4:                        # Random rotation
        transform = True
        rotations = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        lr_image = tf.image.rot90(lr_image, rotations)
        hr_image = tf.image.rot90(hr_image, rotations)
    if tf.random.uniform([]) > 0.4:                        # Random cropping (80%-100% of original size)
        transform = True
        crop_size = tf.random.uniform([], minval=0.8, maxval=1.0)
        lr_crop = tf.image.central_crop(lr_image, crop_size)
        hr_crop = tf.image.central_crop(hr_image, crop_size)
        lr_image = tf.image.resize(lr_crop, tf.shape(lr_image)[:2])
        hr_image = tf.image.resize(hr_crop, tf.shape(hr_image)[:2])
    if transform == False: # If no transformation randomly selected, force horizontal flip
        lr_image = tf.image.flip_left_right(lr_image)
        hr_image = tf.image.flip_left_right(hr_image)

    return lr_image, hr_image


def augment_dataset(dataset):
    """
    Applies augmentation to LR-HR pairs in a tf.data.Dataset.
    Args:
        dataset: A tf.data.Dataset containing (lr_image, hr_image) pairs.
    Returns:
        A tf.data.Dataset with augmented LR-HR image pairs.
    """
    new_dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    return new_dataset


def create_dataset(image_paths, scale, batch_size, target_size=(256, 256), augment=False, augment_factor=1, buffer_size=1000, is_test=False):
    """
    Creates a tf.data.Dataset with optional augmentation, mixing original and augmented images.
    Args:
        image_paths: List of paths to input images.
        scale: Downscaling factor for LR generation.
        batch_size: Number of images per batch.
        target_size: Tuple specifying the target size for images (height, width).
        augment: Whether to augment the dataset.
        augment_factor: How many times to augment the dataset.
        buffer_size: Size of the shuffle buffer.

    Returns:
        A tf.data.Dataset combining original and augmented images.
    """

    # Map the preprocess function --> downscale low resolution images
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda path: preprocess(path, scale, target_size), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Augment and combine datasets
    if augment:
        augmented_dataset = augment_dataset(dataset)  # Apply augmentation
        for _ in range(augment_factor - 1):  # Repeat augmentation as needed
            augmented_dataset = augmented_dataset.concatenate(augment_dataset(dataset))
        dataset = dataset.concatenate(augmented_dataset)  # Combine original and augmented

    # Shuffle, batch, and prefetch
    if buffer_size > 0:  # Shuffle only if buffer_size > 0
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=not is_test).prefetch(tf.data.AUTOTUNE)

    return dataset

#%% Metric function
def psnr_metric(y_true, y_pred):
    """
    Calculates PSNR between 2 tensors.The max_val argument specifies the dynamic
    range of the images. For normalized images, set max_val=1.0.
    """
    return psnr(y_true, y_pred, max_val=1.0)


#%% DIAGNOSTICS:
#   Run the following command in the terminal to visualize training metrics: tensorboard --logdir logs/fit
class DiagnosticsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Logs keys = {list(logs.keys())}")
        train_loss = logs.get('loss')
        train_psnr = logs.get('psnr_metric')
        val_loss = logs.get('val_loss')
        val_psnr = logs.get('val_psnr_metric')

        print(f"Train Loss: {train_loss}, Train PSNR: {train_psnr}")
        print(f"Val Loss: {val_loss}, Val PSNR: {val_psnr}")
            
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=0)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch')

# tensorboard_callback._get_writer(log_dir).flush()


#%% Create train and test datasets
train_dataset = create_dataset(train_image_paths, scaling, batch_size, target_size=hr_img_size, augment=True, augment_factor=aug_factor)
test_dataset  = create_dataset(test_image_paths,  scaling, batch_size, target_size=hr_img_size, augment=False, is_test=True)

# Check the first few elements of the training dataset
for lr, hr in train_dataset.take(1):
    logging.info(f"Training pairs low-res shape: {lr.shape}, High-res shape: {hr.shape}")

# Check the first few elements of the test dataset
for lr, hr in test_dataset.take(1):
    logging.info(f"Test pairs low-res shape: {lr.shape}, High-res shape: {hr.shape}")

# Calculate the total number of image pairs (depends on amount of data augmentation)
flat_dataset = train_dataset.unbatch()
total_image_pairs = flat_dataset.reduce(0, lambda x, _: x + 1).numpy()
logging.info(f"Total image pairs: {total_image_pairs}")


#%% Build the model:
#-------------------
# input_img = Input(shape=(None, None, 1))
# lr_img_size = tuple(dim // scaling for dim in hr_img_size) + (1,)
# input_img = Input(shape=lr_img_size)

input_img = Input(shape=(None, None, 1))  # Dynamic input shape

# 1) Feature extraction layer:
model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU(shared_axes=[1, 2])(model)

# 2) Shrinking layer reducing feature dimensions:
model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(model)
model = PReLU(shared_axes=[1, 2])(model)

# 3) Multiple mapping layers perform transformations in the low-dimensional space:
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU(shared_axes=[1, 2])(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU(shared_axes=[1, 2])(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU(shared_axes=[1, 2])(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = PReLU(shared_axes=[1, 2])(model)

# 4) Expanding layer restoring the dimensionality:
model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_normal')(model)
model = PReLU(shared_axes=[1, 2])(model)

# 5) Deconvolution layer performs upscaling to obtain the final HR image
model = Conv2DTranspose(1, (9, 9), strides=(scaling, scaling), padding='same')(model)


#%% COMPILE THE MODEL:
output_img = model
model = Model(input_img, output_img, name=model_name+'_'+str(scaling)+'X')

if load_former_model == True:
    try:
        model.load_weights(pretrained_weights_path)
        print(f"Successfully loaded pre-trained weights from {pretrained_weights_path}")
    except Exception as e:
        print(f"Could not load pre-trained weights. Training will start from scratch. Error: {e}")
else:
    print("Starting training from scratch.")

# Compile the model (fresh optimizer, but resume from weights)
model.compile(optimizer=Adam(learning_rate=learning_rate),  # You can tweak the learning rate if needed
            loss='mse',
            metrics=[MeanMetricWrapper(psnr_metric, name='psnr_metric')])

model.summary() # print the model summary


#%% MANAGE THE CHECKPOINTS:
# Save the best model based on validation loss
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
filepath_model = f"./checkpoints/{model_fname}_{current_time}.h5"
best_checkpoint = ModelCheckpoint(
    # filepath, monitor='val_psnr', save_best_only=True, mode='max', verbose=1) # Mode set to 'max' because PSNR improves as the model gets better
    filepath=filepath_model, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Save periodic checkpoints every epoch:
steps_per_epoch = len(train_dataset)
save_freq = steps_per_epoch * save_every_n_epochs
periodic_checkpoint = ModelCheckpoint(filepath="./checkpoints/weights_epoch-{epoch:02d}.h5", save_freq=save_freq, verbose=1)

if early_stop == True:
    early_stopping = EarlyStopping(
                        monitor='val_psnr_metric',  # Stop if validation PSNR stops improving
                        patience=10,                # Number of epochs to wait for improvement
                        restore_best_weights=True)  # Restore weights from the best epoch

    # callbacks_list = [best_checkpoint, periodic_checkpoint, tensorboard_callback, early_stopping, DiagnosticsCallback()] # Define checkpoints callback
    callbacks_list = [best_checkpoint, tensorboard_callback, early_stopping, DiagnosticsCallback()] # Define checkpoints callback
else:
    # callbacks_list = [best_checkpoint, periodic_checkpoint, tensorboard_callback, DiagnosticsCallback()] # Define checkpoints callback
    callbacks_list = [best_checkpoint, tensorboard_callback, DiagnosticsCallback()] # Define checkpoints callback
    

#%% Train the model
#----------------
history = model.fit(train_dataset,
              steps_per_epoch = ceil(total_image_pairs / batch_size),
              validation_data=test_dataset,
              epochs=epochs,
              callbacks=callbacks_list)

print("Done training!!!")

#%% Convert history to a DataFrame
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)


#%% Plot PSNR
# Access the training history after training:
# Plot Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.title('Loss During Training')
# Plot PSNR
plt.subplot(1,2,2)
plt.plot(history.history['psnr_metric'], label='Train PSNR')
plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
plt.xlabel('Epoch'), plt.ylabel('PSNR'), plt.legend(), plt.title('PSNR During Training')
plt.show()


