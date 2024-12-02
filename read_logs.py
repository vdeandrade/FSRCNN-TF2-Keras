# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:07:05 2024

@author: Vincent
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.framework import tensor_pb2
from time import sleep

#%%
print("\033[H\033[J"); sleep(0.1) # clear the ipython console before starting the code.

#%%
# Path to the base log directory
log_dir = r"D:\vdeandrade\Deep_learning\GitHub_repos\FSRCNN---Keras\logs\fit\20241129-184246"


#%%
# def plot_metrics_from_logs(log_dir):
"""
Reads TensorBoard logs from 'train' and 'validation' directories
and plots metrics like PSNR and Loss.

Args:
    log_dir (str): Path to the base log directory (e.g., 'logs/fit/...')
    
    run in the main directorythe command: tensorboard --logdir logs/fit/etc
"""
# Paths to training and validation log subdirectories
train_log_dir = os.path.join(log_dir, 'train')
val_log_dir = os.path.join(log_dir, 'validation')


# Initialize event accumulators for train and validation logs
train_acc = EventAccumulator(train_log_dir)
val_acc = EventAccumulator(val_log_dir)


#%%
train_acc.Reload()
val_acc.Reload()
print("Available tags:", train_acc.Tags())

# Print available tags
print("Train log tags:", train_acc.Tags()['scalars'])
print("Validation log tags:", val_acc.Tags()['scalars'])

train_tag_dict = train_acc.Tags()
list_train_tags = train_tag_dict['tensors']
print("Available tensors:", train_tag_dict['tensors'])

#%%
# for tag in train_tag_dict['tensors']:
#     print(f"Tag: {tag}")
#     tensor_events = train_acc.Tensors(tag)
    # for event in tensor_events:
        # print(f"Step: {event.step}, Value: {event.tensor_proto}")

for tag in train_tag_dict['tensors']:
    print(f"Tag: {tag}")


# Extract tensor data for 'epoch_loss' and 'epoch_psnr_metric'
if 'epoch_loss' in train_acc.Tags()['tensors']:
    tensor_events_loss = train_acc.Tensors('epoch_loss')
    train_loss = [tf.make_ndarray(event.tensor_proto).item() for event in tensor_events_loss]
    epochs_train = [event.step for event in tensor_events_loss]

if 'epoch_psnr_metric' in train_acc.Tags()['tensors']:
    tensor_events_psnr = train_acc.Tensors('epoch_psnr_metric')
    train_psnr = [tf.make_ndarray(event.tensor_proto).item() for event in tensor_events_psnr]

# if 0:
#     tag_loss = 'epoch_loss'
#     tag_psnr = 'epoch_psnr_metric'
#     tensor_events = train_acc.Tensors(tag_loss)
#     for event in tensor_events:
#         # Decode the tensor_proto into a NumPy array
#         tensor_loss_value = tf.make_ndarray(event.tensor_proto)
#     tensor_events = train_acc.Tensors(tag_psnr)
#     for event in tensor_events:
#         # Decode the tensor_proto into a NumPy array
#         tensor_psnr_value = tf.make_ndarray(event.tensor_proto)

# #%%
# # Initialize lists for loss and PSNR
# epoch_losses = []
# epoch_psnrs = []

# # Extract epoch_loss
# tag_loss = 'epoch_loss'
# tensor_events_loss = train_acc.Tensors(tag_loss)
# for event in tensor_events_loss:
#     tensor_loss_value = tf.make_ndarray(event.tensor_proto)
#     epoch_losses.append((event.step, tensor_loss_value))  # Store (step, value)

# # Extract epoch_psnr_metric
# tag_psnr = 'epoch_psnr_metric'
# tensor_events_psnr = train_acc.Tensors(tag_psnr)
# for event in tensor_events_psnr:
#     tensor_psnr_value = tf.make_ndarray(event.tensor_proto)
#     epoch_psnrs.append((event.step, tensor_psnr_value))  # Store (step, value)

# # Print results
# print("Epoch losses:", epoch_losses)
# print("Epoch PSNRs:", epoch_psnrs)


# #%%

# # Unzip step and value pairs
# loss_steps, loss_values = zip(*epoch_losses)
# psnr_steps, psnr_values = zip(*epoch_psnrs)

# # Plot Loss
# plt.figure(figsize=(10, 5))
# plt.plot(loss_steps, loss_values, label="Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Over Epochs")
# plt.legend()
# plt.grid()

# # Plot PSNR
# plt.figure(figsize=(10, 5))
# plt.plot(psnr_steps, psnr_values, label="PSNR")
# plt.xlabel("Epoch")
# plt.ylabel("PSNR (dB)")
# plt.title("Training PSNR Over Epochs")
# plt.legend()
# plt.grid()

# plt.show()


