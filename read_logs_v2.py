# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:03:18 2024

@author: Vincent
"""
# import os
import tensorflow as tf
# from tensorflow.python.summary.summary_iterator import SummaryIterator
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
# import numpy as np

# Paths to your log directories
train_log_dir = "logs/fit/20241129-184246/train"
val_log_dir = "logs/fit/20241129-184246/validation"

#%%
def extract_tensor_metrics(log_dir, tags):
    """
    Extract metrics from TensorBoard log files, even if stored as tensors.
    Args:
        log_dir (str): Path to the log directory.
        tags (list): List of tags to extract (e.g., ['epoch_loss', 'epoch_psnr_metric']).
    Returns:
        dict: Dictionary with extracted metrics and their corresponding steps.
    """
    metrics = {tag: [] for tag in tags}
    steps = {tag: [] for tag in tags}

    # Load TensorBoard logs
    acc = EventAccumulator(log_dir)
    acc.Reload()

    # Extract data for each tag
    for tag in tags:
        if tag in acc.Tags()['tensors']:
            tensor_events = acc.Tensors(tag)
            metrics[tag] = [tf.make_ndarray(event.tensor_proto).item() for event in tensor_events]
            steps[tag] = [event.step for event in tensor_events]
        elif tag in acc.Tags()['scalars']:
            scalar_events = acc.Scalars(tag)
            metrics[tag] = [event.value for event in scalar_events]
            steps[tag] = [event.step for event in scalar_events]
        else:
            print(f"Tag '{tag}' not found in {log_dir}")

    return metrics, steps

def plot_metrics(train_metrics, val_metrics, train_steps, val_steps, metric_name):
    """
    Plot training and validation metrics over epochs.
    Args:
        train_metrics (list): List of training metric values.
        val_metrics (list): List of validation metric values.
        train_steps (list): List of training steps.
        val_steps (list): List of validation steps.
        metric_name (str): Name of the metric (e.g., 'PSNR', 'Loss').
    """
    plt.figure(figsize=(10, 6))
    if train_metrics:
        plt.plot(train_steps, train_metrics, label=f'Train {metric_name}', marker='o')
    if val_metrics:
        plt.plot(val_steps, val_metrics, label=f'Validation {metric_name}', marker='x')
    plt.xlabel("Epoch"), plt.ylabel(metric_name), plt.title(f"{metric_name} Over Epochs")
    plt.legend(), plt.grid()
    plt.show()

def main():
    # Tags to extract
    train_tags = ['epoch_loss', 'epoch_psnr_metric']
    val_tags = ['epoch_val_loss', 'epoch_val_psnr_metric']

    # Extract metrics from logs
    train_metrics, train_steps = extract_tensor_metrics(train_log_dir, train_tags)
    val_metrics, val_steps = extract_tensor_metrics(val_log_dir, val_tags)

    # Plot metrics
    plot_metrics(train_metrics.get('epoch_psnr_metric', []),
                 val_metrics.get('epoch_val_psnr_metric', []),
                 train_steps.get('epoch_psnr_metric', []),
                 val_steps.get('epoch_val_psnr_metric', []),
                 metric_name='PSNR')

    plot_metrics(train_metrics.get('epoch_loss', []),
                 val_metrics.get('epoch_val_loss', []),
                 train_steps.get('epoch_loss', []),
                 val_steps.get('epoch_val_loss', []),
                 metric_name='Loss')

if __name__ == "__main__":
    main()
