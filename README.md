# FSRCNN and Enhanced Models for Super-Resolution

This repository provides the implementation of the FSRCNN (Fast Super-Resolution Convolutional Neural Network) as described in the original paper, along with enhanced versions incorporating residual layers and additional regularization techniques. The project includes scripts for training, validation, and inference.

---

## Scripts Overview

### **1. `FSRCNN.py`**
This script contains the implementation of the FSRCNN model based on the original paper.  
Key features:
- Implements `PReLU` with the `shared_axes` option to allow dynamic input sizes, making the model compatible with images of varying dimensions.
- Provides a robust baseline for super-resolution tasks using the original architecture.

### **2. `FSRCNN_ResLayers.py`**
This enhanced version of FSRCNN introduces residual blocks within the mapping layers.  
Key improvements:
- Adds residual connections to enhance feature propagation and network stability.
- Incorporates L2 normalization to regularize the network and improve convergence.

### **3. `FSRCNN_ResLayers_v2.py`**
This script builds upon the enhancements in `FSRCNN_ResLayers.py` by adding a residual connection between the shrinking layer and the feature extraction layer.  
Objective:
- This architecture aims to retain critical image features throughout the network to improve texture reconstruction and overall super-resolution quality.

---

## Inference Scripts

### **`inference_improved_model.py`**
This script loads a pre-trained model and performs inference on validation images.  
Workflow:
- Processes validation images to generate super-resolution (SR) outputs.
- Compares SR images against:
  1. The ground truth (original high-resolution image).
  2. Bicubic interpolation results.
  3. The downsampled input image.
- Displays and logs PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure) for both SR and bicubic images, offering quantitative evaluation.

### **`simple_inferences.py`**
A simplified version of the inference script for quick testing.  
Functionality:
- Accepts a single image and generates its SR version using both the super-resolution model and bicubic interpolation.
- Ideal for demonstration purposes or quick evaluations of the trained model.

---

## Features and Highlights

1. **Dynamic Input Support**: 
   By implementing `shared_axes` in `PReLU`, the models can handle variable input dimensions, making them versatile for real-world applications.

2. **Residual Connections**: 
   Enhanced models improve feature propagation and stability, leveraging the benefits of residual learning.

3. **L2 Normalization**: 
   Regularization ensures smoother convergence and improved generalization for better SR image quality.

4. **Performance Metrics**: 
   The scripts compute PSNR and SSIM, standard metrics for evaluating image quality, aligning results with human perception.

---

## Datasets

The following datasets were used for training and validation:
- **Urban100**: A dataset of diverse urban scenes with high-frequency details.
- **General100**: A dataset of 100 general-purpose high-resolution images.
- **Set5** and **Set15**: Popular benchmark datasets used for evaluating super-resolution models.

---

## Getting Started

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Additional dependencies: numpy, matplotlib, pillow, scipy

### Training
- Use any of the `FSRCNN.py`, `FSRCNN_ResLayers.py`, or `FSRCNN_ResLayers_v2.py` scripts to train the models with your dataset.
- Models are optimized for 3X upscaling, but the architecture supports other scaling factors with minor adjustments.

### Inference
- Run `inference_improved_model.py` for a full evaluation on validation datasets.
- Use `simple_inferences.py` for quick tests on individual images.

---

## References

This repository is based on the FSRCNN model described in the paper:  
_Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. "Accelerating the Super-Resolution Convolutional Neural Network." In ECCV 2016._  
[Link to paper](https://arxiv.org/abs/1608.00367)
