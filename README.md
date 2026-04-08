# MLP Digit Recognizer (CUDA Accelerated)

A high-performance implementation of a Multi-Layer Perceptron (MLP) designed to recognize handwritten digits from the **MNIST** database. This project leverages **NVIDIA CUDA** to offload heavy mathematical computations to the GPU.

## Key Features
* **GPU Acceleration:** All core neural network operations (Forward Pass, Backward Pass, Weight Updates) are implemented as custom CUDA kernels.
* **Massive Parallelism:** By processing neurons and weights simultaneously across thousands of GPU cores, the training and inference speeds are significantly higher than CPU-only versions.
* **Image Processing Pipeline:** Includes an OpenCV-based preprocessing tool to handle custom `.jpg` files (binarization, auto-cropping, and resizing to 28x28).

## Tech Stack
* **C++**: Core application logic and object-oriented structure.
* **CUDA**: Parallel computing platform for GPU acceleration.
* **OpenCV**: Image loading and preprocessing (Thresholding, Resizing, Morphology).
* **Eigen**: Linear algebra library for efficient data handling.

## Network Architecture
Optimized for the **MNIST** dataset:
- **Input Layer:** 784 neurons (28x28 flattened pixels).
- **Hidden Layers:** Fully configurable (e.g., 128 -> 64).
- **Output Layer:** 10 neurons (representing digits 0-9).
- **Activation Function:** **Leaky ReLU** (to prevent the "dying ReLU" problem during training).

## How It Works

### Preprocessing (`testOwnJpg`)
Custom images are processed to match the MNIST format:
- **Otsu's Binarization:** Automatically separates the digit from the background.
- **Inversion:** Ensures the digit is white on a black background.
- **Bounding Box & Padding:** Crops the digit and adds a margin to improve recognition accuracy.
- **Normalization:** Scales pixel values to the range [0.0, 1.0].

### Inference (`predict`)
The normalized vector is copied from **RAM (Host)** to **VRAM (Device)**. The GPU then executes the forward pass through all layers. The final result is determined by finding the output neuron with the highest activation value (Argmax).

### Training (Backpropagation)
The network learns by calculating the error between the prediction and the ground truth. The `backwardLayer` and `updateWeights` kernels perform Gradient Descent in parallel, adjusting thousands of weights in a single GPU cycle.

## Requirements
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- OpenCV 4.x.
- C++17 compatible compiler.

---
