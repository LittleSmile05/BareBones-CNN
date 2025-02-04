# 📌 Convolutional Neural Network (CNN) from Scratch

## 🚀 Overview
This project implements a **Convolutional Neural Network (CNN) from scratch** using **NumPy** and **Python**—without relying on deep learning frameworks like TensorFlow or PyTorch. The goal is to understand how CNNs function at a fundamental level, including convolutional layers, pooling, activation functions, and backpropagation.

## 📜 Features
- ✅ **Custom implementation of CNN layers** (Convolution, Pooling, Fully Connected)
- ✅ **Forward & backward propagation** for training
- ✅ **Gradient descent optimization** for weight updates
- ✅ **Activation functions (ReLU)**
- ✅ **Support for age prediction from facial images**
- ✅ **Built-in data preprocessing and model evaluation**

## 📂 Project Structure
📦 cnn_from_scratch
 ┣ 📜 cnn_implementation.py   # Complete CNN implementation including:
   - Layer classes (Convolution, ReLU, MaxPool, Dense)
   - Data preprocessing
   - Model training and evaluation
   - Utility functions for visualization

## 🛠️ Installation
Ensure you have **Python 3.x** and install required dependencies:
```bash
pip install numpy opencv-python kagglehub tqdm matplotlib
```

## 📊 Usage
1. The script will automatically:
   - Download the UTKFace dataset using kagglehub
   - Preprocess the images to the required format
   - Train the CNN model on age prediction
   - Save the trained model
   - Display training progress and results

2. Run the implementation:
```bash
python cnn_implementation.py
```

## 🧠 Model Architecture
* **Input Layer**: Accepts RGB images (3 channels, 128x128 pixels)
* **Convolutional Layers**: Three layers with increasing filters (16, 32, 64)
* **ReLU Activation**: After each convolutional layer
* **Max Pooling**: Reduces spatial dimensions while preserving features
* **Fully Connected Layers**: Two dense layers (128 neurons, 1 output)
* **Output**: Predicted age value

## 📈 Features
* Automatic dataset download and preprocessing
* Real-time training progress visualization
* Model saving functionality
* Built-in age prediction testing
* Training history plotting

## 🚀 Future Improvements
* 🔹 Add data augmentation for better generalization
* 🔹 Implement additional activation functions
* 🔹 Add support for different types of classification tasks
* 🔹 Optimize convolutional operations for better performance

## 👨‍💻 Author
**Gunel** – *Software Engineer & AI Enthusiast*
