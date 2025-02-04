# 📌 Convolutional Neural Network (CNN) from Scratch

## 🚀 Overview
This project implements a **Convolutional Neural Network (CNN) from scratch** using **NumPy** and **Python**—without relying on deep learning frameworks like TensorFlow or PyTorch. The goal is to understand how CNNs function at a fundamental level, including convolutional layers, pooling, activation functions, and backpropagation.

## 📜 Features
- ✅ **Custom implementation of CNN layers** (Convolution, Pooling, Fully Connected)
- ✅ **Forward & backward propagation** for training
- ✅ **Gradient descent optimization** for weight updates
- ✅ **Activation functions (ReLU, Softmax, etc.)**
- ✅ **Support for image classification tasks**

## 📂 Project Structure
📦 cnn_from_scratch
 ┣ 📜 cnn_from_scratch.py   # Main implementation
 ┣ 📜 dataset_loader.py     # Loads and preprocesses image data
 ┣ 📜 train.py              # Trains the CNN on a dataset
 ┣ 📜 evaluate.py           # Tests and evaluates the trained model
 ┣ 📜 README.md             # Project documentation

## 🛠️ Installation
Ensure you have **Python 3.x** and install dependencies:
```bash
pip install numpy matplotlib
```

## 📊 Usage
1. Prepare your dataset and ensure it's formatted correctly.
2. Modify `train.py` to specify dataset paths and hyperparameters.
3. Train the model:
```bash
python train.py
```
4. Evaluate the model on test data:
```bash
python evaluate.py
```

## 🧠 Understanding the Model
* **Convolutional Layer**: Extracts spatial features using filters.
* **Pooling Layer**: Reduces dimensionality while retaining key information.
* **Fully Connected Layer**: Makes final predictions based on extracted features.
* **Backpropagation**: Computes gradients and updates weights.

## 📈 Example Results
After training on a sample dataset, the model achieves **XX% accuracy** on the test set.

## 🚀 Future Improvements
* 🔹 Add batch normalization for faster convergence
* 🔹 Implement dropout to prevent overfitting
* 🔹 Expand dataset support and preprocessing methods

## 👨‍💻 Author
**Gunel** – *Software Engineer & AI Enthusiast*
