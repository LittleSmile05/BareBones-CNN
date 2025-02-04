import numpy as np
import cv2
import os
import kagglehub
from typing import List, Tuple, Optional
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

class Layer:
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update(self, learning_rate: float) -> None:
        pass

class ConvolutionLayer(Layer):
    def __init__(self, num_filters: int, filter_size: int, channels: int):
        self.num_filters = num_filters
        self.filter_size = filter_size
        scale = np.sqrt(2.0 / (channels * filter_size * filter_size))
        self.filters = np.random.normal(0, scale, (num_filters, channels, filter_size, filter_size))
        self.biases = np.zeros(num_filters)
        
    def _convolve(self, input_data: np.ndarray, filter: np.ndarray) -> np.ndarray:
        height, width = input_data.shape[0] - self.filter_size + 1, input_data.shape[1] - self.filter_size + 1
        output = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                output[i, j] = np.sum(
                    input_data[i:i+self.filter_size, j:j+self.filter_size] * filter
                )
        return output
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        
        self.output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        for b in range(batch_size):
            for f in range(self.num_filters):
                for c in range(channels):
                    self.output[b, f] += self._convolve(input_data[b, c], self.filters[f, c])
                self.output[b, f] += self.biases[f]
                    
        return self.output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        batch_size, channels = self.input.shape[0], self.input.shape[1]
        self.filter_gradients = np.zeros_like(self.filters)
        self.bias_gradients = np.zeros_like(self.biases)
        input_gradient = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for f in range(self.num_filters):
                self.bias_gradients[f] += np.sum(gradient[b, f])
                for c in range(channels):
                    self.filter_gradients[f, c] += self._convolve(
                        self.input[b, c],
                        gradient[b, f]
                    )
                    input_gradient[b, c] += self._full_convolve(
                        gradient[b, f],
                        self.filters[f, c]
                    )
        
        return input_gradient
    
    def _full_convolve(self, input_data: np.ndarray, filter: np.ndarray) -> np.ndarray:
        filter_flipped = np.flipud(np.fliplr(filter))
        height, width = input_data.shape[0] + self.filter_size - 1, input_data.shape[1] + self.filter_size - 1
        padded_input = np.pad(input_data, self.filter_size - 1)
        return self._convolve(padded_input, filter_flipped)
    
    def update(self, learning_rate: float) -> None:
        self.filters -= learning_rate * self.filter_gradients
        self.biases -= learning_rate * self.bias_gradients

class ReLU(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient * (self.input > 0)

class MaxPool(Layer):
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        output_height = height // self.pool_size
        output_width = width // self.pool_size
        
        self.output = np.zeros((batch_size, channels, output_height, output_width))
        self.max_indices = np.zeros_like(self.output, dtype=np.int32)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.pool_size
                        w_start = j * self.pool_size
                        pool_region = input_data[
                            b, c,
                            h_start:h_start + self.pool_size,
                            w_start:w_start + self.pool_size
                        ]
                        self.output[b, c, i, j] = np.max(pool_region)
                        self.max_indices[b, c, i, j] = np.argmax(pool_region)
        
        return self.output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = self.input.shape
        input_gradient = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(gradient.shape[2]):
                    for j in range(gradient.shape[3]):
                        h_start = i * self.pool_size
                        w_start = j * self.pool_size
                        max_idx = self.max_indices[b, c, i, j]
                        h_max = h_start + max_idx // self.pool_size
                        w_max = w_start + max_idx % self.pool_size
                        input_gradient[b, c, h_max, w_max] = gradient[b, c, i, j]
        
        return input_gradient

class Flatten(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.normal(0, scale, (input_size, output_size))
        self.biases = np.zeros(output_size)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weight_gradients = np.dot(self.input.T, gradient)
        self.bias_gradients = np.sum(gradient, axis=0)
        return np.dot(gradient, self.weights.T)
    
    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradients

class CNN:
    def __init__(self):
        self.layers: List[Layer] = []
        self.training_history = {'loss': [], 'val_loss': []}
        
    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradient: np.ndarray) -> None:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def update(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update(learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, 
              learning_rate: float, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> dict:
        
        num_batches = len(X) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            for i in tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{epochs}'):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                predictions = self.forward(batch_X)
                
                loss = np.mean((predictions - batch_y) ** 2)
                epoch_loss += loss
                
                gradient = 2 * (predictions - batch_y) / batch_size
                
                self.backward(gradient)
                self.update(learning_rate)
            
            avg_loss = epoch_loss / num_batches
            self.training_history['loss'].append(avg_loss)
            
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = np.mean((val_predictions - y_val) ** 2)
                self.training_history['val_loss'].append(val_loss)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
            
        return self.training_history

def preprocess_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  
    return img

def plot_training_history(history: dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_age_prediction_model() -> CNN:
    model = CNN()
    
    model.add(ConvolutionLayer(16, 3, 3))  
    model.add(ReLU())
    model.add(MaxPool())
    
    model.add(ConvolutionLayer(32, 3, 16))
    model.add(ReLU())
    model.add(MaxPool())
    
    model.add(ConvolutionLayer(64, 3, 32))
    model.add(ReLU())
    model.add(MaxPool())
    
    model.add(Flatten())
    model.add(Dense(64 * 14 * 14, 128))
    model.add(ReLU())
    model.add(Dense(128, 1))
    
    return model

if __name__ == "__main__":
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info("Downloading dataset...")
        dataset_path = kagglehub.dataset_download("jangedoo/utkface-new")
        dataset_dir = os.path.join(dataset_path, "UTKFace")
        
        target_size = (128, 128)
        num_samples = 1000  
        train_split = 0.8
        
        logger.info("Loading dataset...")
        X, y = [], []
        
        file_list = [f for f in os.listdir(dataset_dir) 
                    if f.endswith('.jpg')][:num_samples]
        
        for file_name in tqdm(file_list, desc="Processing images"):
            try:
                age = int(file_name.split("_")[0])
                img_path = os.path.join(dataset_dir, file_name)
                img = preprocess_image(img_path, target_size)
                X.append(img)
                y.append(age)
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y) / 100.0  
        
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Dataset loaded: {len(X)} images")
        logger.info(f"Training set: {len(X_train)} images")
        logger.info(f"Validation set: {len(X_val)} images")
        
        logger.info("Creating model...")
        model = create_age_prediction_model()
        
        logger.info("Starting training...")
        history = model.train(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            X_val=X_val,
            y_val=y_val
        )
        
        plot_training_history(history)
     
        logger.info("Saving model...")
        save_path = 'age_prediction_model.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {save_path}")
        
        logger.info("\nTesting predictions on validation set...")
        def predict_age(image_path: str) -> float:
            try:
                img = preprocess_image(image_path, target_size)
                img = np.expand_dims(img, axis=0)
                prediction = model.forward(img)[0]
                return prediction * 100.0  
            except Exception as e:
                logger.error(f"Error predicting age: {e}")
                return None
        
        num_test_samples = min(5, len(X_val))
        test_indices = np.random.choice(len(X_val), num_test_samples, replace=False)
        
        logger.info("\nSample predictions:")
        total_error = 0
        for idx in test_indices:
            img_path = os.path.join(dataset_dir, file_list[split_idx + idx])
            actual_age = int(file_list[split_idx + idx].split("_")[0])
            predicted_age = predict_age(img_path)
            
            if predicted_age is not None:
                error = abs(predicted_age - actual_age)
                total_error += error
                logger.info(f"Image: {file_list[split_idx + idx]}")
                logger.info(f"Predicted age: {predicted_age:.1f}")
                logger.info(f"Actual age: {actual_age}")
                logger.info(f"Error: {error:.1f} years\n")
        
        avg_error = total_error / num_test_samples
        logger.info(f"Average prediction error: {avg_error:.1f} years")
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
