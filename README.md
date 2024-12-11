# CNN-from-scratch
Convolutional Neural Network made from only numpy and used on MNSIT dataset
# Convolutional Neural Network (CNN) from Scratch with Numpy

This repository demonstrates a fully functional implementation of a Convolutional Neural Network (CNN) built from scratch using only **Numpy**. It includes key neural network layers, activation functions, loss functions, and utility scripts to preprocess data and train models. The repository aims to provide an educational exploration of the inner workings of deep learning, bypassing the use of high-level libraries like TensorFlow or PyTorch.

---

## Features

- **Layer Implementations**:
  - **Convolutional Layer**: Implements a 2D convolution operation.
  - **Dense (Fully Connected) Layer**: Implements a fully connected layer.
  - **Reshape Layer**: Changes the shape of tensors to transition between convolutional and dense layers.

- **Activation Functions**:
  - Sigmoid
  - Other activation functions included for flexibility.

- **Loss Functions**:
  - Binary Cross-Entropy Loss

- **Utilities**:
  - Data preprocessing for MNIST.
  - Training and prediction loops.

---

## File Structure

```
.
├── activation.py        # Activation functions
├── activations.py       # Extended activation functions
├── convolutional.py     # Convolutional layer implementation
├── dense.py             # Dense layer implementation
├── layer.py             # Base class for layers
├── losses.py            # Loss functions and their derivatives
├── mnist.py             # MNIST dataset loading utility
├── mnist_conv.py        # MNIST experiment script with convolutional layers
├── network.py           # Neural network training and inference functions
├── reshape.py           # Reshape layer implementation
├── trials.ipynb         # Notebook for experiments and visualizations
├── xor.py               # The Final file to start the training
└── README.md            # Project overview
```

---

## Model Architecture

The implemented CNN model processes the MNIST dataset to classify digits. Below is the architecture:

1. **Input**: Grayscale images of shape (1, 28, 28).
2. **Convolutional Layer**: 3x3 kernel, 5 filters.
3. **Sigmoid Activation**: Applies non-linearity.
4. **Reshape Layer**: Flattens the output of the convolutional layer for dense connections.
5. **Dense Layer**: Fully connected layer with 100 neurons.
6. **Sigmoid Activation**.
7. **Dense Layer**: Output layer with 2 neurons (binary classification).
8. **Sigmoid Activation**.

---

## Code Highlights

### Training the Network
```python
# Neural Network Architecture
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

# Training
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# Testing
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
```

---

## Running the Project

### Prerequisites
- Python 3.12+ (made on)
- Numpy = 2.2.0

### Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/AnantVerma-58/CNN-from-scratch.git
   cd CNN-from-scratch
   ```
2. Install dependencies (Numpy is required):
   ```bash
   pip install numpy==2.2.0
   ```
3. Run experiments:
   - To train the model:
     ```bash
     python xor.py
     ```
---

## Results

- The model is trained on a subset of the MNIST dataset (100 images per class) for binary classification (digits `0` and `1`).
- Accuracy improves over epochs, demonstrating the effectiveness of the CNN implementation.

  View the [interactive decision boundary plot](https://github.com/AnantVerma-58/CNN-from-scratch/docs/plot.html).


---

## Future Improvements

- Extend support for multiclass classification.
- Optimize the convolutional layer for better performance.
- Add GPU support for faster training.
- Implement additional activation and loss functions.

---

## Acknowledgments

This project is inspired by the desire to demystify the workings of Convolutional Neural Networks by building one from scratch. It is an educational exercise in understanding the foundations of deep learning.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
