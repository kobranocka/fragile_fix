import os
import tensorflow as tf

def load_mnist_data():
    """
    Loads the MNIST dataset using TensorFlow's Keras API.
    Returns:
      (train_images, train_labels), (test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values between 0 and 1
    train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

# # Test loading MNIST
# (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
# print(f"MNIST shapes: {train_images.shape}, {train_labels.shape}, {test_images.shape}, {test_labels.shape}")

import pickle
import numpy as np

def load_cifar_batch(filename):
    """
    Loads a single CIFAR batch file (e.g., data_batch_1) and returns
    the images (X) and labels (Y).
    """
    with open(filename, 'rb') as f:
        # The CIFAR-10 data is pickled with Python 2. 
        # encoding='bytes' to load it in Python 3.
        datadict = pickle.load(f, encoding='bytes')
        
        X = datadict[b'data']
        Y = datadict[b'labels']  
        
        # Reshape the data into 4D array: (num_samples, 3, 32, 32)
        X = X.reshape(len(X), 3, 32, 32).transpose(0, 2, 3, 1)
        
        return X, np.array(Y)

import os

def load_cifar10(root_folder):
    """
    Loads the entire CIFAR-10 dataset from the 'cifar-10-batches-py' folder.
    Returns:
      X_train, y_train, X_test, y_test
    """
    xs = []
    ys = []
    
    # Load each training batch
    for b in range(1, 6):
        f = os.path.join(root_folder, f'data_batch_{b}')
        X_batch, Y_batch = load_cifar_batch(f)
        xs.append(X_batch)
        ys.append(Y_batch)
    
    # Concatenate into a single array
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    
    # Load test batch
    test_f = os.path.join(root_folder, 'test_batch')
    X_test, y_test = load_cifar_batch(test_f)
    
    return X_train, y_train, X_test, y_test
