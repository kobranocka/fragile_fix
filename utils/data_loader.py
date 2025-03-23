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
