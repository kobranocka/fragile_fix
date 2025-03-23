import tensorflow as tf
from utils.data_loader import load_mnist_data
from models.lenet import build_lenet
from models.cnn_model import build_basic_cnn

def train_model(model_name='lenet', epochs=10, batch_size=128):
    """
    Trains the specified model (LeNet or CNN) on MNIST and returns training history.
    """
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

    # Choose the model to train
    if model_name == 'lenet':
        model = build_lenet()
    elif model_name == 'cnn':
        model = build_basic_cnn()
    else:
        raise ValueError("Invalid model name. Choose 'lenet' or 'cnn'.")

    # Train the model & save history
    history = model.fit(
        train_images, train_labels,
        epochs=epochs, batch_size=batch_size,
        validation_data=(test_images, test_labels)
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nTest accuracy ({model_name}): {test_acc:.4f}")

    return model, history  
