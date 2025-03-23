import tensorflow as tf
from utils.data_loader import load_mnist_data
from models.lenet import build_lenet
from models.cnn_model import build_basic_cnn

def train_model(train_images, train_labels, test_images, test_labels, 
                input_shape=(28, 28, 1), num_classes=10, model_name='lenet', 
                epochs=10, batch_size=128):
    """
    Trains the specified model (LeNet or CNN) and returns training history.
    """
    # Choose the model to train, pass input_shape and num_classes so that the model matches the data.
    if model_name == 'lenet':
        model = build_lenet(input_shape=input_shape, num_classes=num_classes)
    elif model_name == 'cnn':
        model = build_basic_cnn(input_shape=input_shape, num_classes=num_classes)
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
