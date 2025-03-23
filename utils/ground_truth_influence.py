import tensorflow as tf
import numpy as np

def compute_ground_truth_influence(model, train_images, train_labels, test_sample, test_label, sample_index, fine_tune_epochs=1, batch_size=128):
    """
    Approximates the ground truth influence of a training sample by removing it
    from the training set, fine-tuning the model, and then measuring the change in
    test loss.
    
    Returns the difference in test loss: (loss_without_sample - original_loss)
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Evaluate original test loss
    test_sample_exp = tf.expand_dims(test_sample, axis=0)
    test_label_exp = tf.expand_dims(test_label, axis=0)
    original_loss = loss_fn(test_label_exp, model(test_sample_exp, training=False)).numpy()
    
    # Save original weights to restore later
    original_weights = model.get_weights()
    
    # Remove the training sample from the dataset
    train_images_new = np.delete(train_images, sample_index, axis=0)
    train_labels_new = np.delete(train_labels, sample_index, axis=0)
    
    # Clone the model (to avoid affecting the original)
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    model_copy.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Fine-tune on the reduced dataset
    model_copy.fit(train_images_new, train_labels_new, epochs=fine_tune_epochs, batch_size=batch_size, verbose=0)
    
    # Compute new test loss
    new_loss = loss_fn(test_label_exp, model_copy(test_sample_exp, training=False)).numpy()
    
    # Restore the original model weights if needed
    model.set_weights(original_weights)
    
    # Ground truth influence is the change in test loss
    influence_gt = new_loss - original_loss
    return influence_gt

def compute_ground_truth_influences(model, train_images, train_labels, test_sample, test_label, sample_indices, fine_tune_epochs=1, batch_size=128):
    """
    Computes the ground truth influence estimates for a list of training sample indices.
    """
    influences = []
    for idx in sample_indices:
        inf = compute_ground_truth_influence(model, train_images, train_labels, test_sample, test_label, idx,
                                               fine_tune_epochs=fine_tune_epochs, batch_size=batch_size)
        influences.append(inf)
    return np.array(influences)
