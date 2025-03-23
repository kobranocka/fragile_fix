import tensorflow as tf
import numpy as np

def compute_gradients(model, loss_fn, inputs, labels):
    """
    Computes and returns the flattened gradient of the loss with respect to model parameters.
    Any None gradients are replaced with zeros.
    """
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=False)
        loss = loss_fn(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def hvp(model, inputs, labels, loss_fn, vector):
    """
    Computes the Hessian-vector product (H * vector) using nested gradient tapes.
    This function avoids explicitly forming the Hessian.
    """
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape() as tape1:
            predictions = model(inputs, training=False)
            loss = loss_fn(labels, predictions)
        grads = tape1.gradient(loss, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
        grad_vector = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

    grad_dot = tf.reduce_sum(grad_vector * vector)
    hvps = tape2.gradient(grad_dot, model.trainable_variables)
    hvps = [tf.zeros_like(v) if h is None else h for h, v in zip(hvps, model.trainable_variables)]
    return tf.concat([tf.reshape(h, [-1]) for h in hvps], axis=0)

def conjugate_gradient_solver(model, inputs, labels, loss_fn, b, damping=0.01, max_iter=100, tol=1e-5):
    """
    Solves (H + damping * I)x = b using the Conjugate Gradient method, where H is 
    implicitly defined via Hessian-vector products (HVPs).
    """
    x = tf.zeros_like(b)
    r = b - hvp(model, inputs, labels, loss_fn, x) - damping * x
    p = r
    rs_old = tf.reduce_sum(r * r)

    for _ in range(max_iter):
        Ap = hvp(model, inputs, labels, loss_fn, p) + damping * p
        alpha = rs_old / tf.reduce_sum(p * Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = tf.reduce_sum(r * r)
        if tf.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

def precompute_test_ihvp(model, test_sample, test_label, loss_fn, damping=0.01):
    """
    Precomputes the inverse-Hessian vector product (IHVP) for the test sample:
        H^{-1} * grad(loss(test))
    using the conjugate gradient solver.
    """
    test_sample = tf.expand_dims(test_sample, axis=0)
    test_label = tf.expand_dims(test_label, axis=0)
    grad_test = compute_gradients(model, loss_fn, test_sample, test_label)
    return conjugate_gradient_solver(model, test_sample, test_label, loss_fn, grad_test, damping)

def compute_influence_function(model, train_sample, train_label, ihvp, loss_fn):
    """
    Computes the influence score for a single training sample as:
        I(z) = - grad(train)^T * (H^{-1} * grad(loss(test)))
    
    Note: The negative sign implies that a beneficial training point (one that reduces the loss)
    will have a negative influence estimate. If your ground truth influence is defined oppositely,
    you may choose to flip the sign.
    """
    train_sample = tf.expand_dims(train_sample, axis=0)
    train_label = tf.expand_dims(train_label, axis=0)
    grad_train = compute_gradients(model, loss_fn, train_sample, train_label)
    return -tf.tensordot(grad_train, ihvp, axes=1).numpy()

def compute_influence_on_dataset(model, train_images, train_labels, test_sample, test_label,
                                 loss_fn, damping=0.01, sample_indices=None):
    """
    Computes the standard influence estimates for a subset of training samples.
    """
    if sample_indices is None:
        sample_indices = range(len(train_images))
    ihvp = precompute_test_ihvp(model, test_sample, test_label, loss_fn, damping)
    influences = [
        compute_influence_function(model, train_images[i], train_labels[i], ihvp, loss_fn)
        for i in sample_indices
    ]
    return np.array(influences)

def compute_influence_and_meta_on_sample(model, train_sample, train_label, ihvp, loss_fn, epsilon=1e-3):
    """
    Computes both the standard influence and the meta influence for a single training sample.
    
    Standard influence:
        I(z) = - grad(z)^T * (H^{-1} * grad(loss(test)))
    
    Meta influence (approximated via finite differences):
        M(z) ≈ (I(z + perturbation) - I(z)) / epsilon
    """
    train_sample_exp = tf.expand_dims(train_sample, axis=0)
    train_label_exp = tf.expand_dims(train_label, axis=0)

    # Original influence
    grad = compute_gradients(model, loss_fn, train_sample_exp, train_label_exp)
    influence = -tf.tensordot(grad, ihvp, axes=1)

    # Perturbed influence
    noise = tf.random.normal(shape=tf.shape(train_sample), stddev=epsilon)
    perturbed_sample = tf.expand_dims(train_sample + noise, axis=0)
    grad_perturbed = compute_gradients(model, loss_fn, perturbed_sample, train_label_exp)
    influence_perturbed = -tf.tensordot(grad_perturbed, ihvp, axes=1)

    # Meta influence as the finite-difference quotient
    meta = (influence_perturbed - influence) / epsilon
    return influence.numpy(), meta.numpy()

def compute_influence_and_meta_on_dataset(model, train_images, train_labels, test_sample, test_label,
                                          loss_fn, damping=0.01, sample_indices=None, epsilon=1e-3):
    """
    Computes both standard influence and meta influence scores for a subset of training samples.
    """
    if sample_indices is None:
        sample_indices = range(len(train_images))
    ihvp = precompute_test_ihvp(model, test_sample, test_label, loss_fn, damping)
    
    std_infs, meta_infs = [], []
    for i in sample_indices:
        std, meta = compute_influence_and_meta_on_sample(model, train_images[i], train_labels[i],
                                                         ihvp, loss_fn, epsilon)
        std_infs.append(std)
        meta_infs.append(meta)
    return np.array(std_infs), np.array(meta_infs)
