import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers.legacy import Adam
tf.config.run_functions_eagerly(True)

# Set float precision and random seeds
tf.keras.backend.set_floatx('float32')
np.random.seed(0)
tf.random.set_seed(0)

# Ground truth function S(q_T, x_a, x_b) = exp(-q_T^2 - x_a^2 - x_b^2)
def true_S(q_T, x_a, x_b):
    return np.exp(-q_T**2 - x_a**2 - x_b**2)

# Define DNNs NN_a(x_a, k) and NN_b(x_b, k)
def make_dnn(name):
    inputs = tf.keras.Input(shape=(2,), name=f"{name}_input")  # [x, k]
    x = tf.keras.layers.Dense(32, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(32, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(1, activation='softplus')(x)
    return tf.keras.Model(inputs, outputs, name=name)

NN_a = make_dnn("NN_a")
NN_b = make_dnn("NN_b")

# Integration grid for numerical convolution
n_k = 64
n_phi = 64
k_vals = tf.linspace(0.0, 10.0, n_k)
phi_vals = tf.linspace(0.0, 2*np.pi, n_phi)
k_grid, phi_grid = tf.meshgrid(k_vals, phi_vals, indexing='ij')

def compute_S_pred(q_T, x_a, x_b):
    k = tf.reshape(k_grid, [-1])
    phi = tf.reshape(phi_grid, [-1])

    input_a = tf.stack([tf.fill(k.shape, x_a), k], axis=1)
    val_a = NN_a(input_a)  # shape (n_k*n_phi, 1)

    k_prime = tf.sqrt(q_T**2 + k**2 - 2*q_T*k*tf.cos(phi))
    input_b = tf.stack([tf.fill(k.shape, x_b), k_prime], axis=1)
    val_b = NN_b(input_b)

    integrand = val_a * val_b * tf.reshape(k, (-1, 1))
    dk = 10.0 / n_k
    dphi = 2*np.pi / n_phi
    integral = tf.reduce_sum(integrand) * dk * dphi
    return tf.squeeze(integral)

# Generate synthetic training data
def generate_training_data(n_samples):
    x_a = np.random.uniform(0, 1, n_samples).astype(np.float32)
    x_b = np.random.uniform(0, 1, n_samples).astype(np.float32)
    q_T = np.random.uniform(0, 3, n_samples).astype(np.float32)
    S_vals = true_S(q_T, x_a, x_b).astype(np.float32)
    return x_a, x_b, q_T, S_vals


# Optimizer (use legacy Adam for M1/M2 compatibility)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)


# Training step using tf.function and tf.map_fn
@tf.function
def train_step(xa, xb, qt, true_vals):
    # with tf.GradientTape(persistent=True) as tape:
    with tf.GradientTape(persistent=True) as tape:
        inputs = tf.stack([qt, xa, xb], axis=1)  # shape (batch, 3)

        def wrapped_fn(x):
            return compute_S_pred(x[0], x[1], x[2])

        preds = tf.map_fn(wrapped_fn, inputs, fn_output_signature=tf.float32)
        loss = tf.reduce_mean((preds - true_vals) ** 2)

    grads_a = tape.gradient(loss, NN_a.trainable_weights)
    grads_b = tape.gradient(loss, NN_b.trainable_weights)
    opt.apply_gradients(zip(grads_a, NN_a.trainable_weights))
    opt.apply_gradients(zip(grads_b, NN_b.trainable_weights))
    return loss

# opt_a = tf.keras.optimizers.Adam(learning_rate=1e-3)
# opt_b = tf.keras.optimizers.Adam(learning_rate=1e-3)

# @tf.function
# def train_step(xa, xb, qt, true_vals):
#     with tf.GradientTape(persistent=True) as tape:
#         inputs = tf.stack([qt, xa, xb], axis=1)

#         def wrapped_fn(x):
#             return compute_S_pred(x[0], x[1], x[2])

#         preds = tf.map_fn(wrapped_fn, inputs, fn_output_signature=tf.float32)
#         loss = tf.reduce_mean((preds - true_vals) ** 2)

#     grads_a = tape.gradient(loss, NN_a.trainable_weights)
#     grads_b = tape.gradient(loss, NN_b.trainable_weights)

#     opt_a.apply_gradients(zip(grads_a, NN_a.trainable_weights))
#     opt_b.apply_gradients(zip(grads_b, NN_b.trainable_weights))

#     return loss


# # Optimizer (use legacy Adam for M1/M2 compatibility)
# opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Train model
n_epochs = 500
n_samples = 128
x_a_train, x_b_train, q_T_train, S_train = generate_training_data(n_samples)

for epoch in range(n_epochs):
    loss = train_step(x_a_train, x_b_train, q_T_train, S_train)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

# Save trained models
NN_a.save("NN_a_model.h5")
NN_b.save("NN_b_model.h5")
print("Models saved: 'NN_a_model', 'NN_b_model'")

# Test and compare
x_a_test, x_b_test, q_T_test, S_true_test = generate_training_data(10)
S_pred_test = [compute_S_pred(q, a, b).numpy() for q, a, b in zip(q_T_test, x_a_test, x_b_test)]

for i in range(10):
    print(f"x_a={x_a_test[i]:.2f}, x_b={x_b_test[i]:.2f}, q_T={q_T_test[i]:.2f} | True S={S_true_test[i]:.4f}, Pred S={S_pred_test[i]:.4f}")



'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam


# Set float precision and random seeds
tf.keras.backend.set_floatx('float32')
np.random.seed(0)
tf.random.set_seed(0)

# Ground truth function S(q_T, x_a, x_b) = exp(-q_T^2 - x_a^2 - x_b^2)
def true_S(q_T, x_a, x_b):
    return np.exp(-q_T**2 - x_a**2 - x_b**2)

# Simple NN_a(x_a, k)
def make_dnn(name):
    inputs = tf.keras.Input(shape=(2,), name=f"{name}_input")  # [x, k]
    x = tf.keras.layers.Dense(32, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(32, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(1, activation='softplus')(x)
    return tf.keras.Model(inputs, outputs, name=name)

NN_a = make_dnn("NN_a")
NN_b = make_dnn("NN_b")

# Integration grid for numerical convolution
n_k = 64
n_phi = 64
k_vals = tf.linspace(0.0, 10.0, n_k)
phi_vals = tf.linspace(0.0, 2*np.pi, n_phi)
k_grid, phi_grid = tf.meshgrid(k_vals, phi_vals, indexing='ij')

def compute_S_pred(q_T, x_a, x_b):
    k = tf.reshape(k_grid, [-1])
    phi = tf.reshape(phi_grid, [-1])

    # Inputs for NN_a
    input_a = tf.stack([tf.fill(k.shape, x_a), k], axis=1)
    val_a = NN_a(input_a)  # shape (n_k*n_phi, 1)

    # Compute |q - k| = sqrt(q_T^2 + k^2 - 2 q_T k cos φ)
    k_prime = tf.sqrt(q_T**2 + k**2 - 2*q_T*k*tf.cos(phi))
    input_b = tf.stack([tf.fill(k.shape, x_b), k_prime], axis=1)
    val_b = NN_b(input_b)

    integrand = val_a * val_b * tf.reshape(k, (-1, 1))
    dk = 10.0 / n_k
    dphi = 2*np.pi / n_phi
    integral = tf.reduce_sum(integrand) * dk * dphi
    return tf.squeeze(integral)

# Generate synthetic training data
def generate_training_data(n_samples):
    x_a = np.random.uniform(0, 1, n_samples).astype(np.float32)
    x_b = np.random.uniform(0, 1, n_samples).astype(np.float32)
    q_T = np.random.uniform(0, 3, n_samples).astype(np.float32)
    S_vals = true_S(q_T, x_a, x_b).astype(np.float32)
    return x_a, x_b, q_T, S_vals

# Training step
@tf.function
def train_step(xa, xb, qt, true_vals):
    with tf.GradientTape(persistent=True) as tape:
        inputs = tf.stack([qt, xa, xb], axis=1)  # shape (batch, 3)

        def wrapped_fn(x):
            return compute_S_pred(x[0], x[1], x[2])

        preds = tf.map_fn(wrapped_fn, inputs, fn_output_signature=tf.float32)
        loss = tf.reduce_mean((preds - true_vals) ** 2)

    grads_a = tape.gradient(loss, NN_a.trainable_weights)
    grads_b = tape.gradient(loss, NN_b.trainable_weights)
    opt.apply_gradients(zip(grads_a, NN_a.trainable_weights))
    opt.apply_gradients(zip(grads_b, NN_b.trainable_weights))
    return loss


# Optimizer
#opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
opt = Adam(learning_rate=1e-3)

# Train model
n_epochs = 500
n_samples = 128
x_a_train, x_b_train, q_T_train, S_train = generate_training_data(n_samples)

for epoch in range(n_epochs):
    loss = train_step(x_a_train, x_b_train, q_T_train, S_train)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

# Test and compare
x_a_test, x_b_test, q_T_test, S_true_test = generate_training_data(10)
S_pred_test = [compute_S_pred(q, a, b).numpy() for q, a, b in zip(q_T_test, x_a_test, x_b_test)]

for i in range(10):
    print(f"x_a={x_a_test[i]:.2f}, x_b={x_b_test[i]:.2f}, q_T={q_T_test[i]:.2f} | True S={S_true_test[i]:.4f}, Pred S={S_pred_test[i]:.4f}")
'''