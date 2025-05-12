import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

# from tensorflow.keras.optimizers.legacy import Adam
tf.config.run_functions_eagerly(True)

# Set float precision and random seeds
tf.keras.backend.set_floatx('float32')
np.random.seed(0)
tf.random.set_seed(0)

CSV_FOLDER = "csvs"
os.makedirs(CSV_FOLDER, exist_ok=True)


# Define custom loss function and class
def custom_weighted_loss(y_true, y_pred, w=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if w is not None:
        w = tf.cast(w, tf.float32)
        mean_w = tf.reduce_mean(w)
        weights = w / mean_w
        squared_error = tf.square(y_pred - y_true)
        weighted_squared_error = squared_error * weights
        return tf.reduce_mean(weighted_squared_error)
    else:
        return tf.reduce_mean(tf.square(y_pred - y_true))

class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_weighted_loss"):
        super().__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return custom_weighted_loss(y_true, y_pred)

# Load the trained model and extract the submodel
CSmodel = tf.keras.models.load_model('averaged_model.h5', 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss
    })

SqT_model = CSmodel.get_layer('SqT')
print("Loaded the averaged model")

# New true_S definition using the model
def true_S(q_T, x_a, x_b):
    # Stack inputs and ensure shape is (N, 3)
    inputs = np.stack([q_T, x_a, x_b], axis=-1)
    return SqT_model.predict(inputs)


# Define DNNs NN_a(x_a, k) and NN_b(x_b, k)
def make_dnn(name):
    inputs = tf.keras.Input(shape=(2,), name=f"{name}_input")  # [x, k]
    x = tf.keras.layers.Dense(64, activation='tanh')(inputs)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
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


opt_a = tf.keras.optimizers.Adam(learning_rate=1e-3)
opt_b = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(xa, xb, qt, true_vals):
    with tf.GradientTape(persistent=True) as tape:
        inputs = tf.stack([qt, xa, xb], axis=1)

        def wrapped_fn(x):
            return compute_S_pred(x[0], x[1], x[2])

        preds = tf.map_fn(wrapped_fn, inputs, fn_output_signature=tf.float32)
        loss = tf.reduce_mean((preds - true_vals) ** 2)

    grads_a = tape.gradient(loss, NN_a.trainable_weights)
    grads_b = tape.gradient(loss, NN_b.trainable_weights)

    opt_a.apply_gradients(zip(grads_a, NN_a.trainable_weights))
    opt_b.apply_gradients(zip(grads_b, NN_b.trainable_weights))

    return loss


# Train model
n_epochs = 4000
n_samples = 300
x_a_train, x_b_train, q_T_train, S_train = generate_training_data(n_samples)


starttime = datetime.datetime.now().replace(microsecond=0)
for epoch in range(n_epochs):
    loss = train_step(x_a_train, x_b_train, q_T_train, S_train)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
        time50epochs = datetime.datetime.now().replace(microsecond=0)
        duration_for_50_epochs = time50epochs - starttime
        print(f"Duration at Epoch {epoch} --> {duration_for_50_epochs}")

# Save trained models
NN_a.save("NN_a_model.h5")
NN_b.save("NN_b_model.h5")
print("Models saved: 'NN_a_model', 'NN_b_model'")

finishtime = datetime.datetime.now().replace(microsecond=0)
totalduration = finishtime - starttime
print(f"Total duration --> {totalduration}")

# Generate 100 test data points
x_a_test, x_b_test, q_T_test, S_true_test = generate_training_data(100)

# Ensure all are 1D arrays
x_a_test = np.asarray(x_a_test).flatten()
x_b_test = np.asarray(x_b_test).flatten()
q_T_test = np.asarray(q_T_test).flatten()
S_true_test = np.asarray(S_true_test).flatten()

# Flatten predictions
S_pred_test = [compute_S_pred(q, a, b).numpy().item() for q, a, b in zip(q_T_test, x_a_test, x_b_test)]
S_pred_test = np.asarray(S_pred_test).flatten()

# Now create the DataFrame
df = pd.DataFrame({
    'x1': x_a_test,
    'x2': x_b_test,
    'qT': q_T_test,
    'SqT_true': S_true_test,
    'SqT_pred': S_pred_test
})

# Save to CSV
df.to_csv(str(CSV_FOLDER)+'/SqT_comp.csv', index=False)
print("Saved SqT_comp.csv with 100 data points.")
