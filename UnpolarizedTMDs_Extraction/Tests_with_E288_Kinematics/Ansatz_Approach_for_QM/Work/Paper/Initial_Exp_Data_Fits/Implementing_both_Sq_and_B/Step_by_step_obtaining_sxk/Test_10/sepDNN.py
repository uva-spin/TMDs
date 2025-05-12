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

# Load the trained model for generating synthetic data
CSmodel = tf.keras.models.load_model('averaged_model.h5', 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss
    })

SqT_model = CSmodel.get_layer('SqT')
print("Loaded the averaged model")

def true_S(q_T, x_a, x_b):
    inputs = np.stack([q_T, x_a, x_b], axis=-1)
    return SqT_model.predict(inputs)

# Define DNNs NN_a(x_a, k) and NN_b(x_b, k)
def make_dnn(name):
    inputs = tf.keras.Input(shape=(2,), name=f"{name}_input")
    x = tf.keras.layers.Dense(64, activation='relu6')(inputs)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    outputs = tf.keras.layers.Dense(1, activation='softplus')(x)
    return tf.keras.Model(inputs, outputs, name=name)

NN_a = make_dnn("NN_a")
NN_b = make_dnn("NN_b")

# Integration grid
n_k = 64
n_phi = 64
k_vals = tf.linspace(0.0, 10.0, n_k)
phi_vals = tf.linspace(0.0, 2*np.pi, n_phi)
k_grid, phi_grid = tf.meshgrid(k_vals, phi_vals, indexing='ij')

# Custom layer that mimics compute_S_pred
class ComputeSPredLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        q_T, x_a, x_b = inputs[:, 0], inputs[:, 1], inputs[:, 2]

        k = tf.reshape(k_grid, [-1])
        phi = tf.reshape(phi_grid, [-1])

        input_a = tf.stack([tf.repeat(x_a, n_k * n_phi), tf.tile(k, [tf.shape(x_a)[0]])], axis=1)
        val_a = NN_a(input_a)

        q_T_rep = tf.repeat(q_T, n_k * n_phi)
        x_b_rep = tf.repeat(x_b, n_k * n_phi)
        k_tiled = tf.tile(k, [tf.shape(q_T)[0]])
        phi_tiled = tf.tile(phi, [tf.shape(q_T)[0]])

        k_prime = tf.sqrt(q_T_rep**2 + k_tiled**2 - 2*q_T_rep*k_tiled*tf.cos(phi_tiled))
        input_b = tf.stack([x_b_rep, k_prime], axis=1)
        val_b = NN_b(input_b)

        integrand = val_a * val_b * tf.reshape(k_tiled, (-1, 1))
        integrand = tf.reshape(integrand, (tf.shape(q_T)[0], n_k * n_phi))
        integral = tf.reduce_sum(integrand, axis=1) * (10.0 / n_k) * (2*np.pi / n_phi)

        return tf.reshape(integral, (-1, 1))

def createModel_S_pred():
    input_layer = tf.keras.Input(shape=(3,), name="qT_xa_xb")
    output_layer = ComputeSPredLayer()(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="compute_S_pred_model")
    return model

# Generate training data
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
        preds = createModel_S_pred()(inputs, training=True)
        loss = tf.reduce_mean((preds - tf.reshape(true_vals, (-1, 1))) ** 2)

    grads_a = tape.gradient(loss, NN_a.trainable_weights)
    grads_b = tape.gradient(loss, NN_b.trainable_weights)
    opt_a.apply_gradients(zip(grads_a, NN_a.trainable_weights))
    opt_b.apply_gradients(zip(grads_b, NN_b.trainable_weights))

    return loss

# Train the model
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
createModel_S_pred().save("SqT_pred_model.h5")
print("Models saved: 'NN_a_model', 'NN_b_model', 'SqT_pred_model'")

finishtime = datetime.datetime.now().replace(microsecond=0)
totalduration = finishtime - starttime
print(f"Total duration --> {totalduration}")

# Evaluate the model
x_a_test, x_b_test, q_T_test, S_true_test = generate_training_data(300)
inputs_test = np.stack([q_T_test, x_a_test, x_b_test], axis=-1)
S_pred_test = createModel_S_pred().predict(inputs_test).flatten()

# Save predictions
df = pd.DataFrame({
    'x1': x_a_test.flatten(),
    'x2': x_b_test.flatten(),
    'qT': q_T_test.flatten(),
    'SqT_true': S_true_test.flatten(),
    'SqT_pred': S_pred_test.flatten()
})
df.to_csv(str(CSV_FOLDER)+'/SqT_comp.csv', index=False)
