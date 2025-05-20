import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import sys
import re

# Set float precision and random seeds
tf.keras.backend.set_floatx('float32')
np.random.seed(42)
tf.random.set_seed(42)
tf.config.run_functions_eagerly(True)


# Directory setup
scratch_path = '/scratch/cee9hc/Unpolarized_TMDs/with_E288_E605/Trial_01/'
models_folder = str(scratch_path) + '/Models'
CSV_FOLDER = str(scratch_path) + '/csvs'
nna_models = str(scratch_path) + '/nna_models'
nnb_models = str(scratch_path) + '/nnb_models'
SqT_models = str(scratch_path) + '/SqT_models'
plots_folder = str(scratch_path) + '/training_plots'

# Create directories if they don't exist
os.makedirs(CSV_FOLDER, exist_ok=True)
os.makedirs(nna_models, exist_ok=True)
os.makedirs(nnb_models, exist_ok=True)
os.makedirs(SqT_models, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)


# Train parameters
n_epochs = 500
n_train_samples = 1000
n_val_samples = 200
batch_size = 32

# Integration grid
n_k = 64
n_phi = 64
k_vals = tf.linspace(0.0, 10.0, n_k)
phi_vals = tf.linspace(0.0, 2*np.pi, n_phi)
k_grid, phi_grid = tf.meshgrid(k_vals, phi_vals, indexing='ij')

# Define DNNs NN_a(x_a, k) and NN_b(x_b, k)
def make_dnn(name):
    inputs = tf.keras.Input(shape=(2,), name=f"{name}_input")
    x = tf.keras.layers.Dense(64, activation='relu6')(inputs)
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    x = tf.keras.layers.Dense(64, activation='relu6')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs, outputs, name=name)

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

# Load the specified model
if len(sys.argv) < 2:
    raise ValueError("Usage: sys.argv < 2 ")
model_number = sys.argv[1]
print(f"Loading model from: {model_number}")


# Extract model index from filename
model_filename = os.path.basename(model_number)
match = re.search(r"model_(\d+)", model_filename)
model_idx = match.group(1) if match else "single"

print(f"\nProcessing model {model_number}")

# Load the model with custom objects
selected_SqT_model = tf.keras.models.load_model(model_number, 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss
    })

def true_S(model, q_T, x_a, x_b):
    inputs = np.stack([q_T, x_a, x_b], axis=-1)
    return model.predict(inputs)

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
def generate_kinematics(n_samples):
    x_a = np.random.uniform(0, 1, n_samples).astype(np.float32)
    x_b = np.random.uniform(0, 1, n_samples).astype(np.float32)
    q_T = np.random.uniform(0, 5, n_samples).astype(np.float32)
    return x_a, x_b, q_T

# Generate training data
def generate_training_data(model, x_a, x_b, q_T):
    S_vals = true_S(model, q_T, x_a, x_b).astype(np.float32)
    return S_vals



# Generate training and validation data
x_a_train, x_b_train, q_T_train = generate_kinematics(n_train_samples)
x_a_val, x_b_val, q_T_val = generate_kinematics(n_val_samples)

S_train = generate_training_data(selected_SqT_model, x_a_train, x_b_train, q_T_train)
S_val = generate_training_data(selected_SqT_model, x_a_val, x_b_val, q_T_val)

# Define neural networks
NN_a = make_dnn("NN_a")
NN_b = make_dnn("NN_b")

# Optimizers
opt_a = tf.keras.optimizers.Adam(learning_rate=1e-3)
opt_b = tf.keras.optimizers.Adam(learning_rate=1e-3)


S_pred_model = createModel_S_pred()

# Define training step
@tf.function
def train_step(xa, xb, qt, true_vals):
    with tf.GradientTape(persistent=True) as tape:
        inputs = tf.stack([qt, xa, xb], axis=1)
        preds = S_pred_model(inputs, training=True)
        loss = tf.reduce_mean((preds - tf.reshape(true_vals, (-1, 1))) ** 2)

    grads_a = tape.gradient(loss, NN_a.trainable_weights)
    grads_b = tape.gradient(loss, NN_b.trainable_weights)
    opt_a.apply_gradients(zip(grads_a, NN_a.trainable_weights))
    opt_b.apply_gradients(zip(grads_b, NN_b.trainable_weights))

    return loss

# Function to evaluate on validation set
def evaluate(xa, xb, qt, true_vals):
    inputs = tf.stack([qt, xa, xb], axis=1)
    preds = S_pred_model(inputs, training=False)
    loss = tf.reduce_mean((preds - tf.reshape(true_vals, (-1, 1))) ** 2)
    return loss.numpy()

# Lists to store losses for plotting
train_losses = []
val_losses = []

starttime = datetime.datetime.now().replace(microsecond=0)

# # Training loop
# for epoch in range(n_epochs):
#     # Training phase
#     epoch_loss = 0.0
#     for i in range(0, n_train_samples, batch_size):
#         xa_batch = x_a_train[i:i+batch_size]
#         xb_batch = x_b_train[i:i+batch_size]
#         qt_batch = q_T_train[i:i+batch_size]
#         S_batch = S_train[i:i+batch_size]
#         batch_loss = train_step(xa_batch, xb_batch, qt_batch, S_batch)

#         # Stop early if NaN is encountered in batch loss
#         if tf.math.is_nan(batch_loss):
#             print(f"NaN encountered in training at epoch {epoch}, batch {i // batch_size}. Aborting training.")
#             break

#         epoch_loss += batch_loss.numpy() * len(xa_batch)

#     # Finalize epoch loss
#     epoch_loss /= n_train_samples

#     # Check if training loss is NaN after epoch
#     if np.isnan(epoch_loss):
#         print(f"NaN detected in epoch {epoch} training loss. Stopping training.")
#         break

#     train_losses.append(epoch_loss)

#     # Validation phase
#     val_loss = evaluate(x_a_val, x_b_val, q_T_val, S_val)
    
#     # Stop early if val loss is NaN
#     if np.isnan(val_loss):
#         print(f"NaN detected in epoch {epoch} validation loss. Stopping training.")
#         break

#     val_losses.append(val_loss)

#     # Print update every 50 epochs
#     if epoch % 50 == 0:
#         time_current = datetime.datetime.now().replace(microsecond=0)
#         duration = time_current - starttime
#         print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
#         print(f"Duration at Epoch {epoch} --> {duration}")


# Training loop
for epoch in range(n_epochs):
    # Training phase
    epoch_loss = 0.0
    for i in range(0, n_train_samples, batch_size):
        xa_batch = x_a_train[i:i+batch_size]
        xb_batch = x_b_train[i:i+batch_size]
        qt_batch = q_T_train[i:i+batch_size]
        S_batch = S_train[i:i+batch_size]
        batch_loss = train_step(xa_batch, xb_batch, qt_batch, S_batch)
        epoch_loss += batch_loss.numpy() * len(xa_batch)
    
    epoch_loss /= n_train_samples
    train_losses.append(epoch_loss)
    
    # Validation phase
    val_loss = evaluate(x_a_val, x_b_val, q_T_val, S_val)
    val_losses.append(val_loss)
        
    if epoch % 50 == 0:
        time_current = datetime.datetime.now().replace(microsecond=0)
        duration = time_current - starttime
        print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Duration at Epoch {epoch} --> {duration}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title(f'Model Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Log scale for better visualization

# Save the plot
plot_path = os.path.join(plots_folder, f"loss_plot_model_{model_idx}.png")
plt.savefig(plot_path)
plt.close()
print(f"Loss plot saved to {plot_path}")

# Save the loss data for future reference
loss_df = pd.DataFrame({
    'epoch': range(n_epochs),
    'train_loss': train_losses,
    'val_loss': val_losses
})
loss_df.to_csv(os.path.join(plots_folder, f"loss_data_model_{model_idx}.csv"), index=False)

# Skip saving if training diverged
if len(train_losses) == 0 or np.isnan(train_losses[-1]) or np.isnan(val_losses[-1]):
    print(f"Training resulted in NaN loss. Skipping model save and output.")
    sys.exit(0)


# Save the trained models
NN_a.save(f"{nna_models}/NN_a_model_{model_idx}.h5")
NN_b.save(f"{nnb_models}/NN_b_model_{model_idx}.h5")
# For Future
# NN_a.save(f"{nna_models}/NN_a_model_{model_idx}.keras")
S_pred_model.save(f"{SqT_models}/SqT_pred_model_{model_idx}.h5")
print(f"Models saved: 'NN_a_model_{model_idx}', 'NN_b_model_{model_idx}', 'SqT_pred_model_{model_idx}'")

# Evaluation and save predictions
inputs_test = np.stack([q_T_train, x_a_train, x_b_train], axis=-1)
S_pred_test = S_pred_model.predict(inputs_test).flatten()

df = pd.DataFrame({
    'x1': x_a_train.flatten(),
    'x2': x_b_train.flatten(),
    'qT': q_T_train.flatten(),
    'SqT_true': S_train.flatten(),
    'SqT_pred': S_pred_test.flatten()
})
df.to_csv(f"{CSV_FOLDER}/SqT_predictions_{model_idx}.csv", index=False)

# Create and save a scatter plot of true vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(S_train.flatten(), S_pred_test.flatten(), alpha=0.5)
plt.plot([min(S_train), max(S_train)], [min(S_train), max(S_train)], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'True vs Predicted Values')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, f"prediction_scatter_model_{model_idx}.png"))
plt.close()

finishtime = datetime.datetime.now().replace(microsecond=0)
totalduration = finishtime - starttime
print(f"\nProcessing complete. Total duration --> {totalduration}")