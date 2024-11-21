import numpy as np
from scipy.integrate import simpson
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt

# Load the pseudo data
data = pd.read_csv("A_pT_data.csv")
pT_values = data['pT'].values
A_true = data['A'].values

# Define the DNN model for S1 and S2
def build_dnn():
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(88, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='relu')  # Output S(k)
    ])
    return model

# Instantiate DNNs
dnn_S1 = build_dnn()
dnn_S2 = build_dnn()


def custom_loss(A_true, pT_values, dnn_S1, dnn_S2):
    k = tf.linspace(0.0, 2.0, 100)  # Discretized integration range as a tf.Tensor
    loss = tf.constant(0.0, dtype=tf.float32)

    for i, pT in enumerate(pT_values):
        # Compute DNN outputs
        k_tf = tf.reshape(k, (-1, 1))  # Reshape for input to DNNs
        pT_minus_k = tf.reshape(pT - k, (-1, 1))  # Compute (pT - k)

        S1_k = dnn_S1(k_tf)  # S1(k)
        S2_pT_minus_k = dnn_S2(pT_minus_k)  # S2(pT - k)
        S1_pT_minus_k = dnn_S1(pT_minus_k)  # S1(pT - k)
        S2_k = dnn_S2(k_tf)  # S2(k)

        # Integrand
        integrand = S1_k * S2_pT_minus_k + S1_pT_minus_k * S2_k

        # Compute integral using the trapezoidal rule (TensorFlow operations)
        dk = (k[-1] - k[0]) / tf.cast(tf.size(k) - 1, tf.float32)  # Step size
        pred_A = tf.reduce_sum(integrand) * dk

        # Accumulate loss
        loss += (A_true[i] - pred_A) ** 2

    return loss / len(pT_values)


# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 1000
losses = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = custom_loss(A_true, pT_values, dnn_S1, dnn_S2)
    grads = tape.gradient(loss, dnn_S1.trainable_variables + dnn_S2.trainable_variables)
    optimizer.apply_gradients(zip(grads, dnn_S1.trainable_variables + dnn_S2.trainable_variables))
    losses.append(loss)

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.savefig('Losses.pdf')



# True S1(k) and S2(k)
def S1_true(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 4))

def S2_true(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 1))

# Generate predictions
k_test = np.linspace(0, 2, 100).reshape(-1, 1)
S1_pred = dnn_S1.predict(k_test).flatten()
S2_pred = dnn_S2.predict(k_test).flatten()

# True values
k_test_flat = k_test.flatten()
S1_actual = S1_true(k_test_flat)
S2_actual = S2_true(k_test_flat)

# Plot S1(k)
plt.figure(figsize=(10, 8))
plt.plot(k_test_flat, S1_actual, label="True S1(k)", color="blue")
plt.plot(k_test_flat, S1_pred, label="Predicted S1(k)", linestyle="dashed", color="red")
plt.xlabel("k")
plt.ylabel("S1(k)")
plt.title("Comparison of True and Predicted S1(k)")
plt.legend()
plt.grid()
plt.savefig('S1_comparison.pdf')

# Plot S2(k)
plt.figure(figsize=(10, 8))
plt.plot(k_test_flat, S2_actual, label="True S2(k)", color="blue")
plt.plot(k_test_flat, S2_pred, label="Predicted S2(k)", linestyle="dashed", color="red")
plt.xlabel("k")
plt.ylabel("S2(k)")
plt.title("Comparison of True and Predicted S2(k)")
plt.legend()
plt.grid()
plt.savefig('S2_comparison.pdf')

