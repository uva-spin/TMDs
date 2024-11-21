import numpy as np
import pandas as pd
from scipy.integrate import simpson
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the pseudo data
data = pd.read_csv("A_qT_data.csv")
qT_values = data['qT'].values
A_true = data['A'].values

# Define the DNN model for S1 and S2
def build_dnn():
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='exponential')  # Ensure positive output
    ])
    return model

# Instantiate DNNs
dnn_S1 = build_dnn()
dnn_S2 = build_dnn()

# Custom loss function with double integration
def custom_loss(A_true, qT_values, dnn_S1, dnn_S2):
    k = tf.linspace(0.0, 2.0, 100)  # Discretized k values
    phi = tf.linspace(0.0, 2 * np.pi, 100)  # Discretized phi values
    loss = tf.constant(0.0, dtype=tf.float32)

    for i, qT in enumerate(qT_values):
        integrand_values = tf.TensorArray(tf.float32, size=len(k))

        for j, k_val in enumerate(k):
            k_tf = tf.reshape(k_val, (1, 1))  # Prepare k for DNNs
            phi_tf = tf.reshape(phi, (-1, 1))  # Prepare phi for computation

            # Compute qT**2 + k**2 - 2*qT*k*cos(phi) using TensorFlow operations
            cos_phi = tf.cos(phi_tf)
            sqrt_term = tf.sqrt(qT**2 + k_val**2 - 2 * qT * k_val * cos_phi)

            # Compute DNN outputs
            term1 = dnn_S1(k_tf) * dnn_S2(sqrt_term)
            term2 = dnn_S1(sqrt_term) * dnn_S2(k_tf)

            # Compute the integrand
            integrand = term1 + term2
            phi_integral = tf.reduce_sum(integrand) * (phi[1] - phi[0])
            integrand_values = integrand_values.write(j, phi_integral)

        # Integrate over k
        integrand_values = integrand_values.stack()
        total_integral = tf.reduce_sum(integrand_values) * (k[1] - k[0])

        # Compute loss
        loss += (A_true[i] - total_integral) ** 2

    return loss / len(qT_values)

# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 100
losses = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = custom_loss(A_true, qT_values, dnn_S1, dnn_S2)
    grads = tape.gradient(loss, dnn_S1.trainable_variables + dnn_S2.trainable_variables)
    optimizer.apply_gradients(zip(grads, dnn_S1.trainable_variables + dnn_S2.trainable_variables))
    losses.append(loss)

    if epoch % 2 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.savefig('Losses_qT.pdf')

# Compare predicted and true S1(k), S2(k)
k_test = np.linspace(0, 2, 100).reshape(-1, 1)
S1_pred = dnn_S1.predict(k_test).flatten()
S2_pred = dnn_S2.predict(k_test).flatten()

def S1_true(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 4))

def S2_true(k):
    return np.exp(-4 * k**2 / (4 * k**2 + 1))

S1_actual = S1_true(k_test.flatten())
S2_actual = S2_true(k_test.flatten())

plt.figure(figsize=(10, 8))
plt.plot(k_test, S1_actual, label="True S1(k)", color="blue")
plt.plot(k_test, S1_pred, label="Predicted S1(k)", linestyle="dashed", color="red")
plt.title("Comparison of True and Predicted S1(k)")
plt.xlabel("k")
plt.ylabel("S1(k)")
plt.legend()
plt.grid()
plt.savefig('S1_comparison_qT.pdf')

plt.figure(figsize=(10, 8))
plt.plot(k_test, S2_actual, label="True S2(k)", color="blue")
plt.plot(k_test, S2_pred, label="Predicted S2(k)", linestyle="dashed", color="red")
plt.title("Comparison of True and Predicted S2(k)")
plt.xlabel("k")
plt.ylabel("S2(k)")
plt.legend()
plt.grid()
plt.savefig('S2_comparison_qT.pdf')
