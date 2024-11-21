import numpy as np
import pandas as pd
from scipy.integrate import simpson
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load the pseudo-data
data = pd.read_csv("A_qT_QM_data.csv")
qT_values = data['qT'].values
QM_values = data['QM'].values
A_true = data['A'].values

# Define the true S1(k, QM) and S2(k, QM) functions
def S1_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 4)) * np.log(Q0 / QM)

def S2_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 1)) * np.log(Q0 / QM)

# Define the DNN model for S1(k, QM) and S2(k, QM)
def build_dnn():
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='exponential')  # Ensure positive output
    ])
    return model

# Instantiate DNNs
dnn_S1 = build_dnn()
dnn_S2 = build_dnn()

# Define custom loss function
def custom_loss(A_true, qT_values, QM_values, dnn_S1, dnn_S2):
    k = tf.linspace(0.0001, 10.0, 100)  # Discretized k values
    phi = tf.linspace(0.0, 2 * np.pi, 100)  # Discretized phi values
    loss = tf.constant(0.0, dtype=tf.float32)

    for i, (qT, QM) in enumerate(zip(qT_values, QM_values)):
        integrand_values = tf.TensorArray(tf.float32, size=len(k))

        for j, k_val in enumerate(k):
            k_tf = tf.cast(tf.reshape(k_val, (1, 1)), dtype=tf.float32)
            phi_tf = tf.cast(tf.reshape(phi, (-1, 1)), dtype=tf.float32)
            QM_tf = tf.cast(tf.reshape(QM, (1, 1)), dtype=tf.float32)

            # Compute qT**2 + k**2 - 2*qT*k*cos(phi)
            cos_phi = tf.cos(phi_tf)
            sqrt_term = tf.sqrt(tf.cast(qT**2, tf.float32) + tf.cast(k_val**2, tf.float32) - 
                                2 * tf.cast(qT, tf.float32) * tf.cast(k_val, tf.float32) * cos_phi)

            # Combine inputs for DNNs
            S1_inputs = tf.concat([k_tf, QM_tf], axis=1)
            QM_tf_broadcasted = tf.broadcast_to(QM_tf, [100, 1])
            S2_inputs = tf.concat([sqrt_term, QM_tf_broadcasted], axis=1)

            # Compute outputs
            term1 = dnn_S1(S1_inputs) * dnn_S2(S2_inputs)
            term2 = dnn_S1(S2_inputs) * dnn_S2(S1_inputs)

            # Compute the integrand
            integrand = term1 + term2
            phi_integral = tf.reduce_sum(integrand) * (phi[1] - phi[0])
            integrand_values = integrand_values.write(j, phi_integral)

        # Integrate over k
        integrand_values = integrand_values.stack()
        total_integral = tf.reduce_sum(integrand_values) * (k[1] - k[0])

        # Compute loss
        loss += (tf.cast(A_true[i], tf.float32) - total_integral) ** 2

    return loss / len(qT_values)

# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 3
losses = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = custom_loss(A_true, qT_values, QM_values, dnn_S1, dnn_S2)
    grads = tape.gradient(loss, dnn_S1.trainable_variables + dnn_S2.trainable_variables)
    optimizer.apply_gradients(zip(grads, dnn_S1.trainable_variables + dnn_S2.trainable_variables))
    losses.append(loss)

    if epoch % 2 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.6f}")

# Plot loss curve
plt.figure(figsize=(10, 8))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.savefig('Losses.pdf')

# Generate predictions
k_test = np.linspace(0.0001, 10, 100).reshape(-1, 1)
QM_test = np.full_like(k_test, QM_values[0])  # Example QM value
inputs_test = np.hstack([k_test, QM_test])

S1_pred = dnn_S1.predict(inputs_test).flatten()
S2_pred = dnn_S2.predict(inputs_test).flatten()

# True values
S1_actual = S1_true(k_test.flatten(), QM_test.flatten())
S2_actual = S2_true(k_test.flatten(), QM_test.flatten())

# Plot S1(k, QM)
plt.figure(figsize=(10, 8))
plt.plot(k_test.flatten(), S1_actual, label="True S1(k, QM)", color="blue")
plt.plot(k_test.flatten(), S1_pred, label="Predicted S1(k, QM)", linestyle="dashed", color="red")
plt.xlabel("k")
plt.ylabel("S1(k, QM)")
plt.title("Comparison of True and Predicted S1(k, QM)")
plt.legend()
plt.grid()
plt.savefig('S1_comparison.pdf')

# Plot S2(k, QM)
plt.figure(figsize=(10, 8))
plt.plot(k_test.flatten(), S2_actual, label="True S2(k, QM)", color="blue")
plt.plot(k_test.flatten(), S2_pred, label="Predicted S2(k, QM)", linestyle="dashed", color="red")
plt.xlabel("k")
plt.ylabel("S2(k, QM)")
plt.title("Comparison of True and Predicted S2(k, QM)")
plt.legend()
plt.grid()
plt.savefig('S2_comparison.pdf')
