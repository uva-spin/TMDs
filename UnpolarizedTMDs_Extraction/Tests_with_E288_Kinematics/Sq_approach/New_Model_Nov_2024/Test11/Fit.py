import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
import os

# Function to create folders
def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# User inputs
kmin = 0.0001
kmax = 10.0
kbins = 100
phibins = 100
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 500
print_epochs = 10
models_folder = 'Models'

create_folders(models_folder)

# Load the pseudo-data
data = pd.read_csv("A_qT_QM_data.csv")
qT_values = data['qT'].values
QM_values = data['QM'].values
A_true = data['A'].values

# Define the DNN model for S1(k, QM) and S2(k, QM)
def build_dnn():
    # Input layers
    k_input = layers.Input(shape=(1,), name="k_input")
    QM_input = layers.Input(shape=(1,), name="QM_input")
    
    # Subnetwork for `k`
    k_subnet = layers.Dense(64, activation='relu6')(k_input)
    k_subnet = layers.Dense(32, activation='relu6')(k_subnet)
    
    # Subnetwork for `QM`
    QM_subnet = layers.Dense(64, activation='relu6')(QM_input)
    QM_subnet = layers.Dense(32, activation='relu6')(QM_subnet)
    
    # Combine the outputs
    combined = layers.Concatenate()([k_subnet, QM_subnet])
    combined = layers.Dense(16, activation='relu6')(combined)
    combined = layers.Dense(8, activation='relu6')(combined)
    output = layers.Dense(1, activation='exponential')(combined)  # Final output
    
    # Define the model
    model = models.Model(inputs=[k_input, QM_input], outputs=output)
    return model

# Instantiate DNNs
dnn_S1 = build_dnn()
dnn_S2 = build_dnn()

# Define custom loss function
def custom_loss(A_true, qT_values, QM_values, dnn_S1, dnn_S2):
    k = tf.linspace(kmin, kmax, kbins)  # Discretized k values
    phi = tf.linspace(0.0, 2 * np.pi, phibins)  # Discretized phi values
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

            # Separate inputs for S1 and S2
            S1_inputs = [k_tf, QM_tf]
            QM_tf_broadcasted = tf.broadcast_to(QM_tf, [len(phi), 1])  # Broadcast QM for phi
            S2_inputs = [sqrt_term, QM_tf_broadcasted]

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
losses = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = custom_loss(A_true, qT_values, QM_values, dnn_S1, dnn_S2)
    grads = tape.gradient(loss, dnn_S1.trainable_variables + dnn_S2.trainable_variables)
    optimizer.apply_gradients(zip(grads, dnn_S1.trainable_variables + dnn_S2.trainable_variables))
    losses.append(loss)

    if epoch % print_epochs == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.6f}")

# Save the trained models
dnn_S1.save(str(models_folder)+'/S1_model.h5')
dnn_S2.save(str(models_folder)+'/S2_model.h5')

print("Models saved successfully!")

# Plot loss curve
plt.figure(figsize=(10, 8))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.savefig('Losses.pdf')


# Concatenate outputs of both sub-networks. One can simply used the first line but the rest of the lines to capture the mathematical operation for k and QM