import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


models_folder = 'Models'

# Load the trained models
dnn_S1 = tf.keras.models.load_model(str(models_folder)+'/S1_model.h5')
dnn_S2 = tf.keras.models.load_model(str(models_folder)+'/S2_model.h5')

# Define the true S1(k, QM) and S2(k, QM) functions
def S1_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 4)) * np.log(Q0 / QM)

def S2_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 1)) * np.log(Q0 / QM)

# Generate test data
k_test = np.linspace(0, 2, 100).reshape(-1, 1)
QM_test = np.full_like(k_test, 5)  # Example QM value
inputs_test = np.hstack([k_test, QM_test])

# Get predictions from the models
S1_pred = dnn_S1.predict(inputs_test).flatten()
S2_pred = dnn_S2.predict(inputs_test).flatten()

# Compute the true values
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

print("Plots saved successfully!")
