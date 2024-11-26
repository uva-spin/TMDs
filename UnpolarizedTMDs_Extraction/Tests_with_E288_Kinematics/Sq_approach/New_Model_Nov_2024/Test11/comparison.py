import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

models_folder = 'Models'

# Load the trained models
dnn_S1 = tf.keras.models.load_model(str(models_folder) + '/S1_model.h5')
dnn_S2 = tf.keras.models.load_model(str(models_folder) + '/S2_model.h5')

# Define the true S1(k, QM) and S2(k, QM) functions
def S1_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 4)) * np.log(Q0 / QM)

def S2_true(k, QM):
    Q0 = 100
    return np.exp(-4 * k**2 / (4 * k**2 + 1)) * np.log(Q0 / QM)

# Generate test data
k_test = np.linspace(0.0001, 2, 100).reshape(-1, 1)
QM_test = np.array([1, 2, 3])  # Three QM values for comparison

# Initialize storage for results
all_S1_pred = []
all_S2_pred = []
all_S1_actual = []
all_S2_actual = []

# Loop through QM values and compute results
for QM in QM_test:
    QM_array = np.full_like(k_test, QM)  # Create a column of QM values matching k_test
    
    # Prepare inputs for multi-input models
    S1_inputs = [k_test, QM_array]
    S2_inputs = [k_test, QM_array]
    
    # Get predictions from the models
    S1_pred = dnn_S1.predict(S1_inputs).flatten()
    S2_pred = dnn_S2.predict(S2_inputs).flatten()
    
    # Compute true values
    S1_actual = S1_true(k_test.flatten(), QM_array.flatten())
    S2_actual = S2_true(k_test.flatten(), QM_array.flatten())
    
    # Store results for plotting
    all_S1_pred.append(S1_pred)
    all_S2_pred.append(S2_pred)
    all_S1_actual.append(S1_actual)
    all_S2_actual.append(S2_actual)

# Plot S1(k, QM) for each QM
plt.figure(figsize=(10, 8))
for i, QM in enumerate(QM_test):
    plt.plot(k_test.flatten(), all_S1_actual[i], label=f"True S1(k, QM={QM})", color=f"C{i}")
    plt.plot(k_test.flatten(), all_S1_pred[i], label=f"Predicted S1(k, QM={QM})", linestyle="dashed", color=f"C{i}")
plt.xlabel("k")
plt.ylabel("S1(k, QM)")
plt.title("Comparison of True and Predicted S1(k, QM) for Different QM")
plt.legend()
plt.grid()
plt.savefig('S1_comparison_multiple_QM.pdf')

# Plot S2(k, QM) for each QM
plt.figure(figsize=(10, 8))
for i, QM in enumerate(QM_test):
    plt.plot(k_test.flatten(), all_S2_actual[i], label=f"True S2(k, QM={QM})", color=f"C{i}")
    plt.plot(k_test.flatten(), all_S2_pred[i], label=f"Predicted S2(k, QM={QM})", linestyle="dashed", color=f"C{i}")
plt.xlabel("k")
plt.ylabel("S2(k, QM)")
plt.title("Comparison of True and Predicted S2(k, QM) for Different QM")
plt.legend()
plt.grid()
plt.savefig('S2_comparison_multiple_QM.pdf')

print("Plots saved successfully!")
