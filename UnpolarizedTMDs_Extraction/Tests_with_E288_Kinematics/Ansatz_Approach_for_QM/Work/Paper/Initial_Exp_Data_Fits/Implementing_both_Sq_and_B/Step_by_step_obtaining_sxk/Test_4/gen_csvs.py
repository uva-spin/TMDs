import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from model import *

# Constants for data generation
NUM_SAMPLES = 100
QT_FIXED = 1.0
X_MIN = 0.1
X_MAX = 0.3

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

CSV_folder = 'csvs'
create_folders(CSV_folder)

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class TMDIntegrationLayer(tf.keras.layers.Layer):
    def __init__(self, k_bins=10, phi_bins=10, k_lower=0.0, k_upper=10.0, **kwargs):
        super().__init__(**kwargs)
        self.k_bins = k_bins
        self.phi_bins = phi_bins
        self.k_lower = float(k_lower)
        self.k_upper = float(k_upper)
        self.k_values = np.linspace(self.k_lower, self.k_upper, self.k_bins, dtype=np.float32)
        self.phi_values = np.linspace(0.0, np.pi, self.phi_bins, dtype=np.float32)
        self.dk = (self.k_upper - self.k_lower) / (self.k_bins - 1)
        self.dphi = np.pi / (self.phi_bins - 1)
        self.modn1 = create_nn_model('n1')
        self.modn2 = create_nn_model('n2')

    def call(self, inputs):
        qT, x1, x2 = inputs
        batch_size = tf.shape(qT)[0]
        result = tf.zeros((batch_size, 1), dtype=tf.float32)

        for k_val in self.k_values:
            k_factor = k_val
            for phi_val in self.phi_values:
                kb_tensor = tf.sqrt(qT**2 + k_val**2 - 2*qT*k_val*tf.cos(phi_val))
                n1_input = tf.concat([x1, tf.ones_like(x1) * k_val], axis=1)
                n2_input = tf.concat([x2, kb_tensor], axis=1)
                nn1_out = self.modn1(n1_input)
                nn2_out = self.modn2(n2_input)
                contribution = nn1_out * nn2_out * k_factor * self.dk * self.dphi
                result += contribution
        return result

def load_model():
    model_file = 'DNNmodels/model0.h5'
    custom_objects = {
        'mse_loss': mse_loss,
        'TMDIntegrationLayer': TMDIntegrationLayer
    }
    return tf.keras.models.load_model(model_file, custom_objects=custom_objects)

def generate_data(num_samples=NUM_SAMPLES):
    qT = np.full(num_samples, QT_FIXED, dtype=np.float32)
    x1 = np.random.uniform(X_MIN, X_MAX, num_samples).astype(np.float32)
    x2 = np.random.uniform(X_MIN, X_MAX, num_samples).astype(np.float32)
    return pd.DataFrame({'qT': qT, 'x1': x1, 'x2': x2})


def main():
    print("Loading model0.h5...")
    model = load_model()

    df = generate_data()

    # Prepare input for model prediction
    qT_array = df['qT'].values.reshape(-1, 1)
    x1_array = df['x1'].values.reshape(-1, 1)
    x2_array = df['x2'].values.reshape(-1, 1)

    print("Generating SqT predictions...")
    SqT_pred = model.predict([qT_array, x1_array, x2_array], verbose=0).flatten()
    df['SqT_pred'] = SqT_pred
    df.to_csv(os.path.join(CSV_folder, 'SqT_results.csv'), index=False)
    print("SqT predictions saved.")

    # Extract n1 model from TMDIntegrationLayer
    tmd_layer = next(layer for layer in model.layers if isinstance(layer, TMDIntegrationLayer))
    n1_model = tmd_layer.modn1
    n2_model = tmd_layer.modn2

    # Grid generation for x1 and k
    print("Generating n1(x1, k) grid...")
    x1_vals = np.linspace(X_MIN, X_MAX, 100)
    k_vals = np.linspace(0.0, 10.0, 100)
    grid_data = []

    for x1 in x1_vals:
        for k in k_vals:
            input_val = np.array([[x1, k]], dtype=np.float32)
            n1_out = n1_model.predict(input_val, verbose=0)[0][0]
            n2_out = n2_model.predict(input_val, verbose=0)[0][0]
            grid_data.append({'x1': x1, 'k': k, 'n1': n1_out, 'n2': n2_out})

    grid_df = pd.DataFrame(grid_data)
    grid_df.to_csv(os.path.join(CSV_folder, 'n1n2_grid.csv'), index=False)
    print("n1 grid predictions saved.")

if __name__ == "__main__":
    main()

# def main():
#     # Load the model
#     print("Loading trained model...")
#     model = load_models()

#     df = generate_data()

#     # Prepare input arrays
#     qT_array = df['qT'].values.reshape(-1, 1)
#     x1_array = df['x1'].values.reshape(-1, 1)
#     x2_array = df['x2'].values.reshape(-1, 1)

#     fine_k_values = np.linspace(0, 10, len(x1_array))

#     SqT_org = df['SqT_org'].values
#     SqT_predictions = model.predict([qT_array, x1_array, x2_array], verbose=0)

#     # Extract TMDIntegrationLayer
#     tmd_layer = None
#     for layer in model.layers:
#         if isinstance(layer, TMDIntegrationLayer):
#             tmd_layer = layer
#             break

#     n1_pred_for_x1 = []
#     n2_pred_for_x2 = []

#     if tmd_layer:
#         n1_model = tmd_layer.modn1
#         n2_model = tmd_layer.modn2

#         for i in range(len(x1_array)):
#             x1_val = x1_array[i][0]
#             x2_val = x2_array[i][0]
#             n1_outputs_each_k = []
#             n2_outputs_each_k = []

#             for k_val in fine_k_values:
#                 # --- n1 prediction
#                 n1_input = np.array([[x1_val, k_val]])
#                 n1_output = n1_model.predict(n1_input, verbose=0)[0][0]
#                 n1_outputs_each_k.append(n1_output)

#                 # --- n2 prediction (directly using x2 and k)
#                 n2_input = np.array([[x2_val, k_val]])
#                 n2_output = n2_model.predict(n2_input, verbose=0)[0][0]
#                 n2_outputs_each_k.append(n2_output)

#             n1_pred_for_x1.append({
#                 'x1': x1_val,
#                 'k_values': fine_k_values,
#                 'n1_values': np.array(n1_outputs_each_k)
#             })

#             n2_pred_for_x2.append({
#                 'x2': x2_val,
#                 'k_values': fine_k_values,
#                 'n2_values': np.array(n2_outputs_each_k)
#             })

#     # Save SqT predictions
#     SqT_df = pd.DataFrame({
#         'x1': df['x1'],
#         'x2': df['x2'],
#         'qT': df['qT'],
#         'SqT_org': SqT_org,
#         'SqT_pred_mean': SqT_predictions.flatten()
#     })
#     SqT_df.to_csv(os.path.join(CSV_folder, 'SqT_results.csv'), index=False)
#     print("Results for SqT are saved")

#     # Save n1 predictions
#     n1_data_rows = []
#     for item in n1_pred_for_x1:
#         x1_val = item['x1']
#         for k_val, n1_val in zip(item['k_values'], item['n1_values']):
#             n1_data_rows.append({
#                 'x1': x1_val,
#                 'k': k_val,
#                 'n1': n1_val
#             })
#     pd.DataFrame(n1_data_rows).to_csv(os.path.join(CSV_folder, 'n1_results.csv'), index=False)
#     print("Results for n1 are saved")

#     # Save n2 predictions
#     n2_data_rows = []
#     for item in n2_pred_for_x2:
#         x2_val = item['x2']
#         for k_val, n2_val in zip(item['k_values'], item['n2_values']):
#             n2_data_rows.append({
#                 'x2': x2_val,
#                 'k': k_val,
#                 'n2': n2_val
#             })
#     pd.DataFrame(n2_data_rows).to_csv(os.path.join(CSV_folder, 'n2_results.csv'), index=False)
#     print("Results for n2 are saved")


if __name__ == "__main__":
    main()
