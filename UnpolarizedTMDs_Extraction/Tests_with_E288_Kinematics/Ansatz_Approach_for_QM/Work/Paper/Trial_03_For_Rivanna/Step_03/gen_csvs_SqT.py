import os
import re
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

# ==== Configuration ====
CSV_FOLDER = "comp_results_csvs"
NN_A_FOLDER = "nna_models"
NN_B_FOLDER = "nnb_models"
SQT_SOURCE_FOLDER = "../Step_01/Models"


#k_low = 0.0
#k_high = 10.0

k_low = 0.0001
k_high = 5.0

N_K = 64
N_PHI = 64
K_VALS = tf.linspace(k_low, k_high, N_K)
PHI_VALS = tf.linspace(0.0, 2 * np.pi, N_PHI)
K_GRID, PHI_GRID = tf.meshgrid(K_VALS, PHI_VALS, indexing='ij')

# ==== Utilities ====
def create_folder(path):
    os.makedirs(path, exist_ok=True)
    print(f"Folder '{path}' is ready.")

def extract_model_index(filename):
    match = re.search(r"_(\d+)\.h5$", filename)
    return int(match.group(1)) if match else None

# ==== Custom Loss ====
def custom_weighted_loss(y_true, y_pred, w=None):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    if w is not None:
        w = tf.cast(w, tf.float32)
        weights = w / tf.reduce_mean(w)
        return tf.reduce_mean(tf.square(y_pred - y_true) * weights)
    return tf.reduce_mean(tf.square(y_pred - y_true))

class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_weighted_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return custom_weighted_loss(y_true, y_pred)

# ==== Load Models ====
def get_model_files_by_index(folder):
    return {extract_model_index(f): f for f in os.listdir(folder) if f.endswith('.h5')}

def load_true_models():
    files = sorted(os.listdir(SQT_SOURCE_FOLDER))
    models = []
    for fname in files:
        if not fname.endswith('.h5'):
            continue
        model = tf.keras.models.load_model(
            os.path.join(SQT_SOURCE_FOLDER, fname),
            custom_objects={
                'custom_weighted_loss': custom_weighted_loss,
                'CustomWeightedLoss': CustomWeightedLoss,
                'train_weighted_loss': custom_weighted_loss
            }
        ).get_layer('SqT')
        models.append(model)
    return models

def build_pred_model(nn_a, nn_b):
    class SqTPredictionLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            q_T, x_a, x_b = inputs[:, 0], inputs[:, 1], inputs[:, 2]
            k = tf.reshape(K_GRID, [-1])
            phi = tf.reshape(PHI_GRID, [-1])

            input_a = tf.stack([
                tf.repeat(x_a, N_K * N_PHI),
                tf.tile(k, [tf.shape(x_a)[0]])
            ], axis=1)
            val_a = nn_a(input_a)

            q_T_rep = tf.repeat(q_T, N_K * N_PHI)
            x_b_rep = tf.repeat(x_b, N_K * N_PHI)
            k_tiled = tf.tile(k, [tf.shape(q_T)[0]])
            phi_tiled = tf.tile(phi, [tf.shape(q_T)[0]])

            k_prime = tf.sqrt(q_T_rep**2 + k_tiled**2 - 2 * q_T_rep * k_tiled * tf.cos(phi_tiled))
            input_b = tf.stack([x_b_rep, k_prime], axis=1)
            val_b = nn_b(input_b)

            integrand = val_a * val_b * tf.reshape(k_tiled, (-1, 1))
            integrand = tf.reshape(integrand, (tf.shape(q_T)[0], N_K * N_PHI))
            integral = tf.reduce_sum(integrand, axis=1) * (k_high / N_K) * (2 * np.pi / N_PHI)

            return tf.reshape(integral, (-1, 1))

    inputs = tf.keras.Input(shape=(3,), name="qT_xa_xb")
    outputs = SqTPredictionLayer()(inputs)
    return tf.keras.Model(inputs, outputs)

def load_all_pred_models(valid_indices):
    models = []
    for idx in valid_indices:
        nn_a = tf.keras.models.load_model(os.path.join(NN_A_FOLDER, f"NN_a_model_{idx}.h5"), compile=False)
        nn_b = tf.keras.models.load_model(os.path.join(NN_B_FOLDER, f"NN_b_model_{idx}.h5"), compile=False)
        models.append(build_pred_model(nn_a, nn_b))
    return models

# ==== Grid Evaluation ====
def generate_grid(fixed_var, fixed_value, n_samples):
    qT_vals = np.linspace(0, 5, n_samples, dtype=np.float32)
    x_vals = np.linspace(0, 1, n_samples, dtype=np.float32)

    if fixed_var == "x1":
        x1_grid, qT_grid = np.meshgrid(np.full(n_samples, fixed_value), qT_vals, indexing="ij")
        x2_grid = np.tile(x_vals, (n_samples, 1))
    elif fixed_var == "x2":
        x2_grid, qT_grid = np.meshgrid(np.full(n_samples, fixed_value), qT_vals, indexing="ij")
        x1_grid = np.tile(x_vals, (n_samples, 1))
    else:
        raise ValueError("Invalid fixed_var. Must be 'x1' or 'x2'.")

    x1, x2, qT = x1_grid.ravel(), x2_grid.ravel(), qT_grid.ravel()
    inputs = np.stack([qT, x1, x2], axis=-1)

    true_vals, pred_vals, abs_diffs = [], [], []
    for true_model, pred_model in zip(SqT_Source_models, SqT_Pred_models):
        sqt_true = true_model.predict(inputs, verbose=0).flatten()
        sqt_pred = pred_model.predict(inputs, verbose=0).flatten()
        true_vals.append(sqt_true)
        pred_vals.append(sqt_pred)
        abs_diffs.append(np.abs(sqt_true - sqt_pred))

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'qT': qT,
        'SqT_true_mean': np.mean(true_vals, axis=0),
        'SqT_true_std': np.std(true_vals, axis=0),
        'SqT_pred_mean': np.mean(pred_vals, axis=0),
        'SqT_pred_std': np.std(pred_vals, axis=0),
        'dSqT': np.mean(abs_diffs, axis=0)
    })

    return df

# ==== Main Workflow ====
def main():
    create_folder(CSV_FOLDER)

    nn_a_indices = get_model_files_by_index(NN_A_FOLDER)
    nn_b_indices = get_model_files_by_index(NN_B_FOLDER)
    sqt_indices  = get_model_files_by_index(SQT_SOURCE_FOLDER)

    valid_indices = sorted(set(nn_a_indices) & set(nn_b_indices) & set(sqt_indices))
    print(f"Using model indices: {valid_indices}")

    global SqT_Source_models
    SqT_Source_models = load_true_models()

    global SqT_Pred_models
    SqT_Pred_models = load_all_pred_models(valid_indices)

    start = datetime.datetime.now().replace(microsecond=0)

    for xb in [0.2, 0.4, 0.6]:
        df = generate_grid('x2', xb, 100)
        filename = f"SqT_xb_{int(xb * 100):02d}.csv"
        df.to_csv(os.path.join(CSV_FOLDER, filename), index=False)

    end = datetime.datetime.now().replace(microsecond=0)
    print(f"Total Duration --> {end - start}")

if __name__ == "__main__":
    main()
