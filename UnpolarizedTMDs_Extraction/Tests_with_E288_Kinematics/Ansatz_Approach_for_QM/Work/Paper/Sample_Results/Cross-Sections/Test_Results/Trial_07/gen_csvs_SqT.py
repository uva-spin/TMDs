import os
import re
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

# ==== Configuration ====
CSV_FOLDER = "Results_csvs"
SQT_SOURCE_FOLDER = 'Models'



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
    for true_model in SqT_Source_models:
        sqt_true = true_model.predict(inputs, verbose=0).flatten()
        true_vals.append(sqt_true)

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'qT': qT,
        'SqT_true_mean': np.mean(true_vals, axis=0),
        'SqT_true_std': np.std(true_vals, axis=0)
    })

    return df

# ==== Main Workflow ====
def main():
    create_folder(CSV_FOLDER)

    sqt_indices  = get_model_files_by_index(SQT_SOURCE_FOLDER)

    valid_indices = sorted(set(sqt_indices))
    print(f"Using model indices: {valid_indices}")

    global SqT_Source_models
    SqT_Source_models = load_true_models()

    start = datetime.datetime.now().replace(microsecond=0)

    for x2 in [0.2, 0.4, 0.6]:
        df = generate_grid('x2', x2, 100)
        filename = f"SqT_x2_{int(x2 * 100):02d}.csv"
        df.to_csv(os.path.join(CSV_FOLDER, filename), index=False)

    for x1 in [0.2, 0.4, 0.6]:
        df = generate_grid('x1', x1, 100)
        filename = f"SqT_x1_{int(x1 * 100):02d}.csv"
        df.to_csv(os.path.join(CSV_FOLDER, filename), index=False)

    # # Save x1 grid
    # for val in [0.2, 0.4, 0.6]:
    #     df = generate_grid("x1", val, 100)
    #     df.to_csv(os.path.join(CSV_FOLDER, f"x1_fixed_{val}.csv"), index=False)
    #     print(f"x1_fixed_{val}.csv saved.")

    # # Save x2 grid
    # for val in [0.2, 0.4, 0.6]:
    #     df = generate_grid("x2", val, 100)
    #     df.to_csv(os.path.join(CSV_FOLDER, f"x2_fixed_{val}.csv"), index=False)
    #     print(f"x2_fixed_{val}.csv saved.")

    end = datetime.datetime.now().replace(microsecond=0)
    print(f"Total Duration --> {end - start}")

if __name__ == "__main__":
    main()
