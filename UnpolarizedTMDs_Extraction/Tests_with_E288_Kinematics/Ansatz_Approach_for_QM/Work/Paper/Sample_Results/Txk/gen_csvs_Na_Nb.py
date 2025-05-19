import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime



# Constants
X_MIN = 0.1
X_MAX = 0.6
K_MIN = 0.0
K_MAX = 10.0
N_X = 500
N_K = 500

x_vals = np.linspace(X_MIN, X_MAX, N_X)
k_vals = np.linspace(K_MIN, K_MAX, N_K)

# Create meshgrid and flatten
x_grid, k_grid = np.meshgrid(x_vals, k_vals, indexing='ij')
x_flat = x_grid.ravel()
k_flat = k_grid.ravel()
inputs = np.stack([x_flat, k_flat], axis=1).astype(np.float32)  # shape (250000, 2)


CSV_FOLDER = 'csvs'

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

nna_models_folder = '../Step_03_Fitting_SqT_Extracting_NaNb/nna_models'
nnb_models_folder = '../Step_03_Fitting_SqT_Extracting_NaNb/nnb_models'

# nna_model_files = [f for f in os.listdir(nna_models_folder) if f.endswith('.h5')]
# nna_models_list = [tf.keras.models.load_model(os.path.join(nna_model_files, f)) for f in nna_model_files]

# nnb_model_files = [f for f in os.listdir(nnb_models_folder) if f.endswith('.h5')]
# nnb_models_list = [tf.keras.models.load_model(os.path.join(nnb_model_files, f)) for f in nnb_model_files]

nna_model_files = [f for f in os.listdir(nna_models_folder) if f.endswith('.h5')]
nna_models_list = [tf.keras.models.load_model(os.path.join(nna_models_folder, f)) for f in nna_model_files]

nnb_model_files = [f for f in os.listdir(nnb_models_folder) if f.endswith('.h5')]
nnb_models_list = [tf.keras.models.load_model(os.path.join(nnb_models_folder, f)) for f in nnb_model_files]


def generate_n1n2_grid(nna_models_list, nnb_models_list):

    temp_predictions_nna = []
    for i, model in enumerate(nna_models_list):
        preds_a = model.predict(inputs, verbose=0).flatten()
        # if np.isnan(preds_a).any():
        #     print(f"Warning: Model {i} in nna_models_list produced NaNs")
        # temp_predictions_nna.append(preds_a)
        if not np.isnan(preds_a).any():
            temp_predictions_nna.append(preds_a)
        else:
            print(f"Skipping model {i} in nna_models_list due to NaNs")

    temp_predictions_nna = np.array(temp_predictions_nna)

    temp_predictions_nnb = []
    for i, model in enumerate(nnb_models_list):
        preds_b = model.predict(inputs, verbose=0).flatten()
        # if np.isnan(preds_b).any():
        #     print(f"Warning: Model {i} in nnb_models_list produced NaNs")
        # temp_predictions_nnb.append(preds_b)
        if not np.isnan(preds_b).any():
            temp_predictions_nnb.append(preds_b)
        else:
            print(f"Skipping model {i} in nnb_models_list due to NaNs")
    temp_predictions_nnb = np.array(temp_predictions_nnb)


    # temp_predictions_nna = np.array([model.predict(inputs, verbose=0).flatten() for model in nna_models_list])
    nna_mean = np.mean(temp_predictions_nna, axis=0)
    nna_std = np.std(temp_predictions_nna, axis=0)

    print(temp_predictions_nna)

    # temp_predictions_nnb = np.array([model.predict(inputs, verbose=0).flatten() for model in nnb_models_list])
    nnb_mean = np.mean(temp_predictions_nnb, axis=0)
    nnb_std = np.std(temp_predictions_nnb, axis=0)

    # Build DataFrame
    df = pd.DataFrame({
        'x': x_flat,
        'k': k_flat,
        'nna_mean': nna_mean,
        'nna_std': nna_std,
        'nnb_mean': nnb_mean,
        'nnb_std': nnb_std,
    })

    df.to_csv(os.path.join(CSV_FOLDER, "n1n2_grid.csv"), index=False)
    print("n1n2_grid.csv saved successfully.")


def main():
    create_folders(CSV_FOLDER)
    starttime = datetime.datetime.now().replace(microsecond=0)
    generate_n1n2_grid(nna_models_list, nnb_models_list)
    finishtime = datetime.datetime.now().replace(microsecond=0)
    totalduration = finishtime - starttime
    print(f"Total duration --> {totalduration}")

if __name__ == "__main__":
    main()
