import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime

# Constants
X_MIN = 0.1
X_MAX = 0.3
K_MIN = 0.0
K_MAX = 10.0
N_X = 100
N_K = 100
CSV_FOLDER = 'csvs'

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

def load_models():
    NN_a = tf.keras.models.load_model("NN_a_model.h5", compile=False)
    NN_b = tf.keras.models.load_model("NN_b_model.h5", compile=False)
    return NN_a, NN_b

def generate_n1n2_grid(NN_a, NN_b):
    x_vals = np.linspace(X_MIN, X_MAX, N_X)
    k_vals = np.linspace(K_MIN, K_MAX, N_K)

    data = []
    for x in x_vals:
        for k in k_vals:
            inp = np.array([[x, k]], dtype=np.float32)
            n1 = NN_a.predict(inp, verbose=0)[0][0]
            n2 = NN_b.predict(inp, verbose=0)[0][0]
            data.append({'x': x, 'k': k, 'n1': n1, 'n2': n2})

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(CSV_FOLDER, "n1n2_grid.csv"), index=False)
    print("n1n2_grid.csv saved successfully.")

def main():
    create_folders(CSV_FOLDER)
    NN_a, NN_b = load_models()
    starttime = datetime.datetime.now().replace(microsecond=0)
    generate_n1n2_grid(NN_a, NN_b)
    finishtime = datetime.datetime.now().replace(microsecond=0)
    totalduration = finishtime - starttime
    print(f"Total duration --> {totalduration}")

if __name__ == "__main__":
    main()
