import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")


CSV_FOLDER = 'comp_results_csvs'
create_folders(CSV_FOLDER)

selected_CS_models_folder = '../Step_02_Finding_Models_within_2Sigma/Models'
SqT_Pred_folder = 'SqT_models'

########## Loading True Models ################

# Define custom loss function and class
def custom_weighted_loss(y_true, y_pred, w=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if w is not None:
        w = tf.cast(w, tf.float32)
        mean_w = tf.reduce_mean(w)
        weights = w / mean_w
        squared_error = tf.square(y_pred - y_true)
        weighted_squared_error = squared_error * weights
        return tf.reduce_mean(weighted_squared_error)
    else:
        return tf.reduce_mean(tf.square(y_pred - y_true))

class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_weighted_loss"):
        super().__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return custom_weighted_loss(y_true, y_pred)
    
def createModel_S_pred():
    input_layer = tf.keras.Input(shape=(3,), name="qT_xa_xb")
    output_layer = ComputeSPredLayer()(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="compute_S_pred_model")
    return model

def true_S(model,q_T, x_a, x_b):
    inputs = np.stack([q_T, x_a, x_b], axis=-1)
    return model.predict(inputs)



NN_a = tf.keras.models.load_model('nna_models/NN_a_model_0.h5', compile=False)
NN_b = tf.keras.models.load_model('nnb_models/NN_b_model_0.h5', compile=False)


# Integration grid
n_k = 64
n_phi = 64
k_vals = tf.linspace(0.0, 10.0, n_k)
phi_vals = tf.linspace(0.0, 2*np.pi, n_phi)
k_grid, phi_grid = tf.meshgrid(k_vals, phi_vals, indexing='ij')


# You must redefine the custom layer class
class ComputeSPredLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        q_T, x_a, x_b = inputs[:, 0], inputs[:, 1], inputs[:, 2]

        k = tf.reshape(k_grid, [-1])
        phi = tf.reshape(phi_grid, [-1])

        input_a = tf.stack([tf.repeat(x_a, n_k * n_phi), tf.tile(k, [tf.shape(x_a)[0]])], axis=1)
        val_a = NN_a(input_a)

        q_T_rep = tf.repeat(q_T, n_k * n_phi)
        x_b_rep = tf.repeat(x_b, n_k * n_phi)
        k_tiled = tf.tile(k, [tf.shape(q_T)[0]])
        phi_tiled = tf.tile(phi, [tf.shape(q_T)[0]])

        k_prime = tf.sqrt(q_T_rep**2 + k_tiled**2 - 2*q_T_rep*k_tiled*tf.cos(phi_tiled))
        input_b = tf.stack([x_b_rep, k_prime], axis=1)
        val_b = NN_b(input_b)

        integrand = val_a * val_b * tf.reshape(k_tiled, (-1, 1))
        integrand = tf.reshape(integrand, (tf.shape(q_T)[0], n_k * n_phi))
        integral = tf.reduce_sum(integrand, axis=1) * (10.0 / n_k) * (2*np.pi / n_phi)

        return tf.reshape(integral, (-1, 1))



SqT_Source_files = [f for f in os.listdir(selected_CS_models_folder) if f.endswith('.h5')]
SqT_Source_models = [tf.keras.models.load_model(os.path.join(selected_CS_models_folder, f), 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss}) for f in SqT_Source_files]
SqT_Source_models = [temp_model.get_layer('SqT') for temp_model in SqT_Source_models]

SqT_Pred_files = [f for f in os.listdir(SqT_Pred_folder) if f.endswith('.h5')]
SqT_Pred_models = [tf.keras.models.load_model(os.path.join(SqT_Pred_folder, f), custom_objects={"ComputeSPredLayer": ComputeSPredLayer}) for f in SqT_Pred_files]



def generate_grids_fixed_x(fixed_var, x_value, n_samples):
    qT_vals = np.linspace(0, 5, n_samples).astype(np.float32)
    x_vals = np.linspace(0, 1, n_samples).astype(np.float32)

    if fixed_var == 'x1':
        x1_grid, qT_grid = np.meshgrid(np.full(n_samples, x_value), qT_vals, indexing='ij')
        x2_grid = np.tile(x_vals, (1, n_samples))
    elif fixed_var == 'x2':
        x2_grid, qT_grid = np.meshgrid(np.full(n_samples, x_value), qT_vals, indexing='ij')
        x1_grid = np.tile(x_vals, (1, n_samples))
    else:
        raise ValueError("fixed_var must be either 'x1' or 'x2'")

    # Flatten the grids
    x1_flat = x1_grid.ravel()
    x2_flat = x2_grid.ravel()
    qT_flat = qT_grid.ravel()

    # Stack inputs
    inputs = np.stack([qT_flat, x1_flat, x2_flat], axis=-1).astype(np.float32)

    # Predict
    temp_SqT_True = np.array([
        model.predict(inputs, verbose=0).flatten() for model in SqT_Source_models
    ])
    SqT_True_mean = np.mean(temp_SqT_True, axis=0)
    SqT_True_std = np.std(temp_SqT_True, axis=0)

    temp_SqT_Pred = np.array([
        model.predict(inputs, verbose=0).flatten() for model in SqT_Pred_models
    ])
    SqT_Pred_mean = np.mean(temp_SqT_Pred, axis=0)
    SqT_Pred_std = np.std(temp_SqT_Pred, axis=0)

    dSqT = np.abs(SqT_True_mean - SqT_Pred_mean)

    # Create DataFrame
    df = pd.DataFrame({
        'x1': x1_flat,
        'x2': x2_flat,
        'qT': qT_flat,
        'SqT_true_mean': SqT_True_mean,
        'SqT_true_std': SqT_True_std,
        'SqT_pred_mean': SqT_Pred_mean,
        'SqT_pred_std': SqT_Pred_std,
        'dSqT': dSqT
    })

    return df




def main():
    create_folders(CSV_FOLDER)
    starttime = datetime.datetime.now().replace(microsecond=0)
    df_xb_02 = generate_grids_fixed_x('x2',0.2,100)
    df_xb_02.to_csv(str(CSV_FOLDER)+'/SqT_xb_02.csv', index=False)
    df_xb_04 = generate_grids_fixed_x('x2',0.4,100)
    df_xb_04.to_csv(str(CSV_FOLDER)+'/SqT_xb_04.csv', index=False)
    df_xb_06 = generate_grids_fixed_x('x2',0.6,100)
    df_xb_06.to_csv(str(CSV_FOLDER)+'/SqT_xb_06.csv', index=False)
    finishtime = datetime.datetime.now().replace(microsecond=0)
    totalduration = finishtime - starttime
    print(f"Total duration --> {totalduration}")

if __name__ == "__main__":
    main()




