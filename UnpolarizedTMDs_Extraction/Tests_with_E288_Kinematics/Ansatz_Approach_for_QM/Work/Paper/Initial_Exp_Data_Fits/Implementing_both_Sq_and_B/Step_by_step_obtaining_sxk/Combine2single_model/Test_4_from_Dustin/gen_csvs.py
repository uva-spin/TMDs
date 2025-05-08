import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lhapdf
from functions_and_constants import *

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Create Results Folder
results_folder = 'Results_csvs'
create_folders(results_folder)

# Load LHAPDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Load Data
E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
# E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E605.csv")
# E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E772.csv")


# data = pd.concat([E288_200,E288_300,E288_400,E605,E772], ignore_index=True)
# data = pd.concat([E288_400], ignore_index=True)
data = pd.concat([E288_200,E288_300,E288_400], ignore_index=True)

# data = pd.concat([E288_200])

models_folder = 'Models'


x1_values = tf.constant(data['x1'].values, dtype=tf.float32)
x2_values = tf.constant(data['x2'].values, dtype=tf.float32)
qT_values = tf.constant(data['qT'].values, dtype=tf.float32)
QM_values = tf.constant(data['QM'].values, dtype=tf.float32)
xF_values = tf.constant(data['xF'].values, dtype=tf.float32)
A_true_values = tf.constant(data['A'].values, dtype=tf.float32)



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
    # For model loading (when w not available), fall back to MSE
    else:
        return tf.reduce_mean(tf.square(y_pred - y_true))

# Create a wrapper class to make the loss function serializable
class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_weighted_loss"):
        super().__init__(name=name)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        return custom_weighted_loss(y_true, y_pred)

# Load the average nodel with proper custom objects
model = tf.keras.models.load_model('averaged_model.h5', 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss
    })

print("Loaded the averaged model")



def QM_int(QM):
    return (-1)/(2*QM**2)

def compute_QM_integrals(QM_array):
    QM_array = np.atleast_1d(QM_array) 
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated




### Cross-Section #####

# Compute A Predictions
def compute_A(x1, x2, qT, QM, pdfs_x1x2, pdfs_x2x1):
    # Get Predictions from All Models
    # fDNN_contributions = np.array(model.predict([qT,QM,x1,x2, pdfs_x1x2, pdfs_x2x1], verbose=0))
    # fDNN_mean = np.mean(fDNN_contributions, axis=0)
    # fDNN_std = np.std(fDNN_contributions, axis=0)
    fDNN_mean = model.predict([qT,QM,x1,x2, pdfs_x1x2, pdfs_x2x1], verbose=0).flatten()

    factor_temp = factor

    QM_integral_temp = compute_QM_integrals(QM)

    A_pred = fDNN_mean * factor_temp * hc_factor * QM_integral_temp
    # A_std = fDNN_std * factor_temp * hc_factor * QM_integral_temp

    return A_pred



def prep_data_for_plots(df):
    temp_df = {'x1': [],
        'x2': [],
        'qT': [],
        'QM': [],
        'xF': [],
        'pdfs_x1x2' : [],
        'pdfs_x2x1' : [],
        'A_true': [],
        'A_true_err': [],
        'A_pred': []}
    # ,        'A_pred_err': []

    qT = df['qT'].values
    QM = df['QM'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    xF = df['xF'].values

    # Compute PDFs
    f_u_x1 = pdf(NNPDF4_nlo, 2, x1, QM)
    f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2, QM)
    f_u_x2 = pdf(NNPDF4_nlo, 2, x2, QM)
    f_ubar_x1 = pdf(NNPDF4_nlo, -2, x1, QM)
    f_d_x1 = pdf(NNPDF4_nlo, 1, x1, QM)
    f_dbar_x2 = pdf(NNPDF4_nlo, -1, x2, QM)
    f_d_x2 = pdf(NNPDF4_nlo, 1, x2, QM)
    f_dbar_x1 = pdf(NNPDF4_nlo, -1, x1, QM)
    f_s_x1 = pdf(NNPDF4_nlo, 3, x1, QM)
    f_sbar_x2 = pdf(NNPDF4_nlo, -3, x2, QM)
    f_s_x2 = pdf(NNPDF4_nlo, 3, x2, QM)
    f_sbar_x1 = pdf(NNPDF4_nlo, -3, x1, QM)

    pdfs_x1x2 = (eu2*np.array(f_u_x1) * np.array(f_ubar_x2) + 
            ed2*np.array(f_d_x1) * np.array(f_dbar_x2) + 
            es2*np.array(f_s_x1) * np.array(f_sbar_x2))
    

    pdfs_x2x1 = (eu2*np.array(f_u_x2) * np.array(f_ubar_x1)+ 
            ed2*np.array(f_d_x2) * np.array(f_dbar_x1)+ 
            es2*np.array(f_s_x2) * np.array(f_sbar_x1))


    temp_df['pdfs_x1x2'] = pdfs_x1x2
    temp_df['pdfs_x2x1'] = pdfs_x2x1

    A_true = df['A'].values
    A_true_err = df['dA'].values
    A_pred = compute_A(x1, x2, qT, QM, pdfs_x1x2, pdfs_x2x1)

    temp_df['x1'] = x1
    temp_df['x2'] = x2
    temp_df['qT'] = qT
    temp_df['QM'] = QM
    temp_df['xF'] = xF
    temp_df['pdfs_x1x2'] = pdfs_x1x2
    temp_df['pdfs_x2x1'] = pdfs_x2x1
    temp_df['A_true'] = A_true
    temp_df['A_true_err'] = A_true_err
    temp_df['A_pred'] = A_pred
    # temp_df['A_pred_err'] = A_std

    return pd.DataFrame(temp_df)



E288_200_result_df = prep_data_for_plots(E288_200)
E288_200_result_df.to_csv(f'{results_folder}/E288_200_results.csv')

E288_300_result_df = prep_data_for_plots(E288_300)
E288_300_result_df.to_csv(f'{results_folder}/E288_300_results.csv')

E288_400_result_df = prep_data_for_plots(E288_400)
E288_400_result_df.to_csv(f'{results_folder}/E288_400_results.csv')


E288_200_with_cuts = E288_200_result_df
#E288_200_with_cuts=E288_200_result_df[E288_200_result_df['qT'] < 0.2 * E288_200_result_df['QM']]
E288_200_with_cuts = E288_200_with_cuts[(9.0 > E288_200_with_cuts['QM']) | (E288_200_with_cuts['QM'] > 11.0)]
E288_200_with_cuts.to_csv(f'{results_folder}/E288_200_results_with_cuts.csv')

E288_300_with_cuts = E288_300_result_df
#E288_300_with_cuts=E288_300_result_df[E288_300_result_df['qT'] < 0.2 * E288_300_result_df['QM']]
E288_300_with_cuts = E288_300_with_cuts[(9.0 > E288_300_with_cuts['QM']) | (E288_300_with_cuts['QM'] > 11.0)]
E288_300_with_cuts.to_csv(f'{results_folder}/E288_300_results_with_cuts.csv')

E288_400_with_cuts = E288_400_result_df
#E288_400_with_cuts=E288_400_result_df[E288_400_result_df['qT'] < 0.2 * E288_400_result_df['QM']]
E288_400_with_cuts = E288_400_with_cuts[(9.0 > E288_400_with_cuts['QM']) | (E288_400_with_cuts['QM'] > 11.0)]
E288_400_with_cuts.to_csv(f'{results_folder}/E288_400_results_with_cuts.csv')

