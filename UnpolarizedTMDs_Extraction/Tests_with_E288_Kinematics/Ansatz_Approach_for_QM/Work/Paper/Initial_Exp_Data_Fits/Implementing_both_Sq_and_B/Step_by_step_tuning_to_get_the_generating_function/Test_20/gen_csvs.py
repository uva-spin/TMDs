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
A_true_values = tf.constant(data['A'].values, dtype=tf.float32)



# Define Custom Loss Function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # MSE loss

# Load All Trained Models
model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
models_list = [tf.keras.models.load_model(os.path.join(models_folder, f), custom_objects={'custom_loss': custom_loss}) for f in model_files]


print(f"Loaded {len(models_list)} models from '{models_folder}'.")


# def QM_int(QM):
#     return (-1)/(2*QM**2)

def QM_int(QM):
    return np.log(QM)

def compute_QM_integrals(QM_array):
    QM_array = np.atleast_1d(QM_array) 
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated




### Cross-Section #####

# Compute A Predictions
def compute_A(x1, x2, qT, QM, SqrtS):
    # Get Predictions from All Models
    fDNN_contributions = np.array([model.predict([qT,QM], verbose=0).flatten() for model in models_list])
    fDNN_mean = np.mean(fDNN_contributions, axis=0)
    fDNN_std = np.std(fDNN_contributions, axis=0)

    factor_temp = factor

    f_u_x1 = (1/x1)*pdf(NNPDF4_nlo, 2, x1, QM) 
    f_ubar_x2 = (1/x2)*pdf(NNPDF4_nlo, -2, x2, QM)
    f_u_x2 = (1/x2)*pdf(NNPDF4_nlo, 2, x2, QM)
    f_ubar_x1 = (1/x1)*pdf(NNPDF4_nlo, -2, x1, QM)
    f_d_x1 = (1/x1)*pdf(NNPDF4_nlo, 1, x1, QM) 
    f_dbar_x2 = (1/x2)*pdf(NNPDF4_nlo, -1, x2, QM)
    f_d_x2 = (1/x2)*pdf(NNPDF4_nlo, 1, x2, QM)
    f_dbar_x1 = (1/x1)*pdf(NNPDF4_nlo, -1, x1, QM)
    f_s_x1 = (1/x1)*pdf(NNPDF4_nlo, 3, x1, QM) 
    f_sbar_x2 = (1/x2)*pdf(NNPDF4_nlo, -3, x2, QM)
    f_s_x2 = (1/x2)*pdf(NNPDF4_nlo, 3, x2, QM)
    f_sbar_x1 = (1/x1)*pdf(NNPDF4_nlo, -3, x1, QM)



    ux1ubarx2_temp = eu2*np.array(f_u_x1)*np.array(f_ubar_x2)
    ubarx1ux2_temp = eu2*np.array(f_u_x2)*np.array(f_ubar_x1)
    dx1dbarx2_temp = ed2*np.array(f_d_x1)*np.array(f_dbar_x2)
    dbarx1dx2_temp = ed2*np.array(f_d_x2)*np.array(f_dbar_x1)
    sx1sbarx2_temp = es2*np.array(f_s_x1)*np.array(f_sbar_x2)
    sbarx1sx2_temp = es2*np.array(f_s_x2)*np.array(f_sbar_x1)

    PDFs_temp = ux1ubarx2_temp + ubarx1ux2_temp + dx1dbarx2_temp + dbarx1dx2_temp + sx1sbarx2_temp + sbarx1sx2_temp


    QM_integral_temp = compute_QM_integrals(QM)

    qTcorr = 1 + ((qT*qT)/(QM*QM))
    A_pred = fDNN_mean * factor_temp * PDFs_temp * hc_factor * QM_integral_temp /(SqrtS*SqrtS*qTcorr)
    A_std = fDNN_std * factor_temp * PDFs_temp * hc_factor * QM_integral_temp /(SqrtS*SqrtS*qTcorr)

    return A_pred, A_std


def prep_data_for_plots(df):
    temp_df = {'x1': [],
        'x2': [],
        'qT': [],
        'QM': [],
        'A_true': [],
        'A_true_err': [],
        'A_pred': [],
        'A_pred_err': []}

    qT = df['qT'].values
    QM = df['QM'].values
    x1 = df['x1'].values
    x2 = df['x2'].values
    SqrtS = df['SqrtS'].values
    A_true = df['A'].values
    A_true_err = df['dA'].values
    A_pred, A_std = compute_A(x1, x2, qT, QM, SqrtS)

    temp_df['x1'] = x1
    temp_df['x2'] = x2
    temp_df['qT'] = qT
    temp_df['QM'] = QM
    temp_df['A_true'] = A_true
    temp_df['A_true_err'] = A_true_err
    temp_df['A_pred'] = A_pred
    temp_df['A_pred_err'] = A_std

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





######### Generate csv files for B(QM)  comparisons #################


# Generate qT, QM Ranges for Comparison
QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
qT_values = np.linspace(data['qT'].min(), data['qT'].max(), 200)
k_values = np.linspace(data['qT'].min(), data['qT'].max(), 200)


# Prepare contributions arrays
sq_contributions = []
bqm_contributions = []
SB2_contributions = []

# Iterate through your list of SB_models
for model in models_list:
    SqT_model = model.get_layer('SqT')
    BQM_model = model.get_layer('BQM')
    SB2_model = model
    
    # Predict with each sub-model separately
    sq_vals = SqT_model.predict(qT_values, verbose=0).flatten()
    bqm_vals = BQM_model.predict(QM_values, verbose=0).flatten()
    SB2_vals = model.predict([qT_values,QM_values], verbose=0).flatten()
    
    sq_contributions.append(sq_vals)
    bqm_contributions.append(bqm_vals)
    SB2_contributions.append(SB2_vals)

# Convert to numpy arrays if desired
sq_contributions = np.array(sq_contributions)
bqm_contributions = np.array(bqm_contributions)
SB2_contributions = np.array(SB2_contributions)


# Element-wise mean and std across models
SqTmean = np.mean(sq_contributions, axis=0)
SqTstd = np.std(sq_contributions, axis=0)

# print(bqm_contributions)
B2QMmean = np.mean(bqm_contributions, axis=0)
B2QMstd = np.std(bqm_contributions, axis=0)

# Assuming BQM is always positive, this makes sense
Bmean_values = np.sqrt(B2QMmean)
Bstd_values = np.sqrt(B2QMstd)

SB2mean = np.mean(SB2_contributions, axis=0)
SB2std = np.std(SB2_contributions, axis=0)

# SqTmean = np.mean(sq_contributions)
# SqTstd = np.std(sq_contributions)

# B2QMmean = np.mean(bqm_contributions)
# B2QMstd = np.std(bqm_contributions)

# Bmean_values = np.sqrt(B2QMmean)
# Bstd_values = np.sqrt(B2QMstd)

# SB2mean = np.mean(SB2_contributions)
# SB2std = np.std(SB2_contributions)




temp_df = {'qT': [],
    'QM': [],
    'B2_calc_mean': [],
    'B2_calc_std': [],
    'B_calc_mean': [],
    'B_calc_std': [],
    'SqT_mean': [],
    'SqT_std': [],
    'SB2mean':[],
    'SB2std':[]}
    

temp_df['qT'] = qT_values
temp_df['QM'] = QM_values

temp_df['B2_calc_mean'] = B2QMmean
temp_df['B2_calc_std'] = B2QMstd

temp_df['B_calc_mean'] = Bmean_values
temp_df['B_calc_std'] = Bstd_values

temp_df['SqT_mean'] = SqTmean
temp_df['SqT_std'] = SqTstd

temp_df['SB2mean'] = SB2mean
temp_df['SB2std'] = SB2std


results_csv_df = pd.DataFrame(temp_df)
results_csv_df.to_csv(f'{results_folder}/comparison_results.csv')
print("CSV files are generated and saved!")

# ## prep_BQM_and_SqT_Grid(k,qT,QM,B2true,B2mean,B2std,Btrue,Bmean,Bstd,SqT,Sk,SB2mean,SB2std,SkBQMmean,SkBQMstd)
# results_csv_df = prep_BQM_and_SqT_Grid(k_values,qT_values,QM_values,B2True_values,B2mean_values,B2std_values,BTrue_values,Bmean_values,Bstd_values,SqT_calc,Sk_calc,SB2_product_true,SB2_product_mean,SB2_product_stdv,SkBQM_product_true,SkBQM_product_mean,SkBQM_product_stdv)
