import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lhapdf
from functions_and_constants import *

# Configuration variables for grid generation
NUM_XF_VALUES = 5  # Number of xF values to generate
NUM_X1_VALUES = 10  # Number of x1 values for each xF
NUM_QT_VALUES = 20  # Number of qT values
NUM_QM_VALUES = 20  # Number of QM values

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

data = pd.concat([E288_200, E288_300, E288_400], ignore_index=True)

models_folder = '../../Step_by_step_tuning_to_get_sqT/Test_68/Models'

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

# Load All Trained Models with proper custom objects
model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
models_list = [tf.keras.models.load_model(
    os.path.join(models_folder, f), 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss  # In case the model used this name
    }
) for f in model_files]

print(f"Loaded {len(models_list)} models from '{models_folder}'.")

######### Generate grid for model outputs #################

# Get data ranges for grid creation
qT_min, qT_max = data['qT'].min(), data['qT'].max()
QM_min, QM_max = data['QM'].min(), data['QM'].max()
xF_min, xF_max = data['xF'].min(), data['xF'].max()
x1_min, x1_max = data['x1'].min(), data['x1'].max()
x2_min, x2_max = data['x2'].min(), data['x2'].max()

# Generate the grid
xF_values = np.linspace(xF_min, xF_max, NUM_XF_VALUES)
qT_values = np.linspace(qT_min, qT_max, NUM_QT_VALUES)
QM_values = np.linspace(QM_min, QM_max, NUM_QM_VALUES)

# Initialize the grid dataframe
grid_data = {
    'qT': [],
    'x1': [],
    'x2': [],
    'xF': [],
    'QM': [],
    'SqT_mean': [],
    'SqT_std': [],
    'BQM_mean': [],
    'BQM_std': [],
    'B_mean': [],    # B = sqrt(BQM)
    'B_std': []
}

# Create grid combinations
for xF in xF_values:
    # For each xF, generate x1 values and compute corresponding x2 values
    x1_values = np.linspace(max(xF, x1_min), min(x1_max, xF + x2_max), NUM_X1_VALUES)
    
    for x1 in x1_values:
        # Calculate x2 from xF and x1: xF = x1 - x2 => x2 = x1 - xF
        x2 = x1 - xF
        
        # Skip if x2 is outside the valid range
        if x2 < x2_min or x2 > x2_max:
            continue
        
        for qT in qT_values:
            for QM in QM_values:
                # Store grid point kinematics
                grid_data['qT'].append(qT)
                grid_data['x1'].append(x1)
                grid_data['x2'].append(x2)
                grid_data['xF'].append(xF)
                grid_data['QM'].append(QM)
                
                # Prepare inputs for model predictions
                qT_input = np.array([[qT]], dtype='float32')
                QM_input = np.array([[QM]], dtype='float32')
                x1_input = np.array([[x1]], dtype='float32')
                x2_input = np.array([[x2]], dtype='float32')
                
                # Compute PDFs (to be consistent with original code)
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
                
                PDFs_x1x2 = (eu2*f_u_x1 * f_ubar_x2 + 
                           ed2*f_d_x1 * f_dbar_x2 + 
                           es2*f_s_x1 * f_sbar_x2)
                
                PDFs_x2x1 = (eu2*f_u_x2 * f_ubar_x1 + 
                           ed2*f_d_x2 * f_dbar_x1 + 
                           es2*f_s_x2 * f_sbar_x1)
                
                PDFs_x1x2_input = np.array([[PDFs_x1x2]], dtype='float32')
                PDFs_x2x1_input = np.array([[PDFs_x2x1]], dtype='float32')
                
                # Collect predictions from all models
                SqT_preds = []
                BQM_preds = []
                
                for model in models_list:
                    # Get sub-models
                    SqT_model = model.get_layer('SqT')
                    BQM_model = model.get_layer('BQM')
                    
                    # For the SqT model
                    concat_input = np.hstack([qT_input, x1_input, x2_input])
                    SqT_pred = SqT_model.predict(concat_input, verbose=0).flatten()[0]
                    SqT_preds.append(SqT_pred)
                    
                    # For the BQM model
                    BQM_pred = BQM_model.predict(QM_input, verbose=0).flatten()[0]
                    BQM_preds.append(BQM_pred)
                
                # Calculate mean and std for the predictions
                SqT_mean = np.mean(SqT_preds)
                SqT_std = np.std(SqT_preds)
                BQM_mean = np.mean(BQM_preds)
                BQM_std = np.std(BQM_preds)
                
                # Calculate B (sqrt of BQM)
                B_mean = np.sqrt(BQM_mean)
                B_std = np.sqrt(BQM_std)
                
                # Store prediction results
                grid_data['SqT_mean'].append(SqT_mean)
                grid_data['SqT_std'].append(SqT_std)
                grid_data['BQM_mean'].append(BQM_mean)
                grid_data['BQM_std'].append(BQM_std)
                grid_data['B_mean'].append(B_mean)
                grid_data['B_std'].append(B_std)

# Create DataFrame and save to CSV
grid_df = pd.DataFrame(grid_data)
csv_filename = f'{results_folder}/grid_results_xF{NUM_XF_VALUES}_x1{NUM_X1_VALUES}_qT{NUM_QT_VALUES}_QM{NUM_QM_VALUES}.csv'
grid_df.to_csv(csv_filename, index=False)
print(f"Grid generated and saved to: {csv_filename}")
print(f"Grid shape: {grid_df.shape} - {len(grid_data['qT'])} total points")
print(f"Sample grid point: {grid_df.iloc[0].to_dict()}")

# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import lhapdf
# from functions_and_constants import *

# def create_folders(folder_name):
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#         print(f"Folder '{folder_name}' created successfully!")
#     else:
#         print(f"Folder '{folder_name}' already exists!")

# # Create Results Folder
# results_folder = 'Results_csvs'
# create_folders(results_folder)

# # Load LHAPDF Set
# NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')

# def pdf(pdfset, flavor, x, QQ):
#     return pdfset.xfxQ(flavor, x, QQ)

# # Load Data
# E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
# E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
# E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
# # E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E605.csv")
# # E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E772.csv")


# # data = pd.concat([E288_200,E288_300,E288_400,E605,E772], ignore_index=True)
# # data = pd.concat([E288_400], ignore_index=True)
# data = pd.concat([E288_200,E288_300,E288_400], ignore_index=True)

# # data = pd.concat([E288_200])

# models_folder = '../../Step_by_step_tuning_to_get_sqT/Test_68/Models'

# # Here we obtain all the values of kinematics from the combined data file
# x1_values = tf.constant(data['x1'].values, dtype=tf.float32)
# x2_values = tf.constant(data['x2'].values, dtype=tf.float32)
# qT_values = tf.constant(data['qT'].values, dtype=tf.float32)
# QM_values = tf.constant(data['QM'].values, dtype=tf.float32)
# xF_values = tf.constant(data['xF'].values, dtype=tf.float32)
# A_true_values = tf.constant(data['A'].values, dtype=tf.float32)



# def custom_weighted_loss(y_true, y_pred, w=None):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
    
#     if w is not None:
#         w = tf.cast(w, tf.float32)
#         mean_w = tf.reduce_mean(w)
#         weights = w / mean_w
#         squared_error = tf.square(y_pred - y_true)
#         weighted_squared_error = squared_error * weights
#         return tf.reduce_mean(weighted_squared_error)
#     # For model loading (when w not available), fall back to MSE
#     else:
#         return tf.reduce_mean(tf.square(y_pred - y_true))

# # Create a wrapper class to make the loss function serializable
# class CustomWeightedLoss(tf.keras.losses.Loss):
#     def __init__(self, name="custom_weighted_loss"):
#         super().__init__(name=name)
    
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         return custom_weighted_loss(y_true, y_pred)

# # Load All Trained Models with proper custom objects
# model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
# models_list = [tf.keras.models.load_model(
#     os.path.join(models_folder, f), 
#     custom_objects={
#         'custom_weighted_loss': custom_weighted_loss,
#         'CustomWeightedLoss': CustomWeightedLoss,
#         'train_weighted_loss': custom_weighted_loss  # In case the model used this name
#     }
# ) for f in model_files]

# print(f"Loaded {len(models_list)} models from '{models_folder}'.")



# ######### Generate csv files for B(QM)  comparisons #################


# # Generate qT, QM Ranges for Comparison
# QM_values = np.linspace(data['QM'].min(), data['QM'].max(), 200)
# qT_values = np.linspace(data['qT'].min(), data['qT'].max(), 200)
# xF_values = np.linspace(data['xF'].min(), data['xF'].max(), 200)
# x1_values = np.linspace(data['x1'].min(), data['x1'].max(), 200)
# x2_values = np.linspace(data['x2'].min(), data['x2'].max(), 200)



# # Compute PDFs
# f_u_x1 = pdf(NNPDF4_nlo, 2, x1_values, QM_values)
# f_ubar_x2 = pdf(NNPDF4_nlo, -2, x2_values, QM_values)
# f_u_x2 = pdf(NNPDF4_nlo, 2, x2_values, QM_values)
# f_ubar_x1 = pdf(NNPDF4_nlo, -2, x1_values, QM_values)
# f_d_x1 = pdf(NNPDF4_nlo, 1, x1_values, QM_values)
# f_dbar_x2 = pdf(NNPDF4_nlo, -1, x2_values, QM_values)
# f_d_x2 = pdf(NNPDF4_nlo, 1, x2_values, QM_values)
# f_dbar_x1 = pdf(NNPDF4_nlo, -1, x1_values, QM_values)
# f_s_x1 = pdf(NNPDF4_nlo, 3, x1_values, QM_values)
# f_sbar_x2 = pdf(NNPDF4_nlo, -3, x2_values, QM_values)
# f_s_x2 = pdf(NNPDF4_nlo, 3, x2_values, QM_values)
# f_sbar_x1 = pdf(NNPDF4_nlo, -3, x1_values, QM_values)

# PDFs_x1x2 = (eu2*np.array(f_u_x1) * np.array(f_ubar_x2) + 
#         ed2*np.array(f_d_x1) * np.array(f_dbar_x2) + 
#         es2*np.array(f_s_x1) * np.array(f_sbar_x2))


# PDFs_x2x1 = (eu2*np.array(f_u_x2) * np.array(f_ubar_x1)+ 
#         ed2*np.array(f_d_x2) * np.array(f_dbar_x1)+ 
#         es2*np.array(f_s_x2) * np.array(f_sbar_x1))


# # Prepare contributions arrays
# sq_contributions = []
# bqm_contributions = []
# SB2_contributions = []


# # Iterate through your list of SB_models
# for model in models_list:
#     SqT_model = model.get_layer('SqT')
#     BQM_model = model.get_layer('BQM')
    
#     # Print model input info for debugging
#     print("Model input details:")
#     for i, inp in enumerate(model.inputs):
#         print(f"Input {i} name: {inp.name}, shape: {inp.shape}")
    
#     # Convert to numpy arrays with explicit shapes
#     n_samples = len(qT_values)
#     qT_reshaped = np.array(qT_values, dtype='float32').reshape(n_samples, 1)
#     QM_reshaped = np.array(QM_values, dtype='float32').reshape(n_samples, 1)
#     # xF_reshaped = np.array(xF_values, dtype='float32').reshape(n_samples, 1)
#     x1_reshaped = np.array(x1_values, dtype='float32').reshape(n_samples, 1)
#     x2_reshaped = np.array(x2_values, dtype='float32').reshape(n_samples, 1)
#     PDFs_x1x2_reshaped = np.array(PDFs_x1x2, dtype='float32').reshape(n_samples, 1)
#     PDFs_x2x1_reshaped = np.array(PDFs_x2x1, dtype='float32').reshape(n_samples, 1)
    
#     # For the SqT model
#     concat_input = np.hstack([qT_reshaped, x1_reshaped, x2_reshaped])
#     sq_vals = SqT_model.predict(concat_input, verbose=0).flatten()
    
#     # For the BQM model
#     bqm_vals = BQM_model.predict(QM_reshaped, verbose=0).flatten()
    
#     # For the full model - try an alternative approach with a dictionary
#     # This maps the input tensor names to the actual values
#     input_dict = {
#         'qT': qT_reshaped,
#         'QM': QM_reshaped,
#         'x1': x1_reshaped,
#         'x2': x2_reshaped,
#         'pdfs_x1x2' : PDFs_x1x2_reshaped,
#         'pdfs_x2x1' : PDFs_x2x1_reshaped
#     }
    
#     try:
#         # Try dictionary-based approach
#         SB2_vals = model.predict(input_dict, verbose=0).flatten()
#     except Exception as e1:
#         print(f"Dictionary input failed: {e1}")
#         try:
#             # Try with named argument approach
#             SB2_vals = model.predict(x={'qT': qT_reshaped, 'QM': QM_reshaped, 'x1': {x1_reshaped.shape}, 'x2': {x2_reshaped.shape}, 'pdfs_x1x2' : {PDFs_x1x2_reshaped.shape}, 'pdfs_x2x1': {PDFs_x2x1_reshaped.shape}}, verbose=0).flatten()
#         except Exception as e2:
#             print(f"Named argument input failed: {e2}")
#             try:
#                 # Last resort - create a single batch with all samples
#                 SB2_vals = model([qT_reshaped, QM_reshaped, x1_reshaped, x2_reshaped, PDFs_x1x2_reshaped, PDFs_x2x1_reshaped]).numpy().flatten()
#             except Exception as e3:
#                 print(f"Direct model call failed: {e3}")
#                 # If all approaches fail, create dummy values
#                 SB2_vals = np.zeros_like(sq_vals)
#                 print("Using dummy values for SB2 due to prediction failures")
    
#     sq_contributions.append(sq_vals)
#     bqm_contributions.append(bqm_vals)
#     SB2_contributions.append(SB2_vals)



# # Convert to numpy arrays if desired
# sq_contributions = np.array(sq_contributions)
# bqm_contributions = np.array(bqm_contributions)
# SB2_contributions = np.array(SB2_contributions)


# # Element-wise mean and std across models
# SqTmean = np.mean(sq_contributions, axis=0)
# SqTstd = np.std(sq_contributions, axis=0)

# # print(bqm_contributions)
# B2QMmean = np.mean(bqm_contributions, axis=0)
# B2QMstd = np.std(bqm_contributions, axis=0)

# # Assuming BQM is always positive, this makes sense
# Bmean_values = np.sqrt(B2QMmean)
# Bstd_values = np.sqrt(B2QMstd)

# SB2mean = np.mean(SB2_contributions, axis=0)
# SB2std = np.std(SB2_contributions, axis=0)




# temp_df = {'qT': [],
#     'QM': [],
#     'B2_calc_mean': [],
#     'B2_calc_std': [],
#     'B_calc_mean': [],
#     'B_calc_std': [],
#     'SqT_mean': [],
#     'SqT_std': [],
#     'SB2mean':[],
#     'SB2std':[]}
    

# temp_df['qT'] = qT_values
# temp_df['QM'] = QM_values

# temp_df['B2_calc_mean'] = B2QMmean
# temp_df['B2_calc_std'] = B2QMstd

# temp_df['B_calc_mean'] = Bmean_values
# temp_df['B_calc_std'] = Bstd_values

# temp_df['SqT_mean'] = SqTmean
# temp_df['SqT_std'] = SqTstd

# temp_df['SB2mean'] = SB2mean
# temp_df['SB2std'] = SB2std


# results_csv_df = pd.DataFrame(temp_df)
# results_csv_df.to_csv(f'{results_folder}/comparison_results.csv')
# print("CSV files are generated and saved!")

# # ## prep_BQM_and_SqT_Grid(k,qT,QM,B2true,B2mean,B2std,Btrue,Bmean,Bstd,SqT,Sk,SB2mean,SB2std,SkBQMmean,SkBQMstd)
# # results_csv_df = prep_BQM_and_SqT_Grid(k_values,qT_values,QM_values,B2True_values,B2mean_values,B2std_values,BTrue_values,Bmean_values,Bstd_values,SqT_calc,Sk_calc,SB2_product_true,SB2_product_mean,SB2_product_stdv,SkBQM_product_true,SkBQM_product_mean,SkBQM_product_stdv)
