import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import lhapdf
import time
from functions_and_constants import *

# Configuration variables for grid generation
NUM_XF_VALUES = 5  # Number of xF values to generate
NUM_X1_VALUES = 10  # Number of x1 values for each xF
NUM_QT_VALUES = 20  # Number of qT values
NUM_QM_VALUES = 20  # Number of QM values
BATCH_SIZE = 1024   # Batch size for GPU processing

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

# First prepare the full grid of inputs
start_time = time.time()
print("Building grid points...")

valid_combinations = []
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
                valid_combinations.append((qT, x1, x2, xF, QM))

print(f"Generated {len(valid_combinations)} valid grid points in {time.time() - start_time:.2f} seconds")

# Prepare arrays for batch processing
qT_array = np.array([combo[0] for combo in valid_combinations], dtype=np.float32)
x1_array = np.array([combo[1] for combo in valid_combinations], dtype=np.float32)
x2_array = np.array([combo[2] for combo in valid_combinations], dtype=np.float32)
xF_array = np.array([combo[3] for combo in valid_combinations], dtype=np.float32)
QM_array = np.array([combo[4] for combo in valid_combinations], dtype=np.float32)

# Initialize result arrays
SqT_means = np.zeros(len(valid_combinations), dtype=np.float32)
SqT_stds = np.zeros(len(valid_combinations), dtype=np.float32)
BQM_means = np.zeros(len(valid_combinations), dtype=np.float32)
BQM_stds = np.zeros(len(valid_combinations), dtype=np.float32)
B_means = np.zeros(len(valid_combinations), dtype=np.float32)
B_stds = np.zeros(len(valid_combinations), dtype=np.float32)

total_batches = int(np.ceil(len(valid_combinations) / BATCH_SIZE))

# Process batch by batch
print("Processing predictions in batches...")
for batch_idx in range(total_batches):
    # Simple progress tracking
    if batch_idx % max(1, total_batches // 10) == 0:
        progress = batch_idx / total_batches * 100
        print(f"Progress: {progress:.1f}% - Batch {batch_idx}/{total_batches}")
    
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(valid_combinations))
    
    batch_size = end_idx - start_idx
    
    # Get batch data
    qT_batch = qT_array[start_idx:end_idx].reshape(batch_size, 1)
    x1_batch = x1_array[start_idx:end_idx].reshape(batch_size, 1)
    x2_batch = x2_array[start_idx:end_idx].reshape(batch_size, 1)
    QM_batch = QM_array[start_idx:end_idx].reshape(batch_size, 1)
    
    # Build SqT input tensor (concatenated)
    SqT_input = np.hstack([qT_batch, x1_batch, x2_batch])
    
    # Compute PDFs in batch
    PDFs_x1x2_batch = np.zeros(batch_size)
    PDFs_x2x1_batch = np.zeros(batch_size)
    
    # Batch computation of PDFs might be tricky with lhapdf, so we'll loop here
    for i in range(batch_size):
        x1, x2, QM = x1_batch[i, 0], x2_batch[i, 0], QM_batch[i, 0]
        
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
        
        PDFs_x1x2_batch[i] = (eu2*f_u_x1 * f_ubar_x2 + 
                          ed2*f_d_x1 * f_dbar_x2 + 
                          es2*f_s_x1 * f_sbar_x2)
        
        PDFs_x2x1_batch[i] = (eu2*f_u_x2 * f_ubar_x1 + 
                          ed2*f_d_x2 * f_dbar_x1 + 
                          es2*f_s_x2 * f_sbar_x1)
    
    # Reshape for model input
    PDFs_x1x2_batch = PDFs_x1x2_batch.reshape(batch_size, 1)
    PDFs_x2x1_batch = PDFs_x2x1_batch.reshape(batch_size, 1)
    
    # Lists to store predictions from each model
    SqT_model_preds = []
    BQM_model_preds = []
    
    for model in models_list:
        # Get sub-models
        SqT_model = model.get_layer('SqT')
        BQM_model = model.get_layer('BQM')
        
        # Batch predict for both models
        SqT_pred_batch = SqT_model.predict(SqT_input, verbose=0, batch_size=batch_size)
        BQM_pred_batch = BQM_model.predict(QM_batch, verbose=0, batch_size=batch_size)
        
        SqT_model_preds.append(SqT_pred_batch)
        BQM_model_preds.append(BQM_pred_batch)
    
    # Convert to arrays for easier calculation
    SqT_all_preds = np.array(SqT_model_preds)  # Shape: [num_models, batch_size, 1]
    BQM_all_preds = np.array(BQM_model_preds)  # Shape: [num_models, batch_size, 1]
    
    # Calculate mean and std across models (axis 0)
    SqT_mean_batch = np.mean(SqT_all_preds, axis=0).flatten()
    SqT_std_batch = np.std(SqT_all_preds, axis=0).flatten()
    BQM_mean_batch = np.mean(BQM_all_preds, axis=0).flatten()
    BQM_std_batch = np.std(BQM_all_preds, axis=0).flatten()
    
    # Calculate B (sqrt of BQM)
    B_mean_batch = np.sqrt(BQM_mean_batch)
    B_std_batch = np.sqrt(BQM_std_batch)
    
    # Store the results
    SqT_means[start_idx:end_idx] = SqT_mean_batch
    SqT_stds[start_idx:end_idx] = SqT_std_batch
    BQM_means[start_idx:end_idx] = BQM_mean_batch
    BQM_stds[start_idx:end_idx] = BQM_std_batch
    B_means[start_idx:end_idx] = B_mean_batch
    B_stds[start_idx:end_idx] = B_std_batch

# Store results in the grid_data dictionary
grid_data['qT'] = qT_array
grid_data['x1'] = x1_array
grid_data['x2'] = x2_array
grid_data['xF'] = xF_array
grid_data['QM'] = QM_array
grid_data['SqT_mean'] = SqT_means
grid_data['SqT_std'] = SqT_stds
grid_data['BQM_mean'] = BQM_means
grid_data['BQM_std'] = BQM_stds
grid_data['B_mean'] = B_means
grid_data['B_std'] = B_stds

print(f"Predictions completed in {time.time() - start_time:.2f} seconds total")

# Create DataFrame and save to CSV
grid_df = pd.DataFrame(grid_data)
csv_filename = f'{results_folder}/grid_results_xF{NUM_XF_VALUES}_x1{NUM_X1_VALUES}_qT{NUM_QT_VALUES}_QM{NUM_QM_VALUES}.csv'
grid_df.to_csv(csv_filename, index=False)
print(f"Grid generated and saved to: {csv_filename}")
print(f"Grid shape: {grid_df.shape} - {len(grid_data['qT'])} total points")
print(f"Sample grid point: {grid_df.iloc[0].to_dict()}")