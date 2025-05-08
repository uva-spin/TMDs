import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

# Load Data
try:
    E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
    E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
    E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
    # E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E605.csv")
    # E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E772.csv")
    # data = pd.concat([E288_200,E288_300,E288_400,E605,E772], ignore_index=True)
    # data = pd.concat([E288_400], ignore_index=True)
    data = pd.concat([E288_200, E288_300, E288_400], ignore_index=True)
except FileNotFoundError:
    print("Warning: Data files not found. Using a fallback method for testing.")
    # Create dummy data if files cannot be loaded (for testing purposes)
    data = pd.DataFrame({
        'qT': np.linspace(0.1, 5.0, 100),
        'QM': np.ones(100) * 91.0,
        'x1': np.ones(100) * 0.3,
        'x2': np.ones(100) * 0.2,
        'pdfs_x1x2': np.ones(100),
        'pdfs_x2x1': np.ones(100),
        'dsigma': np.random.rand(100) * 100,
        'error': np.random.rand(100) * 10
    })

models_folder = 'Models'
if not os.path.exists(models_folder):
    models_folder = '../../../Step_by_step_tuning_to_get_sqT/Test_68/Models'

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
        return custom_weighted_loss(y_true, y_pred, sample_weight)

model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
models_list = [tf.keras.models.load_model(
    os.path.join(models_folder, f), 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss}) for f in model_files]

### Cross-Section #####

alpha = 1/137
hc_factor = 3.89 * 10**8
factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)

def QM_int(QM):
    return (-1)/(2*QM**2)

def compute_QM_integrals(QM_array):
    QM_array = np.atleast_1d(QM_array) 
    QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
    return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated

# Compute A Predictions
def compute_A(x1, x2, qT, QM, pdfs_x1x2, pdfs_x2x1):
    # Get Predictions from All Models
    fDNN_contributions = np.array([model.predict([qT, QM, x1, x2, pdfs_x1x2, pdfs_x2x1], verbose=0).flatten() for model in models_list])
    fDNN_mean = np.mean(fDNN_contributions, axis=0)
    fDNN_std = np.std(fDNN_contributions, axis=0)
    
    # Calculate 95% confidence interval using scipy.stats
    n_models = len(models_list)
    confidence = 0.95
    degrees_freedom = n_models - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
    margin_of_error = t_critical * (fDNN_std / np.sqrt(n_models))
    confidence_interval = (fDNN_mean - margin_of_error, fDNN_mean + margin_of_error)

    factor_temp = factor
    QM_integral_temp = compute_QM_integrals(QM)

    A_pred = fDNN_mean * factor_temp * hc_factor * QM_integral_temp
    A_std = fDNN_std * factor_temp * hc_factor * QM_integral_temp
    A_conf_interval = (
        confidence_interval[0] * factor_temp * hc_factor * QM_integral_temp,
        confidence_interval[1] * factor_temp * hc_factor * QM_integral_temp
    )

    return A_pred, A_std, A_conf_interval

def prepare_data_from_df(df):
    """
    Prepare input data from DataFrame for model prediction
    """
    qT = df['qT'].values.reshape(-1, 1)
    QM = df['QM'].values.reshape(-1, 1)
    x1 = df['x1'].values.reshape(-1, 1)
    x2 = df['x2'].values.reshape(-1, 1)
    
    # Check if the pdfs columns exist, otherwise create them with ones
    if 'pdfs_x1x2' in df.columns and 'pdfs_x2x1' in df.columns:
        pdfs_x1x2 = df['pdfs_x1x2'].values.reshape(-1, 1)
        pdfs_x2x1 = df['pdfs_x2x1'].values.reshape(-1, 1)
    else:
        pdfs_x1x2 = np.ones_like(qT)
        pdfs_x2x1 = np.ones_like(qT)
    
    return [qT, QM, x1, x2, pdfs_x1x2, pdfs_x2x1]

def find_closest_model_to_mean(models_list, data_df=None):
    """
    Find the model whose predictions are closest to the mean predictions of all models.
    
    Parameters:
    - models_list: List of trained models
    - data_df: DataFrame containing the experimental data
    
    Returns:
    - The model from models_list that has predictions closest to the ensemble mean
    """
    # Use the provided DataFrame for evaluation
    if data_df is None:
        data_df = data  # Use the global data variable
    
    # Prepare inputs from DataFrame
    input_data = prepare_data_from_df(data_df)
    
    # Get predictions from all models
    all_predictions = np.array([model.predict(input_data, verbose=0) for model in models_list])
    
    # Calculate mean predictions across all models
    mean_predictions = np.mean(all_predictions, axis=0)
    
    # Calculate MSE between each model's predictions and the mean predictions
    mse_to_mean = np.mean((all_predictions - mean_predictions)**2, axis=(1, 2))
    
    # Find the model with the smallest MSE to the mean predictions
    closest_model_idx = np.argmin(mse_to_mean)
    
    print(f"Model {model_files[closest_model_idx]} is closest to the ensemble mean")
    print(f"MSE to mean: {mse_to_mean[closest_model_idx]:.6f}")
    
    # Plot results by energy ranges (E288_200, E288_300, E288_400)
    # Create a plot to visualize the comparison
    plt.figure(figsize=(15, 10))
    
    # Plot all model predictions (faded)
    qT = input_data[0].flatten()
    for i, preds in enumerate(all_predictions):
        alpha = 0.2 if i != closest_model_idx else 0.8
        label = f"Closest model" if i == closest_model_idx else None
        plt.plot(qT, preds.flatten(), alpha=alpha, color='blue' if i == closest_model_idx else 'gray', 
                 linewidth=2 if i == closest_model_idx else 1, label=label)
    
    # Plot mean prediction
    plt.plot(qT, mean_predictions.flatten(), 'r-', linewidth=2, label='Ensemble Mean')
    
    # Calculate 95% confidence interval
    n_models = len(models_list)
    std_dev = np.std(all_predictions, axis=0)
    t_critical = stats.t.ppf(0.975, n_models - 1)  # 95% CI
    margin_error = t_critical * (std_dev / np.sqrt(n_models))
    
    lower_bound = mean_predictions - margin_error
    upper_bound = mean_predictions + margin_error
    
    # Plot confidence interval
    plt.fill_between(qT, lower_bound.flatten(), upper_bound.flatten(), 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('qT (GeV)')
    plt.ylabel('Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Create separate plots for each energy range
    energy_datasets = {
        'E288_200': E288_200 if 'E288_200' in locals() else None,
        'E288_300': E288_300 if 'E288_300' in locals() else None,
        'E288_400': E288_400 if 'E288_400' in locals() else None
    }
    
    plt.figure(figsize=(18, 12))
    
    for i, (name, ds) in enumerate(energy_datasets.items()):
        if ds is not None:
            plt.subplot(2, 2, i+1)
            
            # Prepare inputs for this energy dataset
            inputs = prepare_data_from_df(ds)
            
            # Get predictions
            all_preds = np.array([model.predict(inputs, verbose=0) for model in models_list])
            mean_preds = np.mean(all_preds, axis=0)
            std_dev = np.std(all_preds, axis=0)
            closest_preds = all_preds[closest_model_idx]
            
            # Calculate confidence interval
            t_critical = stats.t.ppf(0.975, n_models - 1)
            margin_error = t_critical * (std_dev / np.sqrt(n_models))
            lower = mean_preds - margin_error
            upper = mean_preds + margin_error
            
            # Plot
            qT_vals = inputs[0].flatten()
            plt.plot(qT_vals, mean_preds.flatten(), 'r-', linewidth=2, label='Ensemble Mean')
            plt.plot(qT_vals, closest_preds.flatten(), 'b--', linewidth=2, label='Selected Model')
            plt.fill_between(qT_vals, lower.flatten(), upper.flatten(), color='red', alpha=0.2, label='95% CI')
            
            # If actual cross section data is available, plot it
            if 'dsigma' in ds.columns and 'error' in ds.columns:
                plt.errorbar(ds['qT'], ds['dsigma'], yerr=ds['error'], fmt='ko', label='Experimental Data')
            
            plt.title(f'{name} Energy Range')
            plt.xlabel('qT (GeV)')
            plt.ylabel('Cross Section')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_comparisons.png')
    plt.close()
    
    return models_list[closest_model_idx]


def compare_model_with_data(model, data_df=None):
    """
    Compare model predictions with actual experimental data
    """
    if data_df is None:
        data_df = data
        
    input_data = prepare_data_from_df(data_df)
    
    # Get predictions
    predictions = model.predict(input_data, verbose=0)
    
    # Calculate cross-section values
    qT = input_data[0]
    QM = input_data[1]
    factor_temp = factor
    QM_integral_temp = compute_QM_integrals(QM)
    cross_section = predictions * factor_temp * hc_factor * QM_integral_temp
    
    # Plot comparison with experimental data if available
    if 'dsigma' in data_df.columns and 'error' in data_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Plot model predictions
        plt.plot(qT, cross_section, 'r-', linewidth=2, label='Model Prediction')
        
        # Plot experimental data with error bars
        plt.errorbar(data_df['qT'], data_df['dsigma'], yerr=data_df['error'], 
                    fmt='ko', label='Experimental Data')
        
        plt.title('Model Predictions vs Experimental Data')
        plt.xlabel('qT (GeV)')
        plt.ylabel('Cross Section')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('model_vs_data.png')
        plt.close()
        
        # Calculate metrics
        mse = np.mean((cross_section.flatten() - data_df['dsigma'])**2)
        weighted_mse = np.mean(((cross_section.flatten() - data_df['dsigma']) / data_df['error'])**2)
        
        print(f"MSE with experimental data: {mse:.6f}")
        print(f"Weighted MSE (χ²/d.o.f): {weighted_mse:.6f}")
        
        return mse, weighted_mse
    
    return None

# Create specific plots for E288 data by QM bins
def plot_by_QM_bins(averaged_model):
    """Create plots comparing model predictions with data by QM bins"""
    
    # Define QM bins present in the data
    QM_bins = {
        'E288_200 (QM=4-5)': E288_200[E288_200['QM'].between(4, 5)],
        'E288_200 (QM=5-6)': E288_200[E288_200['QM'].between(5, 6)],
        'E288_200 (QM=6-7)': E288_200[E288_200['QM'].between(6, 7)],
        'E288_200 (QM=7-8)': E288_200[E288_200['QM'].between(7, 8)],
        'E288_200 (QM=8-9)': E288_200[E288_200['QM'].between(8, 9)],
        'E288_300 (QM=4-5)': E288_300[E288_300['QM'].between(4, 5)],
        'E288_300 (QM=5-6)': E288_300[E288_300['QM'].between(5, 6)],
        'E288_300 (QM=6-7)': E288_300[E288_300['QM'].between(6, 7)],
        'E288_300 (QM=7-8)': E288_300[E288_300['QM'].between(7, 8)],
        'E288_300 (QM=8-9)': E288_300[E288_300['QM'].between(8, 9)],
        'E288_300 (QM=10-11)': E288_300[E288_300['QM'].between(10, 11)],
        'E288_300 (QM=11-12)': E288_300[E288_300['QM'].between(11, 12)],
        'E288_400 (QM=4-5)': E288_400[E288_400['QM'].between(4, 5)],
        'E288_400 (QM=5-6)': E288_400[E288_400['QM'].between(5, 6)],
        'E288_400 (QM=6-7)': E288_400[E288_400['QM'].between(6, 7)],
        'E288_400 (QM=7-8)': E288_400[E288_400['QM'].between(7, 8)],
        'E288_400 (QM=8-9)': E288_400[E288_400['QM'].between(8, 9)],
        'E288_400 (QM=11-12)': E288_400[E288_400['QM'].between(11, 12)],
        'E288_400 (QM=12-13)': E288_400[E288_400['QM'].between(12, 13)],
        'E288_400 (QM=13-14)': E288_400[E288_400['QM'].between(13, 14)]
    }
    
    # Create a multi-page figure (multiple plots)
    fig_rows = 5
    fig_cols = 4
    fig_count = 0
    
    # Calculate how many figures needed
    n_figures = (len(QM_bins) + (fig_rows * fig_cols) - 1) // (fig_rows * fig_cols)
    
    for fig_idx in range(n_figures):
        plt.figure(figsize=(20, 24))
        
        for i in range(fig_rows * fig_cols):
            bin_idx = fig_idx * (fig_rows * fig_cols) + i
            if bin_idx >= len(QM_bins):
                break
                
            bin_name = list(QM_bins.keys())[bin_idx]
            bin_data = QM_bins[bin_name]
            
            if len(bin_data) == 0:
                continue
            
            plt.subplot(fig_rows, fig_cols, i+1)
            
            # Prepare inputs for this bin
            inputs = prepare_data_from_df(bin_data)
            
            # Get predictions from all models
            all_preds = np.array([model.predict(inputs, verbose=0) for model in models_list])
            mean_preds = np.mean(all_preds, axis=0)
            std_dev = np.std(all_preds, axis=0)
            closest_model_preds = averaged_model.predict(inputs, verbose=0)
            
            # Calculate confidence interval
            n_models = len(models_list)
            t_critical = stats.t.ppf(0.975, n_models - 1)
            margin_error = t_critical * (std_dev / np.sqrt(n_models))
            lower = mean_preds - margin_error
            upper = mean_preds + margin_error
            
            # Calculate cross section values
            qT = inputs[0]
            QM = inputs[1]
            QM_integral_temp = compute_QM_integrals(QM)
            
            # Convert predictions to cross sections
            closest_xsec = closest_model_preds * factor * hc_factor * QM_integral_temp
            mean_xsec = mean_preds * factor * hc_factor * QM_integral_temp
            lower_xsec = lower * factor * hc_factor * QM_integral_temp
            upper_xsec = upper * factor * hc_factor * QM_integral_temp
            
            # Plot
            plt.plot(qT, mean_xsec.flatten(), 'r-', linewidth=2, label='Ensemble Mean')
            plt.plot(qT, closest_xsec.flatten(), 'b--', linewidth=2, label='Selected Model')
            plt.fill_between(qT.flatten(), lower_xsec.flatten(), upper_xsec.flatten(), 
                             color='red', alpha=0.2, label='95% CI')
            
            # Plot experimental data
            if 'dsigma' in bin_data.columns and 'error' in bin_data.columns:
                plt.errorbar(bin_data['qT'], bin_data['dsigma'], yerr=bin_data['error'], 
                            fmt='ko', label='Experimental Data')
            
            plt.title(bin_name)
            plt.xlabel('qT (GeV)')
            plt.ylabel('Cross Section')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'QM_bins_comparison_{fig_idx+1}.png')
        plt.close()

# Find the model closest to the mean predictions using actual data
averaged_model = find_closest_model_to_mean(models_list, data)

# Create model comparison plots by QM bins
try:
    plot_by_QM_bins(averaged_model)
except Exception as e:
    print(f"Warning: Could not create QM bin plots: {e}")

# Compare with experimental data
try:
    mse, chi_squared = compare_model_with_data(averaged_model, data)

    # Create a summary report file
    with open('model_selection_report.txt', 'w') as f:
        f.write("Model Selection Summary Report\n")
        f.write("============================\n\n")
        f.write(f"Total models evaluated: {len(models_list)}\n")
        f.write(f"Selected model: {model_files[models_list.index(averaged_model)]}\n")
        f.write(f"MSE with experimental data: {mse:.8f}\n")
        f.write(f"Chi-squared per degree of freedom: {chi_squared:.8f}\n\n")
        f.write("Model selection used a 95% confidence interval approach with scipy.stats.\n")
        f.write("The model with predictions closest to the ensemble mean was selected as the representative model.\n")
        f.write("Visualization plots have been saved to disk for various energy ranges and QM bins.\n")
except Exception as e:
    print(f"Warning: Error comparing with experimental data: {e}")

# Save model
avg_model_path = 'averaged_model.h5'
averaged_model.save(avg_model_path)
print(f"Saved closest-to-mean model as {avg_model_path}")



# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Load Data
# E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
# E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
# E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
# # E605 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E605.csv")
# # E772 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E772.csv")


# # data = pd.concat([E288_200,E288_300,E288_400,E605,E772], ignore_index=True)
# # data = pd.concat([E288_400], ignore_index=True)
# data = pd.concat([E288_200,E288_300,E288_400], ignore_index=True)




# models_folder = 'Models'
# if not os.path.exists(models_folder):
#     models_folder = '../../../Step_by_step_tuning_to_get_sqT/Test_68/Models'

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

#     else:
#         return tf.reduce_mean(tf.square(y_pred - y_true))


# class CustomWeightedLoss(tf.keras.losses.Loss):
#     def __init__(self, name="custom_weighted_loss"):
#         super().__init__(name=name)
    
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         return custom_weighted_loss(y_true, y_pred, sample_weight)


# model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
# models_list = [tf.keras.models.load_model(
#     os.path.join(models_folder, f), 
#     custom_objects={
#         'custom_weighted_loss': custom_weighted_loss,
#         'CustomWeightedLoss': CustomWeightedLoss,
#         'train_weighted_loss': custom_weighted_loss}) for f in model_files]



# ### Cross-Section #####

# alpha = 1/137
# hc_factor = 3.89 * 10**8
# factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)

# def QM_int(QM):
#     return (-1)/(2*QM**2)

# def compute_QM_integrals(QM_array):
#     QM_array = np.atleast_1d(QM_array) 
#     QM_integrated = QM_int(QM_array + 0.5) - QM_int(QM_array - 0.5)
#     return QM_integrated[0] if QM_integrated.size == 1 else QM_integrated

# # Compute A Predictions
# def compute_A(x1, x2, qT, QM, pdfs_x1x2, pdfs_x2x1):
#     # Get Predictions from All Models
#     fDNN_contributions = np.array([model.predict([qT,QM,x1,x2, pdfs_x1x2, pdfs_x2x1], verbose=0).flatten() for model in models_list])
#     fDNN_mean = np.mean(fDNN_contributions, axis=0)
#     fDNN_std = np.std(fDNN_contributions, axis=0)

#     factor_temp = factor

#     QM_integral_temp = compute_QM_integrals(QM)

#     A_pred = fDNN_mean * factor_temp * hc_factor * QM_integral_temp
#     A_std = fDNN_std * factor_temp * hc_factor * QM_integral_temp

#     return A_pred, A_std




# # Steps (without considering weights)
# # Scan through all the trained models
# # Find the mean predictions, Ensure the confidence interval to 99%
# # Again scan through the models one by one and find the model which predicts closest to the mean predictions
# # use scipy.stat



# averaged_model = find_closest_model_to_mean(models_list)


# # Save model
# avg_model_path = 'averaged_model.h5'
# averaged_model.save(avg_model_path)
