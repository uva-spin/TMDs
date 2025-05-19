import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import lhapdf
import datetime
from pathlib import Path

def create_folders(folder_name):
    folder_path = Path(folder_name)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_name}' created successfully!")

filtered_models_folder = Path('Models')
create_folders(filtered_models_folder)

eu2 = (2/3)**2
ed2 = (-1/3)**2
es2 = (-1/3)**2
alpha = 1/137
hc_factor = 3.89 * 10**8
factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)


# Load LHAPDF Set
NNPDF4_nlo = lhapdf.mkPDF('NNPDF40_nlo_as_01180')


def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Load Data
E288_200 = pd.read_csv("../Data_Updated/E288_200.csv")
E288_300 = pd.read_csv("../Data_Updated/E288_300.csv")
E288_400 = pd.read_csv("../Data_Updated/E288_400.csv")
E605 = pd.read_csv("../Data_Updated/E605.csv")

data = pd.concat([E288_200, E288_300, E288_400, E605], ignore_index=True)

source_models_folder = '../Step_01/Models'

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

source_model_files = [f for f in os.listdir(source_models_folder) if f.endswith('.h5')]
source_models_list = [tf.keras.models.load_model(
    os.path.join(source_models_folder, f), 
    custom_objects={
        'custom_weighted_loss': custom_weighted_loss,
        'CustomWeightedLoss': CustomWeightedLoss,
        'train_weighted_loss': custom_weighted_loss}) for f in source_model_files]



def prepare_data_from_df(df):
    
    qT = df['qT'].values.reshape(-1, 1)
    QM = df['QM'].values.reshape(-1, 1)
    x1 = df['x1'].values.reshape(-1, 1)
    x2 = df['x2'].values.reshape(-1, 1)

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
    
    return [qT, QM, x1, x2, pdfs_x1x2, pdfs_x2x1]


def find_models_within_95ci(models_list, data_df):
    """
    Selects and saves models whose predictions fall within the 95% confidence interval
    of the ensemble mean prediction.
    """
    data_df = data if data_df is None else data_df  # Use global data if not passed

    # Prepare inputs
    input_data = prepare_data_from_df(data_df)

    # Get predictions from all models
    all_predictions = np.array([model.predict(input_data, verbose=0) for model in models_list])  # shape: (n_models, n_samples, 1)


    # n_models = len(models_list)
    # mean_predictions = np.mean(all_predictions, axis=0)
    # std_predictions = np.std(all_predictions, axis=0, ddof=1)  # sample std deviation
    # t_critical = stats.t.ppf(0.975, df=n_models-1)  # 95% CI

    # margin_of_error = t_critical * (std_predictions / np.sqrt(n_models))
    # lower_bound = mean_predictions - margin_of_error
    # upper_bound = mean_predictions + margin_of_error

    # Compute mean and standard deviation across models
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)

    # Compute 2-sigma bounds (assuming normality)
    lower_bound = mean_predictions -  2*std_predictions
    upper_bound = mean_predictions + 2*std_predictions


    selected_model_indices = []
    for i, pred in enumerate(all_predictions):
        within_bounds = np.all((pred >= lower_bound) & (pred <= upper_bound))
        if within_bounds:
            selected_model_indices.append(i)

    print(f"{len(selected_model_indices)} models found within 95% CI out of {len(models_list)}")


    # Save selected models
    for idx in selected_model_indices:
        model_path = os.path.join(filtered_models_folder, f"model_{idx:03d}.h5")
        models_list[idx].save(model_path)

    return selected_model_indices


# Usage
starttime_scan_models = datetime.datetime.now().replace(microsecond=0)

# Run the selection and saving
selected_indices = find_models_within_95ci(source_models_list, data_df=data)

finishtime_scan_models = datetime.datetime.now().replace(microsecond=0)
totalduration = finishtime_scan_models - starttime_scan_models
print(f"Total duration to find and save models within 95% CI --> {totalduration}")


