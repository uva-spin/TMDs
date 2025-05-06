import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Constants
eu2 = (2/3)**2
ed2 = (-1/3)**2
es2 = (-1/3)**2
alpha = 1/137
hc_factor = 3.89 * 10**8
factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)

NUM_QT_VALUES = 20
NUM_xF_VALUES = 20

# Load Data
E288_200 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv")
E288_300 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv")
E288_400 = pd.read_csv("/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv")
data = pd.concat([E288_200, E288_300, E288_400], ignore_index=True)

# Define x2_min before reference
x2_min = data['x2'].min()
x2_unique = np.sort(data['x2'].unique())
x2_value = x2_unique[0] if len(x2_unique) > 0 else x2_min

qT_array = np.linspace(data['qT'].min(), data['qT'].max(), NUM_QT_VALUES)
xF_array = np.linspace(data['xF'].min(), data['xF'].max(), NUM_xF_VALUES)
x1_array = xF_array + x2_value
x2_array = np.array(np.linspace(x2_value,x2_value,NUM_xF_VALUES))
SqT_input = np.column_stack([qT_array, x1_array, x2_array])

models_folder = 'Models'
if not os.path.exists(models_folder):
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
        'train_weighted_loss': custom_weighted_loss  # In case the model used this name
    }
) for f in model_files]

print(f"Loaded {len(models_list)} models from '{models_folder}'.")


all_SqT_model_preds = []


for i in range(0, 20):    
    if i == 0:
        SqT_model_preds = []
        
        for model in models_list:
            try:
                SqT_model = model.get_layer('SqT')
                SqT_pred = SqT_model.predict(SqT_input, verbose=0)
                SqT_model_preds.append(SqT_pred)
            except Exception as e:
                print(f"Error during prediction: {e}")
        
        all_SqT_model_preds = SqT_model_preds
    
    SqT_all_preds = np.array(all_SqT_model_preds)
    
    if len(SqT_all_preds) > 0:
        SqT_mean = np.mean(SqT_all_preds, axis=0).flatten()
        SqT_std = np.std(SqT_all_preds, axis=0).flatten()
    else:
        print("Warning: No valid predictions available.")
        SqT_mean = np.zeros(len(SqT_input))
        SqT_std = np.zeros(len(SqT_input))
    
temp_df = {
    'qT': qT_array,
    'x1': x1_array,
    'x2': x2_array,
    'SqT' : SqT_mean,
    'SqT_err' : SqT_std
}

results_csv_df = pd.DataFrame(temp_df)
results_csv_df.to_csv('results.csv')

plt.figure(1,figsize=(10, 6))
plt.errorbar(x1_array, SqT_mean, yerr=SqT_std, fmt='o', alpha=0.5)
plt.title('SqT Predictions with Uncertainty')
plt.xlabel('$x_1')
plt.ylabel('SqT')
plt.tight_layout()
plt.savefig('SqT_vs_x1_predictions.png')


plt.figure(2,figsize=(10, 6))
plt.errorbar(qT_array, SqT_mean, yerr=SqT_std, fmt='o', alpha=0.5)
plt.title('SqT Predictions with Uncertainty')
plt.xlabel('$q_T$')
plt.ylabel('SqT')
plt.tight_layout()
plt.savefig('SqT_vs_qT_predictions.png')