import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.offline import plot

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
        'train_weighted_loss': custom_weighted_loss
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
    'SqT': SqT_mean,
    'SqT_err': SqT_std
}

results_csv_df = pd.DataFrame(temp_df)
results_csv_df.to_csv('results.csv')

# Create matplotlib plots
plt.figure(1, figsize=(10, 6))
plt.errorbar(x1_array, SqT_mean, yerr=SqT_std, fmt='o', alpha=0.5)
plt.title('SqT Predictions with Uncertainty')
plt.xlabel('$x_1$')
plt.ylabel('SqT')
plt.tight_layout()
plt.savefig('SqT_vs_x1_predictions.png')

plt.figure(2, figsize=(10, 6))
plt.errorbar(qT_array, SqT_mean, yerr=SqT_std, fmt='o', alpha=0.5)
plt.title('SqT Predictions with Uncertainty')
plt.xlabel('$q_T$')
plt.ylabel('SqT')
plt.tight_layout()
plt.savefig('SqT_vs_qT_predictions.png')

# Create 3D scatter plot with Plotly
# Prepare mesh grid for 3D visualization
qT_mesh, x1_mesh = np.meshgrid(qT_array, x1_array)
qT_flat = qT_mesh.flatten()
x1_flat = x1_mesh.flatten()

# Create a 2D mesh of points
mesh_points = np.column_stack([qT_flat, x1_flat, np.repeat(x2_value, len(qT_flat))])

# Predict SqT values for each mesh point
mesh_SqT_preds = []
for model in models_list:
    try:
        SqT_model = model.get_layer('SqT')
        mesh_pred = SqT_model.predict(mesh_points, verbose=0)
        mesh_SqT_preds.append(mesh_pred)
    except Exception as e:
        print(f"Error during mesh prediction: {e}")

# Calculate mean and std for mesh predictions
if len(mesh_SqT_preds) > 0:
    mesh_SqT_mean = np.mean(np.array(mesh_SqT_preds), axis=0).flatten()
    mesh_SqT_std = np.std(np.array(mesh_SqT_preds), axis=0).flatten()
else:
    mesh_SqT_mean = np.zeros(len(mesh_points))
    mesh_SqT_std = np.zeros(len(mesh_points))

# Create 3D scatter plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=qT_flat,
        y=x1_flat,
        z=mesh_SqT_mean,
        mode='markers',
        marker=dict(
            size=5,
            color=mesh_SqT_mean,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="SqT Value"),
            showscale=True
        ),
        error_z=dict(
            type='data',
            array=mesh_SqT_std,
            visible=True
        )
    )
])

# Update layout for better visualization
fig.update_layout(
    title='3D Visualization of SqT Predictions',
    scene=dict(
        xaxis_title='qT',
        yaxis_title='x1',
        zaxis_title='SqT',
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=0.8)
        )
    ),
    width=900,
    height=700,
    margin=dict(l=65, r=50, b=65, t=90),
)

# Save the figure as an HTML file
plot(fig, filename='SqT_3D_visualization.html', auto_open=False)
print("3D visualization saved as 'SqT_3D_visualization.html'")
