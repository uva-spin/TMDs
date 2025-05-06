import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

#Constants
eu2 = (2/3)**2
ed2 = (-1/3)**2
es2 = (-1/3)**2
alpha = 1/137
hc_factor = 3.89 * 10**8
factor = ((4*np.pi*alpha)**2)/(9*2*np.pi)

# Configuration variables
NUM_QT_VALUES = 20  # Number of qT values
BATCH_SIZE = 1024   # Batch size for GPU processing

# Here we assume a single value of QM
QM = 5

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# Create Results Folder
results_folder = 'Results_csvs'
create_folders(results_folder)


def pdf(pdfset, flavor, x, QQ):
    return pdfset.xfxQ(flavor, x, QQ)

# Load Data
# Update these paths according to your system
data_paths = {
    'E288_200': "/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_200.csv",
    'E288_300': "/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_300.csv",
    'E288_400': "/home/ishara/Documents/TMDs/Ansatz_Approach_for_QM/Work/Data_Updated/E288_400.csv"
}

# Try to load data with error handling
data_frames = []
for key, path in data_paths.items():
    try:
        df = pd.read_csv(path)
        data_frames.append(df)
    except FileNotFoundError:
        try:
            # Try with a relative path
            alternative_path = os.path.join(".", path)
            df = pd.read_csv(alternative_path)
            data_frames.append(df)
        except FileNotFoundError:
            print(f"Warning: Could not find data file {path}. Using empty dataframe.")
            # Create empty dataframe with required columns
            df = pd.DataFrame(columns=['qT', 'x1', 'x2', 'xF'])
            data_frames.append(df)

data = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame(columns=['qT', 'x1', 'x2', 'xF'])

# Use default ranges if data is empty
if len(data) == 0:
    print("No data loaded. Using default ranges.")
    qT_min, qT_max = 0.1, 5.0
    x1_min, x1_max = 0.1, 0.9
    x2_min, x2_max = 0.1, 0.9
    xF_min, xF_max = -0.1, 0.5
else:
    qT_min, qT_max = data['qT'].min(), data['qT'].max()
    x1_min, x1_max = data['x1'].min(), data['x1'].max()
    x2_min, x2_max = data['x2'].min(), data['x2'].max()
    xF_min, xF_max = data['xF'].min(), data['xF'].max()

# Adjust the models folder path - update this to your system path
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
    # For model loading (when w not available), fall back to MSE
    else:
        return tf.reduce_mean(tf.square(y_pred - y_true))

# Create a wrapper class to make the loss function serializable
class CustomWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_weighted_loss"):
        super().__init__(name=name)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        return custom_weighted_loss(y_true, y_pred)

# Check if models folder exists
if not os.path.exists(models_folder):
    print(f"Error: Models folder '{models_folder}' does not exist.")
    # Create a dummy model for testing if no models are available
    print("Creating a dummy model for testing purposes...")
    input_layer = tf.keras.layers.Input(shape=(3,))
    hidden = tf.keras.layers.Dense(10, activation='relu')(input_layer)
    output = tf.keras.layers.Dense(1)(hidden)
    dummy_model = tf.keras.models.Model(inputs=input_layer, outputs=output, name="dummy_model")
    # Add a layer with name 'SqT' for testing
    SqT_input = tf.keras.layers.Input(shape=(3,))
    hidden = tf.keras.layers.Dense(10, activation='relu')(SqT_input)
    SqT_output = tf.keras.layers.Dense(1)(hidden)
    SqT_model = tf.keras.models.Model(inputs=SqT_input, outputs=SqT_output, name="SqT")
    models_list = [dummy_model]
    # Add SqT as a layer to the dummy model (workaround for testing)
    dummy_model.layers.append(SqT_model)
    dummy_model._layer_by_name = {layer.name: layer for layer in dummy_model.layers}
else:
    # Load All Trained Models with proper custom objects
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    if not model_files:
        print(f"No model files found in '{models_folder}'.")
        exit(1)
        
    models_list = []
    for f in model_files:
        try:
            model = tf.keras.models.load_model(
                os.path.join(models_folder, f), 
                custom_objects={
                    'custom_weighted_loss': custom_weighted_loss,
                    'CustomWeightedLoss': CustomWeightedLoss,
                    'train_weighted_loss': custom_weighted_loss
                }
            )
            models_list.append(model)
        except Exception as e:
            print(f"Error loading model {f}: {e}")

print(f"Loaded {len(models_list)} models.")

######### Generate grid for model outputs #################

# 1. Take the first unique x2 value
x2_unique = np.sort(data['x2'].unique())
x2_value = x2_unique[0] if len(x2_unique) > 0 else x2_min
print(f"Using fixed x2 value: {x2_value}")

# 2. Use a specific xF value from the middle of the dataset
xF_unique = np.sort(data['xF'].unique())
if len(xF_unique) > 0:
    # Take a value from the middle of the range for a more representative slice
    middle_index = len(xF_unique) // 2
    xF_value = xF_unique[middle_index]
    print(f"Using fixed xF value: {xF_value} (from position {middle_index} in sorted unique values)")
else:
    # Fallback if no data is available
    xF_value = xF_min
    print(f"No xF values found in data. Using default: {xF_value}")

# 3. Generate qT values
qT_values = np.linspace(qT_min, qT_max, NUM_QT_VALUES)

# 4. Calculate x1 values from xF and x2: xF = x1 - x2 => x1 = xF + x2
x1_value = xF_value + x2_value
print(f"Calculated x1 value: {x1_value}")

# 5. Create grid points
grid_points = []
for qT in qT_values:
    grid_points.append((qT, x1_value, x2_value, xF_value))

# Convert to numpy arrays for processing
qT_array = np.array([point[0] for point in grid_points], dtype=np.float32)
x1_array = np.array([point[1] for point in grid_points], dtype=np.float32)
x2_array = np.array([point[2] for point in grid_points], dtype=np.float32)
xF_array = np.array([point[3] for point in grid_points], dtype=np.float32)
QM_array = np.full(len(grid_points), QM, dtype=np.float32)

# Initialize result arrays
SqT_means = np.zeros(len(grid_points), dtype=np.float32)
SqT_stds = np.zeros(len(grid_points), dtype=np.float32)

# Process predictions
print("Processing predictions...")
total_points = len(grid_points)
for i in range(0, total_points, BATCH_SIZE):
    end_idx = min(i + BATCH_SIZE, total_points)
    batch_size = end_idx - i
    
    # Get batch data
    qT_batch = qT_array[i:end_idx].reshape(batch_size, 1)
    x1_batch = x1_array[i:end_idx].reshape(batch_size, 1)
    x2_batch = x2_array[i:end_idx].reshape(batch_size, 1)
    
    # Build SqT input tensor
    SqT_input = np.hstack([qT_batch, x1_batch, x2_batch])
    
    # Lists to store predictions from each model
    SqT_model_preds = []
    
    for model in models_list:
        try:
            # Get SqT sub-model
            SqT_model = model.get_layer('SqT')
            
            # Predict for SqT model
            SqT_pred_batch = SqT_model.predict(SqT_input, verbose=0, batch_size=batch_size)
            SqT_model_preds.append(SqT_pred_batch)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Add zeros as fallback
            SqT_model_preds.append(np.zeros((batch_size, 1)))
    
    # Convert to arrays for calculation
    SqT_all_preds = np.array(SqT_model_preds)  # Shape: [num_models, batch_size, 1]
    
    # Calculate mean and std across models (axis 0)
    SqT_mean_batch = np.mean(SqT_all_preds, axis=0).flatten()
    SqT_std_batch = np.std(SqT_all_preds, axis=0).flatten()
    
    # Store the results
    SqT_means[i:end_idx] = SqT_mean_batch
    SqT_stds[i:end_idx] = SqT_std_batch

# Create DataFrame and save to CSV
grid_data = {
    'qT': qT_array,
    'x1': x1_array,
    'x2': x2_array,
    'xF': xF_array,
    'QM': QM_array,
    'SqT_mean': SqT_means,
    'SqT_std': SqT_stds
}

grid_df = pd.DataFrame(grid_data)
csv_filename = f'{results_folder}/fixed_x2_xF_results.csv'
grid_df.to_csv(csv_filename, index=False)
print(f"Grid generated and saved to: {csv_filename}")
print(f"Grid shape: {grid_df.shape}")

# Load the CSV file for plotting
plot_df = pd.read_csv(csv_filename)

# Create plots to visualize the data
# Let's create a comprehensive visualization of the grid

# Create a directory for plots
plots_folder = f'{results_folder}/plots'
create_folders(plots_folder)

# 1. Create a heatmap showing SqT as a function of qT and xF
plt.figure(figsize=(12, 10))

# Reshape data for heatmap - we need to create a 2D grid
qT_unique = np.sort(plot_df['qT'].unique())
xF_unique = np.sort(plot_df['xF'].unique())

# Create a 2D array to hold SqT values
SqT_grid = np.zeros((len(xF_unique), len(qT_unique)))

# Fill the grid with values
for i, xF in enumerate(xF_unique):
    for j, qT in enumerate(qT_unique):
        mask = (plot_df['xF'] == xF) & (plot_df['qT'] == qT)
        if any(mask):
            SqT_grid[i, j] = plot_df.loc[mask, 'SqT_mean'].values[0]

# Create meshgrid for plotting
QT, XF = np.meshgrid(qT_unique, xF_unique)

# Plot the heatmap
plt.pcolormesh(QT, XF, SqT_grid, cmap='viridis', shading='auto')
plt.colorbar(label='SqT mean')
plt.xlabel('qT')
plt.ylabel('xF')
plt.title(f'SqT Heatmap (x2={x2_value:.4f})')
plt.tight_layout()
plt.savefig(f'{results_folder}/SqT_heatmap.png', dpi=300)
plt.close()

print(f"Heatmap saved to: {results_folder}/SqT_heatmap.png")

# 2. Create scatter plots for several xF values
selected_xFs = xF_unique[::max(1, len(xF_unique)//5)]  # Take ~5 equally spaced xF values
plt.figure(figsize=(12, 8))

for xF in selected_xFs:
    subset = plot_df[plot_df['xF'] == xF]
    x1_value = subset['x1'].iloc[0]
    plt.scatter(subset['qT'], subset['SqT_mean'], label=f'xF={xF:.4f}, x1={x1_value:.4f}', 
               s=30, alpha=0.7)

plt.xlabel('qT')
plt.ylabel('SqT mean')
plt.title(f'SqT vs qT for Selected xF Values (fixed x2={x2_value:.4f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'{results_folder}/SqT_vs_qT_selected_xF.png', dpi=300)
plt.close()

print(f"Selected xF plot saved to: {results_folder}/SqT_vs_qT_selected_xF.png")

# 3. Create a 3D surface plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(QT, XF, SqT_grid, cmap='viridis', edgecolor='none', alpha=0.8)

# Add a colorbar
cbar = fig.colorbar(surf, ax=ax, pad=0.1)
cbar.set_label('SqT mean')

# Add labels and title
ax.set_xlabel('qT')
ax.set_ylabel('xF')
ax.set_zlabel('SqT mean')
ax.set_title(f'3D Surface of SqT vs qT and xF (fixed x2={x2_value:.4f})')

# Adjust the viewing angle
ax.view_init(elev=30, azim=45)

# Save the 3D figure
plt.tight_layout()
plt.savefig(f'{results_folder}/SqT_3D_surface.png', dpi=300)
plt.close()

print(f"3D surface plot saved to: {results_folder}/SqT_3D_surface.png")

# 4. Create scatter plot showing x1 vs SqT for different qT values
selected_qTs = qT_unique[::max(1, len(qT_unique)//5)]  # Take ~5 equally spaced qT values
plt.figure(figsize=(12, 8))

for qT in selected_qTs:
    subset = plot_df[plot_df['qT'] == qT]
    plt.scatter(subset['x1'], subset['SqT_mean'], label=f'qT={qT:.4f}', s=30, alpha=0.7)

plt.xlabel('x1')
plt.ylabel('SqT mean')
plt.title(f'SqT vs x1 for Selected qT Values (fixed x2={x2_value:.4f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'{results_folder}/SqT_vs_x1_selected_qT.png', dpi=300)
plt.close()

print(f"Selected qT plot saved to: {results_folder}/SqT_vs_x1_selected_qT.png")