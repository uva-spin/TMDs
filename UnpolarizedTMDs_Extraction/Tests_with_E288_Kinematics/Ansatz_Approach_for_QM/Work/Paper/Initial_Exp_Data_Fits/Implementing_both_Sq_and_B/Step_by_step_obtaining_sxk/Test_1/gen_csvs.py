import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from model import *


def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

CSV_folder = 'csvs'
create_folders(CSV_folder)


# Define the custom loss function from the original code
def mse_loss(y_true, y_pred):
    """Mean squared error loss function."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Recreate the custom layer needed for loading models
class TMDIntegrationLayer(tf.keras.layers.Layer):
    def __init__(self, k_bins=10, phi_bins=10, k_lower=0.0, k_upper=10.0, **kwargs):
        super(TMDIntegrationLayer, self).__init__(**kwargs)
        self.k_bins = k_bins
        self.phi_bins = phi_bins
        self.k_lower = float(k_lower)
        self.k_upper = float(k_upper)
        
        # Pre-compute k and phi values
        self.k_values = np.linspace(self.k_lower, self.k_upper, self.k_bins, dtype=np.float32)
        self.phi_values = np.linspace(0.0, np.pi, self.phi_bins, dtype=np.float32)
        
        # Calculate step sizes
        self.dk = float(self.k_upper - self.k_lower) / float(self.k_bins - 1)
        self.dphi = float(np.pi) / float(self.phi_bins - 1)
        
        # Create the neural networks
        self.modn1 = create_nn_model('n1')
        self.modn2 = create_nn_model('n2')
    
    
    def call(self, inputs):
        qT, x1, x2 = inputs
        
        # Initialize the output tensor with zeros
        batch_size = tf.shape(qT)[0]
        result = tf.zeros((batch_size, 1), dtype=tf.float32)
        
        # Loop through k and phi values
        for k_idx in range(self.k_bins):
            k_val = self.k_values[k_idx]
            k_factor = k_val  # For Jacobian if needed
            
            for phi_idx in range(self.phi_bins):
                phi_val = self.phi_values[phi_idx]
                
                # Calculate kB
                kb_tensor = tf.sqrt(qT**2 + k_val**2 - 2*qT*k_val*tf.cos(phi_val))
                
                # Create input tensors for the neural networks
                n1_input = tf.concat([x1, tf.ones_like(x1) * k_val], axis=1)
                n2_input = tf.concat([x2, kb_tensor], axis=1)
                
                # Get neural network outputs
                nn1_out = self.modn1(n1_input)
                nn2_out = self.modn2(n2_input)
                
                # Multiply, apply integration weights, and add to result
                contribution = nn1_out * nn2_out * k_factor * self.dk * self.dphi
                result += contribution
        
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'k_bins': self.k_bins,
            'phi_bins': self.phi_bins,
            'k_lower': self.k_lower,
            'k_upper': self.k_upper
        })
        return config

# Function to load models with custom objects
def load_models(models_folder='DNNmodels'):
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    
    # Define custom objects for model loading
    custom_objects = {
        'mse_loss': mse_loss,
        'TMDIntegrationLayer': TMDIntegrationLayer
    }
    
    # Load all models
    models_list = []
    for f in model_files:
        model_path = os.path.join(models_folder, f)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        models_list.append(model)
    
    print(f"Loaded {len(models_list)} models from '{models_folder}'.")
    return models_list

def main():
    # Load the original data
    print("Loading original data...")
    df = pd.read_csv('results.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    # Load all trained models
    print("Loading trained models...")
    models = load_models('DNNmodels')
    
    if not models:
        print("No models loaded. Please check the models folder.")
        return
    
    # Prepare input data for prediction
    qT_array = df['qT'].values.reshape(-1, 1)
    x1_array = df['x1'].values.reshape(-1, 1)
    x2_array = df['x2'].values.reshape(-1, 1)

    # Get unique x1 values from the dataset (for better visualization)
    unique_x1 = np.sort(np.unique(df['x1'].values))
    # Create k array with the same dimension as unique_x1
    k_values = np.linspace(0, 10, len(unique_x1))
    # Select a few k values to plot
    selected_k_values = [0.5, 2.0, 5.0, 8.0]
    # Finer k array for smoother curves
    fine_k_values = np.linspace(0, 10, 100)
    
    # Original SqT values and errors
    SqT_orig = df['SqT'].values
    SqT_err_orig = df['SqT_err'].values
    
    # Make predictions with all models
    print("Making predictions with all models...")
    SqT_predictions = []
    n1_pred_for_x1 = []
    
    for i, model in enumerate(models):
        print(f"Predicting with model {i+1}/{len(models)}...")
        pred = model.predict([qT_array, x1_array, x2_array], verbose=0)
        SqT_predictions.append(pred.flatten())
        
        # Extract n1 model from the loaded model's TMDIntegrationLayer
        tmd_layer = None
        for layer in model.layers:
            if isinstance(layer, TMDIntegrationLayer):
                tmd_layer = layer
                break
        
        if tmd_layer:
            n1_model = tmd_layer.modn1
            
            # We want to generate an array of n1 values for each x1
            for x1_val in x1_array:
                n1_outputs_each_k = []
                for k_val in fine_k_values:
                    # Create input for n1 model: [x1, k]
                    n1_input = np.array([[x1_val[0], k_val]])  # Extract the scalar value from x1_val
                    n1_output = n1_model.predict(n1_input, verbose=0)[0][0]
                    n1_outputs_each_k.append(n1_output)
                n1_pred_for_x1.append({
                    'model': i,
                    'x1': x1_val[0],
                    'k_values': fine_k_values,
                    'n1_values': np.array(n1_outputs_each_k)
                })
    
    # Convert to numpy array for easier calculations
    SqT_predictions = np.array(SqT_predictions)
    
    # Calculate mean and standard deviation across all models for SqT
    SqT_mean = np.mean(SqT_predictions, axis=0)
    SqT_std = np.std(SqT_predictions, axis=0)
    
    # Save the prediction results to CSV for future reference
    SqT_df = pd.DataFrame({
        'x1': df['x1'],
        'x2': df['x2'],
        'qT': df['qT'],
        'SqT_orig': SqT_orig,
        'SqT_err_orig': SqT_err_orig,
        'SqT_pred_mean': SqT_mean,
        'SqT_pred_std': SqT_std
    })
    
    SqT_df.to_csv(str(CSV_folder)+'/SqT_results.csv', index=False)
    print("Results for SqT are saved")
    
    # Create and save n1 results to CSV
    print("Processing n1 predictions...")
    # Create a list to hold all the rows for our DataFrame
    n1_data_rows = []
    
    # Flatten the nested structure into rows for the DataFrame
    for item in n1_pred_for_x1:
        model_idx = item['model']
        x1_val = item['x1']
        for k_idx, k_val in enumerate(item['k_values']):
            n1_val = item['n1_values'][k_idx]
            n1_data_rows.append({
                'model': model_idx,
                'x1': x1_val,
                'k': k_val,
                'n1': n1_val
            })
    
    # Create the DataFrame and save to CSV
    n1_df = pd.DataFrame(n1_data_rows)
    # Save to CSV
    n1_df.to_csv(str(CSV_folder)+'/n1_results.csv', index=False)
    print("Results for n1 are saved")
    
    # # Optional: Calculate and print some statistics
    # mean_rel_error = np.mean(np.abs(SqT_mean - SqT_orig) / SqT_orig) * 100
    # print(f"Mean relative error: {mean_rel_error:.2f}%")
    # # Calculate chi-squared
    # chi2 = np.sum(((SqT_mean - SqT_orig) / SqT_err_orig) ** 2)
    # reduced_chi2 = chi2 / len(SqT_orig)
    # print(f"Chi-squared: {chi2:.2f}")
    # print(f"Reduced chi-squared: {reduced_chi2:.2f}")


if __name__ == "__main__":
    main()