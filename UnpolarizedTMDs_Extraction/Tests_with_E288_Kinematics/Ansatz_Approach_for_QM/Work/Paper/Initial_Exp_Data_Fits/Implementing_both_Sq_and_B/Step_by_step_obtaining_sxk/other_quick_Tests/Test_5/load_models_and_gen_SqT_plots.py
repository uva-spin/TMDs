import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the custom loss function from the original code
def mse_loss(y_true, y_pred):
    """Mean squared error loss function."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Recreate the necessary function to create neural network models
Hidden_Layers = 3  # Reduced for simplicity
Nodes_per_HL = 100  # Reduced for simplicity
Learning_Rate = 0.00001
L1_reg = 10**(-12)

def create_nn_model(name):
    # Input shape is 2: either (x1, k) or (x2, kB)
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    
    # First layer
    x = tf.keras.layers.Dense(Nodes_per_HL, activation='relu6', 
                             kernel_initializer=initializer, 
                             kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    
    # Hidden layers
    for _ in range(Hidden_Layers-1):
        x = tf.keras.layers.Dense(Nodes_per_HL, activation='relu6', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    
    # Output layer
    nnout = tf.keras.layers.Dense(1, activation='relu6', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

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
    
    def build(self, input_shape):
        # Nothing to build specifically
        super(TMDIntegrationLayer, self).build(input_shape)
    
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
    """
    Load all trained models from the specified folder.
    
    Args:
        models_folder: Path to the folder containing saved models
        
    Returns:
        List of loaded models
    """
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
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            models_list.append(model)
            print(f"Successfully loaded model: {f}")
        except Exception as e:
            print(f"Failed to load model {f}: {str(e)}")
    
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
    
    # Original SqT values and errors
    SqT_orig = df['SqT'].values
    SqT_err_orig = df['SqT_err'].values
    
    # Make predictions with all models
    print("Making predictions with all models...")
    predictions = []
    for i, model in enumerate(models):
        print(f"Predicting with model {i+1}/{len(models)}...")
        pred = model.predict([qT_array, x1_array, x2_array], verbose=0)
        predictions.append(pred.flatten())
    
    # Convert to numpy array for easier calculations
    predictions = np.array(predictions)
    
    # Calculate mean and standard deviation across all models
    SqT_mean = np.mean(predictions, axis=0)
    SqT_std = np.std(predictions, axis=0)
    
    print("Generating comparison plots...")
    
    # Create directory for plots if it doesn't exist
    if not os.path.exists('comparison_plots'):
        os.makedirs('comparison_plots')
    
    # Plot 1: SqT vs x1
    plt.figure(1, figsize=(10, 6))
    
    # Plot original data with error bars
    plt.errorbar(df['x1'], SqT_orig, yerr=SqT_err_orig, fmt='o', color='blue', 
                 alpha=0.7, label='Original Data')
    
    # Plot model predictions with uncertainty
    plt.errorbar(df['x1'], SqT_mean, yerr=SqT_std, fmt='s', color='red', 
                 alpha=0.7, label='Model Predictions')
    
    plt.title('SqT Predictions vs Original Data (x1)')
    plt.xlabel('$x_1')
    plt.ylabel('SqT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/SqT_vs_x1_comparison.png')
    
    # Plot 2: SqT vs qT
    plt.figure(2, figsize=(10, 6))
    
    # Plot original data with error bars
    plt.errorbar(df['qT'], SqT_orig, yerr=SqT_err_orig, fmt='o', color='blue', 
                 alpha=0.7, label='Original Data')
    
    # Plot model predictions with uncertainty
    plt.errorbar(df['qT'], SqT_mean, yerr=SqT_std, fmt='s', color='red', 
                 alpha=0.7, label='Model Predictions')
    
    plt.title('SqT Predictions vs Original Data (qT)')
    plt.xlabel('$q_T')
    plt.ylabel('SqT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/SqT_vs_qT_comparison.png')
    
    # Generate n1 layer output plots
    print("Generating n1 layer output plots...")
    
    # Extract n1 model from the first loaded model's TMDIntegrationLayer
    tmd_layer = None
    for layer in models[0].layers:
        if isinstance(layer, TMDIntegrationLayer):
            tmd_layer = layer
            break
            
    if tmd_layer is None:
        print("Could not find TMDIntegrationLayer in the model.")
        return
        
    n1_model = tmd_layer.modn1
    
    # Get unique x1 values from the dataset (for better visualization)
    unique_x1 = np.sort(np.unique(df['x1'].values))
    
    # Create k array with the same dimension as unique_x1
    k_values = np.linspace(0, 10, len(unique_x1))
    
    # Plot 3: n1 vs x1 (for fixed k values)
    plt.figure(3, figsize=(10, 6))
    
    # Select a few k values to plot
    selected_k_values = [0.5, 2.0, 5.0, 8.0]
    
    for k_val in selected_k_values:
        n1_outputs = []
        for x1_val in unique_x1:
            # Create input for n1 model: [x1, k]
            n1_input = np.array([[x1_val, k_val]])
            n1_output = n1_model.predict(n1_input, verbose=0)[0][0]
            n1_outputs.append(n1_output)
        
        plt.plot(unique_x1, n1_outputs, 'o-', label=f'k = {k_val}')
    
    plt.title('n1 Layer Output vs x1 (for different k values)')
    plt.xlabel('$x_1')
    plt.ylabel('n1 Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/n1_vs_x1.png')
    
    # Plot 4: n1 vs k (for fixed x1 values)
    plt.figure(4, figsize=(10, 6))
    
    # Select a few x1 values to plot
    # Use quartiles of the x1 distribution for representative values
    x1_quartiles = np.percentile(df['x1'].values, [25, 50, 75])
    selected_x1_values = x1_quartiles
    
    # Finer k array for smoother curves
    fine_k_values = np.linspace(0, 10, 100)
    
    for x1_val in selected_x1_values:
        n1_outputs = []
        for k_val in fine_k_values:
            # Create input for n1 model: [x1, k]
            n1_input = np.array([[x1_val, k_val]])
            n1_output = n1_model.predict(n1_input, verbose=0)[0][0]
            n1_outputs.append(n1_output)
        
        plt.plot(fine_k_values, n1_outputs, '-', label=f'x1 = {x1_val:.3f}')
    
    plt.title('n1 Layer Output vs k (for different x1 values)')
    plt.xlabel('k')
    plt.ylabel('n1 Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_plots/n1_vs_k.png')
    
    # Save the prediction results to CSV for future reference
    results_df = pd.DataFrame({
        'x1': df['x1'],
        'x2': df['x2'],
        'qT': df['qT'],
        'SqT_orig': SqT_orig,
        'SqT_err_orig': SqT_err_orig,
        'SqT_pred_mean': SqT_mean,
        'SqT_pred_std': SqT_std
    })
    
    results_df.to_csv('comparison_plots/prediction_results.csv', index=False)
    print("Results saved to 'comparison_plots/prediction_results.csv'")
    
    # Optional: Calculate and print some statistics
    mean_rel_error = np.mean(np.abs(SqT_mean - SqT_orig) / SqT_orig) * 100
    print(f"Mean relative error: {mean_rel_error:.2f}%")
    
    # Calculate chi-squared
    chi2 = np.sum(((SqT_mean - SqT_orig) / SqT_err_orig) ** 2)
    reduced_chi2 = chi2 / len(SqT_orig)
    print(f"Chi-squared: {chi2:.2f}")
    print(f"Reduced chi-squared: {reduced_chi2:.2f}")
    
    print("Plots saved in 'comparison_plots' directory.")

if __name__ == "__main__":
    main()