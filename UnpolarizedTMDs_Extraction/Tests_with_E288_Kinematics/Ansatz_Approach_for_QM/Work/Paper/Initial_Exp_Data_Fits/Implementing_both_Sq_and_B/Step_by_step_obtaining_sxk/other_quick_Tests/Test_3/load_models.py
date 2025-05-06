import tensorflow as tf
import os
import numpy as np

# Define the custom loss function from the original code
def mse_loss(y_true, y_pred):
    """Mean squared error loss function."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Recreate the necessary function to create neural network models
def create_nn_model(name):
    # Input shape is 2: either (x1, k) or (x2, kB)
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    
    # First layer
    x = tf.keras.layers.Dense(32, activation='relu6', 
                             kernel_initializer=initializer, 
                             kernel_regularizer=tf.keras.regularizers.L1(10**(-12)))(inp)
    
    # Hidden layers
    for _ in range(2):  # 3-1 hidden layers as in original code
        x = tf.keras.layers.Dense(32, activation='relu6', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.L1(10**(-12)))(x)
    
    # Output layer
    nnout = tf.keras.layers.Dense(1, activation='relu6', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.L1(10**(-12)))(x)
    
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
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    
    custom_objects = {
        'mse_loss': mse_loss,
        'TMDIntegrationLayer': TMDIntegrationLayer
    }
    
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


models = load_models('DNNmodels')
print(models)

