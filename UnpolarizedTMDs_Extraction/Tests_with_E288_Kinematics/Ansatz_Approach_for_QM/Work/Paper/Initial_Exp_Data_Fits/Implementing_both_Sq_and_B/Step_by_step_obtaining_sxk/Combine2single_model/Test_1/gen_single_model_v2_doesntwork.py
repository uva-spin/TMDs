import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


### Model Architecture ###

# Define the Progressive DNN Model
def build_progressive_model(input_shape=(1,), depth=4, width=256,
                           L1_reg=1e-12, initializer_range=0.1,
                           use_residual=False, activations=None,
                           output_activation='linear', name=None):
    # Default activations if none provided
    if activations is None:
        activations = ['relu'] * depth
    elif isinstance(activations, str):
        activations = [activations] * depth
    elif len(activations) < depth:
        # Pad with the last activation if list isn't long enough
        activations.extend([activations[-1]] * (depth - len(activations)))
    
    initializer = tf.keras.initializers.RandomUniform(minval=-initializer_range,
                                                     maxval=initializer_range)
    regularizer = tf.keras.regularizers.L1(L1_reg)
    inp = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Dense(width, activation=activations[0],
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer)(inp)
    hidden_layers = [x]
    for i in range(1, depth):
        dense = tf.keras.layers.Dense(width, activation=activations[i],
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     activity_regularizer=regularizer,
                                     name=f"dense_{i}_{np.random.randint(10000)}")
        h = dense(hidden_layers[-1])
        if use_residual:
            x = tf.keras.layers.Add()([hidden_layers[-1], h])
        else:
            x = h
        hidden_layers.append(x)
    out = tf.keras.layers.Dense(1, activation=output_activation,
                               kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name=name)
    return model, hidden_layers

# Define Progressive DNNB
def DNNB(name):
    # Using different activations for each layer
    model, _ = build_progressive_model(
        input_shape=(1,),
        depth=3,
        width=100,
        use_residual=False,
        activations=['relu', 'tanh', 'relu'],  # Different activation for each layer
        output_activation='linear',
        name=name
    )
    return model

# Define Progressive DNNS
def DNNS(name):
    # Using different activations for each layer
    model, _ = build_progressive_model(
        input_shape=(3,),
        depth=3,
        width=100,
        use_residual=False,
        initializer_range=0.1,
        activations=['relu', 'tanh', 'relu'],  # Different activation for each layer
        output_activation='softplus',
        name=name
    )
    return model

def SB_model():
    qT = tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    pdfs_x1x2 = tf.keras.Input(shape=(1,), name='pdfs_x1x2')
    pdfs_x2x1 = tf.keras.Input(shape=(1,), name='pdfs_x2x1') 

    SModel = DNNS('SqT')
    BModel = DNNB('BQM')
    
    concatenatedx1x2 = tf.keras.layers.Concatenate()([qT, x1, x2])
    concatenatedx2x1 = tf.keras.layers.Concatenate()([qT, x2, x1])
    
    Sqx1x2 = SModel(concatenatedx1x2)
    Sqx2x1 = SModel(concatenatedx2x1)
    BQM = BModel(QM)

    pdfs_sqT_x1x2 = tf.keras.layers.Multiply()([pdfs_x1x2, Sqx1x2])
    pdfs_sqT_x2x1 = tf.keras.layers.Multiply()([pdfs_x2x1, Sqx2x1])
    
    # Calculate combined S contribution
    PDFs_S_combined = tf.keras.layers.Add()([pdfs_sqT_x1x2, pdfs_sqT_x2x1])
    
    # Multiply with pre-calculated PDFs
    SB_PDF = tf.keras.layers.Multiply()([PDFs_S_combined, BQM])
    
    return tf.keras.Model([qT, QM, x1, x2, pdfs_x1x2, pdfs_x2x1], SB_PDF)

########################################################################


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
        'train_weighted_loss': custom_weighted_loss}) for f in model_files]


################## APPROACH 1: WEIGHT AVERAGING (YOUR ORIGINAL METHOD) ##################

def create_averaged_model_simple(models_list):
    if not models_list:
        raise ValueError("No models provided")
    
    # Create a fresh model with the same architecture
    fresh_model = SB_model()
    
    # Get the weights from all models
    model_weights = [model.get_weights() for model in models_list]
    
    # Calculate the average weights across all models
    avg_weights = []
    for i in range(len(model_weights[0])):
        # Stack the i-th layer weights from all models
        layer_weights = [model_weight[i] for model_weight in model_weights]
        # Calculate the mean of these weights
        avg_layer_weights = np.mean(layer_weights, axis=0)
        avg_weights.append(avg_layer_weights)
    
    # Set the averaged weights to the fresh model
    fresh_model.set_weights(avg_weights)
    
    return fresh_model

# Create the model with averaged weights
averaged_model = create_averaged_model_simple(models_list)

# Compile the model with appropriate loss function
averaged_model.compile(
    optimizer='adam',
    loss=custom_weighted_loss,
    metrics=['mae']
)

# Save model
avg_model_path = 'averaged_model.h5'
averaged_model.save(avg_model_path)


################## APPROACH 2: ENSEMBLE APPROACH WITH STATISTICS ##################

class EnsembleModel:
    def __init__(self, models_list):
        self.models = models_list
        self.n_models = len(models_list)
    
    def predict(self, inputs, verbose=0):
        """
        Get predictions from all models in the ensemble.
        Returns mean and standard deviation for each prediction.
        
        Args:
            inputs: Input data to predict on
            verbose: Verbosity level
            
        Returns:
            Dictionary with mean and std of predictions
        """
        # Get predictions from each model
        all_predictions = []
        
        for i, model in enumerate(self.models):
            if verbose:
                print(f"Generating predictions for model {i+1}/{self.n_models}")
            
            pred = model.predict(inputs, verbose=0)
            all_predictions.append(pred)
        
        # Stack predictions into a single array of shape (n_models, n_samples, 1)
        all_predictions = np.stack(all_predictions)
        
        # Calculate mean and std across models (axis 0)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'all_predictions': all_predictions
        }

# Create the ensemble model
ensemble_model = EnsembleModel(models_list)

# Example of how to make predictions with both methods
def compare_prediction_methods(input_data):
    """
    Compare predictions from weight-averaged model and ensemble model
    """
    # Get prediction from the weight-averaged model
    avg_prediction = averaged_model.predict(input_data, verbose=0)
    
    # Get prediction statistics from ensemble model
    ensemble_results = ensemble_model.predict(input_data, verbose=0)
    ensemble_mean = ensemble_results['mean']
    ensemble_std = ensemble_results['std']
    
    return {
        'weight_averaged_prediction': avg_prediction,
        'ensemble_mean': ensemble_mean,
        'ensemble_std': ensemble_std
    }

# Save the ensemble model (Since we can't directly save the class, 
# we'll save each model separately and recreate the ensemble when loading)
def save_ensemble_info(ensemble_model, save_dir='ensemble_models'):
    """
    Save information about the ensemble model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each model in the ensemble
    for i, model in enumerate(ensemble_model.models):
        model.save(os.path.join(save_dir, f'model_{i}.h5'))
    
    # Save metadata
    metadata = {
        'n_models': ensemble_model.n_models,
        'model_filenames': [f'model_{i}.h5' for i in range(ensemble_model.n_models)]
    }
    
    # Save metadata as JSON
    import json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    print(f"Ensemble information saved to {save_dir}")

# Function to load the ensemble model
def load_ensemble_model(save_dir='ensemble_models'):
    """
    Load the ensemble model from saved files
    """
    import json
    
    # Load metadata
    with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load all models
    models = []
    for filename in metadata['model_filenames']:
        model = tf.keras.models.load_model(
            os.path.join(save_dir, filename),
            custom_objects={
                'custom_weighted_loss': custom_weighted_loss,
                'CustomWeightedLoss': CustomWeightedLoss,
                'train_weighted_loss': custom_weighted_loss
            }
        )
        models.append(model)
    
    # Create and return the ensemble
    return EnsembleModel(models)

# Example of usage (uncomment to use)
# save_ensemble_info(ensemble_model)
# loaded_ensemble = load_ensemble_model()

################## APPROACH 3: MONTE CARLO DROPOUT (BAYESIAN APPROXIMATION) ##################

def create_mc_dropout_model():
    """
    Create a version of the model with dropout layers for uncertainty estimation
    using Monte Carlo dropout approach
    """
    # Create a fresh model with the same architecture but add dropout
    qT = tf.keras.Input(shape=(1,), name='qT')
    QM = tf.keras.Input(shape=(1,), name='QM')
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    pdfs_x1x2 = tf.keras.Input(shape=(1,), name='pdfs_x1x2')
    pdfs_x2x1 = tf.keras.Input(shape=(1,), name='pdfs_x2x1') 

    # Modified DNNS with dropout
    def DNNS_with_dropout(name):
        model, _ = build_progressive_model(
            input_shape=(3,),
            depth=3,
            width=100,
            use_residual=False,
            initializer_range=0.1,
            activations=['relu', 'tanh', 'relu'],
            output_activation='softplus',
            name=name
        )
        
        # Add dropout layers
        inputs = model.input
        x = inputs
        for i, layer in enumerate(model.layers[1:-1]):  # Skip input and output layers
            x = layer(x)
            # Add dropout after each hidden layer
            x = tf.keras.layers.Dropout(0.1)(x, training=True)  # Keep training=True for inference
        
        outputs = model.layers[-1](x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Modified DNNB with dropout
    def DNNB_with_dropout(name):
        model, _ = build_progressive_model(
            input_shape=(1,),
            depth=3,
            width=100,
            use_residual=False,
            activations=['relu', 'tanh', 'relu'],
            output_activation='linear',
            name=name
        )
        
        # Add dropout layers
        inputs = model.input
        x = inputs
        for i, layer in enumerate(model.layers[1:-1]):  # Skip input and output layers
            x = layer(x)
            # Add dropout after each hidden layer
            x = tf.keras.layers.Dropout(0.1)(x, training=True)  # Keep training=True for inference
        
        outputs = model.layers[-1](x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    SModel = DNNS_with_dropout('SqT')
    BModel = DNNB_with_dropout('BQM')
    
    concatenatedx1x2 = tf.keras.layers.Concatenate()([qT, x1, x2])
    concatenatedx2x1 = tf.keras.layers.Concatenate()([qT, x2, x1])
    
    Sqx1x2 = SModel(concatenatedx1x2)
    Sqx2x1 = SModel(concatenatedx2x1)
    BQM = BModel(QM)

    pdfs_sqT_x1x2 = tf.keras.layers.Multiply()([pdfs_x1x2, Sqx1x2])
    pdfs_sqT_x2x1 = tf.keras.layers.Multiply()([pdfs_x2x1, Sqx2x1])
    
    # Calculate combined S contribution
    PDFs_S_combined = tf.keras.layers.Add()([pdfs_sqT_x1x2, pdfs_sqT_x2x1])
    
    # Multiply with pre-calculated PDFs
    SB_PDF = tf.keras.layers.Multiply()([PDFs_S_combined, BQM])
    
    model = tf.keras.Model([qT, QM, x1, x2, pdfs_x1x2, pdfs_x2x1], SB_PDF)
    
    # Copy weights from the averaged model
    source_model = averaged_model
    target_layers = []
    source_layers = []
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            target_layers.append(layer)
    
    for layer in source_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            source_layers.append(layer)
    
    # Transfer weights for matching dense layers
    for target_layer, source_layer in zip(target_layers, source_layers):
        if target_layer.get_config()['units'] == source_layer.get_config()['units'] and \
           target_layer.get_config()['activation'] == source_layer.get_config()['activation']:
            target_layer.set_weights(source_layer.get_weights())
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss=custom_weighted_loss,
        metrics=['mae']
    )
    
    return model

# Create the MC dropout model
mc_dropout_model = create_mc_dropout_model()

def predict_with_uncertainty(model, inputs, n_iter=100):
    """
    Make predictions with uncertainty using MC Dropout
    
    Args:
        model: The MC dropout model
        inputs: Input data to predict on
        n_iter: Number of forward passes
        
    Returns:
        Dictionary with mean and std of predictions
    """
    predictions = []
    
    for _ in range(n_iter):
        pred = model.predict(inputs, verbose=0)
        predictions.append(pred)
    
    # Stack predictions into a single array
    predictions = np.stack(predictions)
    
    # Calculate mean and std across iterations
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'all_predictions': predictions
    }

# Save the MC dropout model
mc_model_path = 'mc_dropout_model.h5'
mc_dropout_model.save(mc_model_path)

# Example function to compare all three methods
def compare_all_methods(input_data):
    """
    Compare predictions from all three methods:
    1. Weight-averaged model
    2. Ensemble model
    3. MC Dropout model
    
    Args:
        input_data: Input data for prediction
        
    Returns:
        Dictionary with results from all methods
    """
    # Get prediction from the weight-averaged model
    avg_prediction = averaged_model.predict(input_data, verbose=0)
    
    # Get prediction statistics from ensemble model
    ensemble_results = ensemble_model.predict(input_data, verbose=0)
    ensemble_mean = ensemble_results['mean']
    ensemble_std = ensemble_results['std']
    
    # Get prediction statistics from MC Dropout model
    mc_results = predict_with_uncertainty(mc_dropout_model, input_data, n_iter=100)
    mc_mean = mc_results['mean']
    mc_std = mc_results['std']
    
    return {
        'weight_averaged': {
            'prediction': avg_prediction
        },
        'ensemble': {
            'mean': ensemble_mean,
            'std': ensemble_std
        },
        'mc_dropout': {
            'mean': mc_mean,
            'std': mc_std
        }
    }

# Visualization functions
def plot_comparison(input_data, results):
    """
    Plot comparison of predictions from different methods
    
    Args:
        input_data: Input data used for prediction
        results: Results dictionary from compare_all_methods
    """
    plt.figure(figsize=(12, 8))
    
    # Extract predictions
    avg_pred = results['weight_averaged']['prediction']
    ens_mean = results['ensemble']['mean']
    ens_std = results['ensemble']['std']
    mc_mean = results['mc_dropout']['mean']
    mc_std = results['mc_dropout']['std']
    
    # Sort by input value for better visualization
    if isinstance(input_data, dict) and 'qT' in input_data:
        sort_idx = np.argsort(input_data['qT'].flatten())
        x_values = input_data['qT'].flatten()[sort_idx]
    else:
        sort_idx = np.arange(len(avg_pred))
        x_values = np.arange(len(avg_pred))
    
    # Plot weight-averaged model prediction
    plt.plot(x_values, avg_pred.flatten()[sort_idx], 'b-', label='Weight-Averaged')
    
    # Plot ensemble mean with std
    plt.plot(x_values, ens_mean.flatten()[sort_idx], 'r-', label='Ensemble Mean')
    plt.fill_between(
        x_values,
        (ens_mean - ens_std).flatten()[sort_idx],
        (ens_mean + ens_std).flatten()[sort_idx],
        color='r', alpha=0.2, label='Ensemble Std'
    )
    
    # Plot MC dropout mean with std
    plt.plot(x_values, mc_mean.flatten()[sort_idx], 'g-', label='MC Dropout Mean')
    plt.fill_between(
        x_values,
        (mc_mean - mc_std).flatten()[sort_idx],
        (mc_mean + mc_std).flatten()[sort_idx],
        color='g', alpha=0.2, label='MC Dropout Std'
    )
    
    plt.xlabel('Input Value')
    plt.ylabel('Prediction')
    plt.title('Comparison of DNN Model Averaging Methods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    print("Comparison plot saved as 'model_comparison.png'")

# Example usage of all methods
# Note: You would need to replace this with your actual input data
sample_input = {'qT': np.array([[0.1], [0.2], [0.3]]), 'QM': np.array([[1.0], [1.0], [1.0]]), 
               'x1': np.array([[0.5], [0.5], [0.5]]), 'x2': np.array([[0.5], [0.5], [0.5]]),
               'pdfs_x1x2': np.array([[1.0], [1.0], [1.0]]), 'pdfs_x2x1': np.array([[1.0], [1.0], [1.0]])}

results = compare_all_methods(sample_input)
plot_comparison(sample_input, results)