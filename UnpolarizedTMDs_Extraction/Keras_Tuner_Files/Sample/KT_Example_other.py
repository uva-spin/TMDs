# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from kerastuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split

# Step 1: Generating Synthetic Data
# This generates data based on a polynomial function with some noise
np.random.seed(42)  # Seed for reproducibility
x = np.linspace(-1, 1, 400)
y = 0.5 * x**3 - 0.2 * x**2 + 0.1 * x - 0.2 + np.random.normal(scale=0.05, size=x.shape)

# Step 2: Building the Model and Optimizing it with Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,)))  # Input layer
    
    # Tuning the number of layers and their configurations
    # Keras Tuner will choose an optimal number of layers, units in each layer, and their activation functions
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation_' + str(i), ['relu', 'tanh', 'sigmoid'])
        ))
        
    model.add(layers.Dense(1))  # Output layer
    
    # Compilation of the model with a tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return model

# Initiating the tuner
# Here, BayesianOptimization is used to find the best hyperparameter values by building and evaluating different models
tuner = BayesianOptimization(
    build_model,
    objective='val_mean_squared_error',
    max_trials=10,  # The total number of trials (model configurations) to test
    executions_per_trial=1,  # The number of models that should be built and fit for each trial
    directory='my_dir',
    project_name='keras_tuner_demo'
)

# Splitting data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Start the search for the best hyperparameter configuration
# The tuner explores different configurations, evaluating them on the validation set
tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# After the search, retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Step 3: Training the Best Model Found by Keras Tuner
# The model is built using the best hyperparameters and then trained on the full dataset
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# Optional: plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

