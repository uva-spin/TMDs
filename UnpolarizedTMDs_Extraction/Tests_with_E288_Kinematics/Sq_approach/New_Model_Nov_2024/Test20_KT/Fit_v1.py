import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
import os
import keras_tuner as kt  # Keras Tuner library

def create_folders(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

# User inputs
kmin = 0.0001
kmax = 10.0
kbins = 100
phibins = 100
epochs = 1000
print_epochs = 10
models_folder = 'Models'

create_folders(models_folder)

# Load the pseudo-data
data = pd.read_csv("A_qT_QM_data.csv")
qT_values = data['qT'].values
QM_values = data['QM'].values
A_true = data['A'].values

# Define the DNN model for S(k, QM) with tunable hyperparameters
def build_dnn(hp):
    model = models.Sequential([
        layers.Input(shape=(2,)),  # Two inputs: k and QM
        layers.Dense(hp.Int('units_1', min_value=32, max_value=128, step=16), 
                     activation=hp.Choice('activation_1', ['relu', 'relu6'])),
        layers.Dense(hp.Int('units_2', min_value=16, max_value=64, step=16), 
                     activation=hp.Choice('activation_2', ['relu', 'relu6'])),
        layers.Dense(hp.Int('units_3', min_value=8, max_value=32, step=8), 
                     activation=hp.Choice('activation_3', ['relu', 'relu6'])),
        layers.Dense(1, activation='exponential')  # Output layer
    ])

    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse')
    return model

# Define a Keras Tuner class for hyperparameter tuning
tuner = kt.Hyperband(
    build_dnn,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='my_tuner_dir',
    project_name='dnn_tuning'
)

# Split the data into training and validation sets
split_idx = int(0.8 * len(data))
train_data = data.iloc[:split_idx]
val_data = data.iloc[split_idx:]

# Prepare inputs and outputs for Keras Tuner
train_inputs = np.stack((train_data['qT'], train_data['QM']), axis=-1)
train_outputs = train_data['A']
val_inputs = np.stack((val_data['qT'], val_data['QM']), axis=-1)
val_outputs = val_data['A']

# Run hyperparameter search
tuner.search(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), epochs=50)

# Retrieve best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Save the best hyperparameters to a .csv file
best_hps_dict = {key: best_hps.get(key) for key in best_hps.values.keys()}
pd.DataFrame([best_hps_dict]).to_csv('best_hyperparameters.csv', index=False)

# Re-train with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), epochs=100)

# Save the trained model
best_model.save(os.path.join(models_folder, 'best_model.h5'))
print("Best model saved successfully!")

# Plot the training and validation loss
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid()
plt.savefig('Loss_Curve.pdf')
