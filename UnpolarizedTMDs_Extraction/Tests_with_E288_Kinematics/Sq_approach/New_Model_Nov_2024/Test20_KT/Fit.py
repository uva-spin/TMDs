import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import Hyperband

# Define the model building function
def build_model(hp):
    model = keras.Sequential()
    # Add layers with hyperparameters
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=256, step=32),
            activation=hp.Choice("activation", ["relu", "tanh"]),
            input_shape=(2,),  # Input features are combined (qT, QM)
        )
    )
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    model.add(layers.Dense(1, activation="linear"))  # Output layer
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

# Generate some example data (replace these with your actual data)
qT_values = np.random.rand(100)  # Replace with actual qT values
QM_values = np.random.rand(100)  # Replace with actual QM values
A_true = 2 * qT_values + 3 * QM_values + np.random.normal(0, 0.1, size=100)  # Replace with actual A_true values

# Combine qT_values and QM_values into a single feature array
X = np.column_stack((qT_values, QM_values))  # Shape: (num_samples, 2)
y = A_true  # Target values

# Instantiate the tuner
tuner = Hyperband(
    build_model,
    objective="val_mean_absolute_error",
    max_epochs=20,
    factor=3,
    directory="my_dir",
    project_name="hyperparameter_tuning",
)

# Callback to stop early if validation loss does not improve
stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Perform the search
tuner.search(X, y, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X, y, epochs=50, validation_split=0.2)

# Evaluate the model
val_loss, val_mae = model.evaluate(X, y, verbose=0)
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")
