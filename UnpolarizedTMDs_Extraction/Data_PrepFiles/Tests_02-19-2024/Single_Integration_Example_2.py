import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Define the function to integrate
def f(x):
    return x ** 2

# Generate synthetic data for training
x_train = np.linspace(0, 1, 100).reshape(-1, 1)  # Example: integrate from 0 to 1 with 100 points
y_train = f(x_train)

# Compute elemental areas using the trapezoidal rule
elemental_areas = np.diff(x_train.flatten()) * (y_train[:-1] + y_train[1:]) / 2

# Define the DNN model
def DNN_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Create the DNN model
dnn_model = DNN_model()

# Compile the model
dnn_model.compile(optimizer='adam', loss='mse')

# Train the model to predict elemental areas
dnn_model.fit(x_train[:-1], elemental_areas, epochs=10, batch_size=32)

# Evaluate the model
elemental_areas_pred = dnn_model.predict(x_train[:-1])

# Compute the integral using predicted elemental areas
integral_pred = np.sum(elemental_areas_pred)

print("Integral using predicted elemental areas:", integral_pred)
