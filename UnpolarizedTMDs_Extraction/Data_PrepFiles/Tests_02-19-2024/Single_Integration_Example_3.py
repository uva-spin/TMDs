import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Generate synthetic data for f(x)
def generate_data():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)  # Example function f(x)
    return x, y

# Define the DNN model
def DNN_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),  
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  
    ])
    return model

# Define the trapezoidal integral layer
class TrapezoidalIntegralLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(TrapezoidalIntegralLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        
        # Compute dx assuming x is evenly spaced
        dx = x[:, 1:] - x[:, :-1]

        # Compute the integral using the trapezoidal rule
        integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx, axis=1)

        return integral

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],)

# Define the model containing DNN and trapezoidal integral layer
def Integral_model():
    x_input = layers.Input(shape=(1,))
    dnn_output = DNN_model()(x_input)
    integral_output = TrapezoidalIntegralLayer()([x_input, dnn_output])
    model = Model(inputs=x_input, outputs=integral_output)
    return model

# Generate synthetic data for the integral of f(x)
def generate_integral_data():
    x, y = generate_data()
    integral_y = np.trapz(y, x)  # Compute integral of f(x)
    return x[-1], integral_y  # Return only the endpoint and the integral value

# Generate synthetic data
x_train, y_train = generate_integral_data()

# Create the Integral model
integral_model = Integral_model()

# Compile the model
integral_model.compile(optimizer='adam', loss='mse')

# Train the model
integral_model.fit(np.expand_dims(x_train, axis=-1), np.array([y_train]), epochs=100, verbose=2)

# Plot the results
# x, y = generate_data()
# integral_prediction = integral_model.predict(np.expand_dims(x, axis=-1))

# plt.figure(figsize=(10, 6))
# plt.plot(x, y, label='Original Function (f(x))')
# plt.plot(x, integral_prediction, label='Integral Model Prediction')
# plt.xlabel('x')
# plt.ylabel('f(x) / Integral')
# plt.title('Original Function vs. Integral Model Prediction')
# plt.legend()
# plt.grid(True)
# plt.show()
