import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the numerical integration function using Simpson's rule
def simpsons_rule(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    return h/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])

# Define the S(k, Q^2) function using a DNN model
def S_function_model(k, Q2, model):
    inputs = np.column_stack((k, Q2))
    return model.predict(inputs).flatten()

# Define the Sfunction
def Sfunction(k, Q2):
    return 1 / (k**2 + Q2)

# Generate Q2 values
Q2vals = np.linspace(1, 10, 10)

# Generate a dataframe for Q2 values and corresponding FUU values
k_min = 0 
k_max = 1

def FUU_model(Q2arr, model):
    Fvals = []
    for Q2 in Q2arr:
        def integrand_model(k):
            return S_function_model(k, np.full_like(k, Q2), model)
        # Perform the integration using Simpson's rule for the model
        integral_model = simpsons_rule(integrand_model, k_min, k_max, 100)
        Fvals.append(integral_model)
    return np.array(Q2arr), np.array(Fvals)

def FUU_function(Q2arr):
    Svals = []  # To store FUU calculated using Sfunction
    for Q2 in Q2arr:
        def integrand_function(k):
            return Sfunction(k, Q2)
        # Perform the integration using Simpson's rule for the function
        integral_function = simpsons_rule(integrand_function, k_min, k_max, 100)
        Svals.append(integral_function)
    return np.array(Q2arr), np.array(Svals)

# Create a DNN model to mimic the S(k, Q^2) function
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Generate pseudo-data for training
np.random.seed(42)  # For reproducibility
k_train = np.random.uniform(0, 1, size=(1000,))
Q2_train = np.random.uniform(1, 10, size=(1000,))

# Calculate FUU values using the DNN model
_, FUU_train_model = FUU_model(Q2_train, model)

# Reshape the arrays for compatibility with model fitting
FUU_train_model = FUU_train_model.reshape(-1, 1)

# Train the model using FUU_model
model.fit(np.column_stack((Q2_train, np.zeros_like(Q2_train))), FUU_train_model, epochs=100, batch_size=16)

# Use the trained model to calculate FUU for new Q2 values
Q2_new = np.linspace(1, 10, 100)
k_new = np.linspace(0, 1, 100)
inputs_new = np.column_stack((k_new, Q2_new))
predicted_FUU = model.predict(np.column_stack((Q2_new, np.zeros_like(Q2_new))))

# Calculate FUU for new Q2 values using Sfunction
_, Svals = FUU_function(Q2_new)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(Q2_new, predicted_FUU, label='Predicted FUU (DNN)')
plt.plot(Q2_new, Svals, 'r--', label='FUU (Sfunction)', alpha=0.7)
plt.xlabel('Q2')
plt.ylabel('FUU')
plt.legend()
plt.show()
