import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from scipy.integrate import simps

############ Generating Pseudodata #################

def f(x):
    return 2*x + 1

Xvals = np.linspace(0, 1, 1000)

def Apseudo(xx):
    tempx1, tempx2, tempxavg, tempA = [], [], [], []
    for i in range(len(xx)-1):
        tempx1.append(xx[i])
        tempx2.append(xx[i+1])
        tempxavg.append(0.5*(xx[i]+xx[i+1]))
        tempy = simps(f(np.linspace(xx[i], xx[i+1], 50)), dx=(xx[i+1]-xx[i])/50)
        tempA.append(tempy)
    return np.array(tempx1), np.array(tempx2), np.array(tempxavg), np.array(tempA)

x1vals, x2vals, xAvg, Avals = Apseudo(Xvals)

df = pd.DataFrame({'x1': x1vals, 'x2': x2vals, 'x': xAvg, 'A': Avals})

############ Fitting to Pseudodata #################

print(df)

def DNN_model(width=100, L1_reg=10**(-12), activation='relu'):
    inp = tf.keras.Input(shape=(1,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    nnout = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x2)
    mod = tf.keras.Model(inp, nnout)
    return mod

def Amodel(model):
    # Define a custom Lambda layer for Simpson's rule integration
    def simpsons_rule_integration(x):
        # Reshape the input tensor to ensure multiple features per sample
        x = tf.reshape(x, (-1, 2))  # Assuming x has 2 features per sample
        
        # Initialize list to store integral values
        integrals = []
        
        # Iterate over intervals
        for i in range(tf.shape(x)[0]):
            # Generate input values within the interval
            input_values = tf.linspace(x[i, 0], x[i, 1], 50)
            
            # Pass input values through the model to get output values
            output_values = model(input_values[:, tf.newaxis])
            
            # Apply Simpson's rule to integrate output values
            dx = (x[i, 1] - x[i, 0]) / 50
            integral = simps(output_values[:, 0], dx=dx)
            
            # Append integral value to the list
            integrals.append(integral)
        
        # Convert list to tensor
        integrals = tf.convert_to_tensor(integrals)
        return integrals
    
    # Define input layer
    input_layer = model.input
    
    # Define output layer
    output_layer = layers.Lambda(simpsons_rule_integration, name='integral')(input_layer)
    
    # Define the model
    integral_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    return integral_model

# Create the A model
x_train = df['x'].values.reshape(-1, 1)
y_train = df['A'].values.reshape(-1, 1)
dnn_model = DNN_model()
amodel = Amodel(dnn_model)

# Print model summary
amodel.summary()

amodel.compile(optimizer='adam', loss='mse')
amodel.fit(x_train, y_train, epochs=10, batch_size=32)

# Generate predictions using the Amodel
predictions = amodel.predict(x_train)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_train, predictions, label='Predicted A')
plt.plot(x_train, Avals, label='Actual A')
plt.xlabel('x')
plt.ylabel('A')
plt.title('Predicted vs Actual A')
plt.legend()
plt.grid(True)
plt.show()
