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

# Create the DNN model
model = DNN_model()

# Define function to integrate curve
def integrate_curve(y):
    """
    Integrate a curve using Simpson's rule.
    """
    x = np.linspace(0, 1, len(y))
    return simps(y, x)

# Define custom loss function
def custom_loss(y_true, y_pred):
    integral_pred = integrate_curve(y_pred)
    integral_true = integrate_curve(y_true)
    return tf.keras.losses.mean_squared_error(integral_true, integral_pred)

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=custom_loss)

model.summary()

# # Fit the model to the pseudo data
model.fit(xAvg, Avals, epochs=10, batch_size=32)
