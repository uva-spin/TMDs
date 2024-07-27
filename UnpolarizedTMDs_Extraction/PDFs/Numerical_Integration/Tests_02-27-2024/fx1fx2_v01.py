import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.python.ops import array_ops
from scipy.integrate import simps

############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

def fx1x2(x1,x2):
    return f(x1)*f(x2)

x1vals = np.linspace(0.0001, 0.4, 100)
x2vals = np.linspace(0.5, 1, 100)

def Apseudo(x1,x2):
    tempx1, tempx2, tempA = [], [], []
    for i in range(len(x1)):
        for j in range(len(x2)):
            tempx1.append(x1[i])
            tempx2.append(x2[j])
            tempfx1fx2 = fx1x2(x1[i],x2[j])
            tempA.append(tempfx1fx2)
    return np.array(tempx1), np.array(tempx2), np.array(tempA)

x1Vals, x2Vals, Avals = Apseudo(x1vals,x2vals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'A': Avals})
df.to_csv('x1x2vsfx1x2.csv')

############ Fitting to Pseudodata #################

def DNN_model(width=20, L1_reg=10**(-12), activation='relu'):
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    nnout = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x2)
    mod = tf.keras.Model(inp, nnout)
    return mod



# Concatenate input tensors
concatenated_inputs = np.column_stack((x1Vals, x2Vals))

# Create the DNN model
model = DNN_model()

lr = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')

history = model.fit(concatenated_inputs, Avals, epochs=200, verbose=2)


# def f(x):
#     return (x**0.1)*((1-x)**0.3)


# def fx1x2(x1,x2):
#     return f(x1)*f(x2)


# x1vals = np.linspace(0.0001, 0.4, 100)
# x2vals = np.linspace(0.5, 1, 100)

# def Apseudo(x1,x2):
#     tempx1, tempx2, tempA = [], [], []
#     for i in range(len(x1)):
#         for j in range(len(x2)):
#             tempx1.append(x1[j])
#             tempx2.append(x2[j])
#             tempfx1fx2 = fx1x2(x1[i],x2[j])
#             tempA.append(tempfx1fx2)
#     return np.array(tempx1), np.array(tempx2), np.array(tempA)

# x1Vals, x2Vals, Avals = Apseudo(x1vals,x2vals)

# df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'A': Avals})
# df.to_csv('x1x2vsfx1x2.csv')

# ############ Fitting to Pseudodata #################


# def DNN_model(width=100, L1_reg=10**(-12), activation='relu'):
#     inp = tf.keras.Input(shape=(2))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
#     x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     nnout = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x2)
#     mod = tf.keras.Model(inp, nnout)
#     return mod

# #Create the DNN model
# model = DNN_model()(tf.keras.layers.concatenate([x1Vals,x2Vals]))

# lr = 0.00001
# model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')


# history = model.fit([x1Vals,x2Vals], Avals, epochs=200, verbose=2)

# # Plotting the training loss
# plt.figure(1,figsize=(10, 6))
# plt.plot(history.history['loss'])
# plt.title('Model Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()



# # Generate y values for the true function f(x)
# true_y_values = f(kAvg)*fux01

# a_sum = np.sum(Avals)
# print(a_sum)

# # Generate y values for the model predictions
# predicted_y_values = model.predict(kAvg)/(kAvg[1]-kAvg[0])
# #predicted_y_values = model.predict(kAvg)/(kAvg[1]-kAvg[0])/fux01
# #predicted_y_values = model.predict(xAvg)
# #print(np.sum(predicted_y_values))


# # Plot f(x) vs model predictions
# plt.figure(2,figsize=(10, 6))
# #plt.plot(xAvg, Avals, label='True A', color='green')
# #plt.plot(xAvg, diffArray, label='True diff_A', color='green')
# plt.plot(kAvg, true_y_values, label='True Function S(k)', color='blue')
# plt.plot(kAvg, predicted_y_values, label='Model Predictions', color='red')
# plt.title('True Function vs Model Predictions')
# plt.xlabel('k')
# plt.ylabel('S(k)')
# plt.legend()
# plt.show()
