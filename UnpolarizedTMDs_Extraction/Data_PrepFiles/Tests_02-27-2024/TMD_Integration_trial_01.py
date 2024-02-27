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

def S(k):
    return (1/np.pi/4)*np.exp(-k**2/4)

# def S(k):
#     return 2*k**2/(k**2 + 4)

def fxk(x,k):
    return f(x)*S(k)


Xvals = np.linspace(0.0001, 1, 100)
Kvals = np.linspace(0, 5, 100)

def Apseudo(xx,kk):
    tempx, tempkavg, tempA = [], [], []
    for j in range(len(xx)):
        for i in range(len(kk)-1):
            tempx.append(xx[j])
            tempkavg.append(0.5*(kk[i]+kk[i+1]))
            tempy = simps(fxk(xx[j],np.linspace(kk[i], kk[i+1], 50)), dx=(kk[i+1]-kk[i])/50)
            # dx=(kk[i+1]-kk[i])/50
            # tempy = 0.5*(f(kk[i+1])+f(kk[i]))*dx
            tempfxSk = tempy
            tempA.append(tempfxSk)
    return np.array(tempx), np.array(tempkavg), np.array(tempA)

xVals, kAvg, Avals = Apseudo(Xvals,Kvals)

df = pd.DataFrame({'x': xVals, 'k': kAvg, 'A': Avals})

############ Fitting to Pseudodata #################

print(Avals)

plt.figure(1,figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('x')
plt.ylabel('Loss')
plt.show()


# def DNN_model(width=150, L1_reg=10**(-12), activation='relu'):
#     inp = tf.keras.Input(shape=(1,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
#     x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     nnout = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x2)
#     mod = tf.keras.Model(inp, nnout)
#     return mod

# # Create the DNN model
# model = DNN_model()




# true_y_values = f(kAvg)


# lr = 0.00001
# model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')


# history = model.fit(kAvg, Avals, epochs=200, verbose=2)

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
