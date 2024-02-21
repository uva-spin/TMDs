import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from scipy.integrate import simps

############ Generating Pseudodata #################

def f(x):
    return 2*x + 1

Xvals = np.linspace(0, 1, 100)

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


def y_pred_array(xxa):
    tempy = []
    dx = xxa[1]- xxa[0]
    for i in range(len(xxa)):
        #dx = xxa[i+1]- xxa[i]
        tx_low = xxa[i] - dx/2
        tx_high = xxa[i] + dx/2
        xtemp = np.array(np.linspace(tx_low,tx_high,100))
        yy = model.predict(xtemp)
        integral = simps(yy[:,0],xtemp)
        tempy.append(integral)
    return np.array(tempy)


#print(y_pred_array(xAvg))

def custom_loss(y_true,y_pred):
    sqrd_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(sqrd_diff, axis=-1)


# def custom_loss(y_true,y_pred):
#     tempy = []
#     dx = xAvg[1]- xAvg[0]
#     for i in range(len(xAvg)):
#         tx_low = xAvg[i] - dx/2
#         tx_high = xAvg[i] + dx/2
#         xtemp = np.array(np.linspace(tx_low,tx_high,100))
#         #yy = model.predict(xtemp)
#         yy = model(xtemp)
#         integral = simps(yy[:,0],xtemp)
#         tempy.append(integral)
#     y_pred = np.array(tempy)
#     sqrd_diff = tf.square(y_true - y_pred)
#     return tf.reduce_mean(sqrd_diff, axis=-1)



print(custom_loss(Avals,y_pred_array(xAvg)))

#model.compile(optimizer='adam', loss=custom_loss)

#history = model.fit(xAvg[:-1], Avals[:-1], epochs=100, verbose=0)
# # Assuming you have some training data X_train, y_train
# # Train the model
# history = model.fit(xAvg[:-1], Avals[:-1], epochs=100, verbose=0)

# # Plotting the training loss
# plt.plot(history.history['loss'])
# plt.title('Model Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()
