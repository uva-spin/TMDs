import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.python.ops import array_ops
from scipy.integrate import simps

############ Generating Pseudodata #################

# def f(x):
#     return 2*x + 1

# def f(x):
#     return 2*x**3 + 2*x**2 +1

# def f(x):
#     return np.sin(x)

# def f(x):
#     return 2*x**4 + 2*x**3 + 2*x**2 + 1

# def f(x):
#     return 2*x*np.sin(np.pi*x)

def f(k):
    return 4*k/(k**2 + 4)


Kvals = np.linspace(0, 10, 1000)
fux01=0.64

def Apseudo(kk):
    tempx1, tempx2, tempxavg, tempA = [], [], [], []
    for i in range(len(kk)-1):
        tempx1.append(kk[i])
        tempx2.append(kk[i+1])
        tempxavg.append(0.5*(kk[i]+kk[i+1]))
        tempy = simps(f(np.linspace(kk[i], kk[i+1], 50)), dx=(kk[i+1]-kk[i])/50)
        # dx=(kk[i+1]-kk[i])/50
        # tempy = 0.5*(f(kk[i+1])+f(kk[i]))*dx
        tempfxSk = tempy*fux01
        tempA.append(tempfxSk)
    return np.array(tempx1), np.array(tempx2), np.array(tempxavg), np.array(tempA)

k1vals, k2vals, kAvg, Avals = Apseudo(Kvals)

df = pd.DataFrame({'x1': k1vals, 'x2': k2vals, 'x': kAvg, 'A': Avals})

############ Fitting to Pseudodata #################

print(df)


def DNN_model(width=150, L1_reg=10**(-12), activation='relu'):
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




true_y_values = f(kAvg)


lr = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')


history = model.fit(kAvg, Avals, epochs=200, verbose=2)

# Plotting the training loss
plt.figure(1,figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()



# Generate y values for the true function f(x)
true_y_values = f(kAvg)*fux01

a_sum = np.sum(Avals)
print(a_sum)

# Generate y values for the model predictions
predicted_y_values = model.predict(kAvg)/(kAvg[1]-kAvg[0])
#predicted_y_values = model.predict(kAvg)/(kAvg[1]-kAvg[0])/fux01
#predicted_y_values = model.predict(xAvg)
#print(np.sum(predicted_y_values))


# Plot f(x) vs model predictions
plt.figure(2,figsize=(10, 6))
#plt.plot(xAvg, Avals, label='True A', color='green')
#plt.plot(xAvg, diffArray, label='True diff_A', color='green')
plt.plot(kAvg, true_y_values, label='True Function S(k)', color='blue')
plt.plot(kAvg, predicted_y_values, label='Model Predictions', color='red')
plt.title('True Function vs Model Predictions')
plt.xlabel('k')
plt.ylabel('S(k)')
plt.legend()
plt.show()
