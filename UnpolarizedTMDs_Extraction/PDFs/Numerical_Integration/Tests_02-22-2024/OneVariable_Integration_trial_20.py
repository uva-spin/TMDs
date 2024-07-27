import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.python.ops import array_ops
from scipy.integrate import simps

############ Generating Pseudodata #################

def f(x):
    return 2*x + 1

# def f(x):
#     return 2*x**2 + 1

# def f(x):
#     return np.sin(x)

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


def DNN_model(width=20, L1_reg=10**(-12), activation='relu'):
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




# def custom_loss(y_true, y_pred):
#     tempy = []
#     dx = xAvg[1] - xAvg[0]
#     for i in range(len(xAvg)):
#         tx_low = xAvg[i] - dx / 2
#         tx_high = xAvg[i] + dx / 2
#         xtemp = tf.linspace(tx_low, tx_high, 100)
#         yy = model(xtemp)
#         integral = tf.reduce_sum(yy) * dx / 100  
#         tempy.append(integral)
#     y_pred = tf.convert_to_tensor(tempy)
#     sqrd_diff = tf.square(y_true - y_pred)
#     return tf.reduce_mean(sqrd_diff, axis=-1)




def simpsons_rule(y, dx):
    # Simpson's rule formula
    integral = (dx / 3) * (
        y[0] + 4 * tf.reduce_sum(y[1:-1:2]) + 2 * tf.reduce_sum(y[2:-2:2]) + y[-1]
    )
    return integral




def custom_loss(y_true,y_pred):
    tempy = []
    dx = xAvg[1]- xAvg[0]
    for i in range(len(xAvg)):
        tx_low = xAvg[i] - dx/2
        tx_high = xAvg[i] + dx/2
        xtemp = tf.constant(np.linspace(tx_low, tx_high, 100))
        #print(len(xtemp))
        yy = model(xtemp)
        #integral = simps(yy[:,0],xtemp)
        #integral = simps(yy,dx)
        integral = simpsons_rule(yy,dx)
        tempy.append(integral[0])
    # y_pred = tf.convert_to_tensor(tempy)
    y_pred = tempy
    sqrd_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(sqrd_diff, axis=-1)
    # return (tempy,y_true)


def diff(xx,aa):
    dx = xx[1] - xx[0]
    grad = np.gradient(aa,dx)
    return np.array(grad)

#print(diff(xAvg,Avals))

true_y_values = f(xAvg)

# diffArray = diff(xAvg,Avals)
# print(diffArray)

# print(custom_loss(Avals,y_pred_array(xAvg)))
#print(custom_loss(Avals,Avals))

lr = 0.01
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=custom_loss)

#history = model.fit(xAvg[:-1], Avals[:-1], epochs=100, verbose=2)
history = model.fit(xAvg, Avals, epochs=300, verbose=2)
# history = model.fit(xAvg, diffArray, epochs=500, verbose=2)
# history = model.fit(xAvg, true_y_values, epochs=500, verbose=2)


# # Plotting the training loss
# plt.figure(1,figsize=(10, 6))
# plt.plot(history.history['loss'])
# plt.title('Model Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()



# Generate y values for the true function f(x)
true_y_values = f(xAvg)

a_sum = np.sum(Avals)
print(a_sum)

# Generate y values for the model predictions
# predicted_y_values = model.predict(xAvg)/(xAvg[1]-xAvg[0])
predicted_y_values = model.predict(xAvg)


# Plot f(x) vs model predictions
plt.figure(2,figsize=(10, 6))
plt.plot(xAvg, Avals, label='True A', color='green')
#plt.plot(xAvg, diffArray, label='True diff_A', color='green')
#plt.plot(xAvg, true_y_values, label='True Function f(x)', color='blue')
plt.plot(xAvg, predicted_y_values, label='Model Predictions', color='red')
plt.title('True Function vs Model Predictions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
