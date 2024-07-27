import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

def f(x):
    return 2*x + 1

Xvals = np.array(np.linspace(0,1,100))

#print(Xvals)

def simpsons_rule(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    return h/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])


def Apseudo(xx):
    tempx1 = []
    tempx2 = []
    tempxavg = []
    tempA = []
    for i in range(0,len(xx)-1):
        tempx1.append(xx[i])
        tempx2.append(xx[i+1])
        tempxavg.append(0.5*(xx[i]+xx[i+1]))
        tempy = simpsons_rule(f,xx[i],xx[i+1], 50)
        tempA.append(tempy)
    return np.array(tempx1), np.array(tempx2), np.array(tempxavg), np.array(tempA)

x1vals, x2vals, xAvg, Avals = Apseudo(Xvals)

df = pd.DataFrame({'x1': x1vals, 'x2': x2vals, 'x': xAvg, 'A': Avals})

#print(df)

# def DNN_model():
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=(1,)),  # Adjust input shape to (2,)
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model


# def Amodel(model):
#     tempy=model.predict(xAvg)
#     integral = np.trapz(tempy[:,0], xAvg)
#     return tf.keras.Model([xAvg], integral)


# x_train = df['x']
# y_train = df['A']

# dnn_model = DNN_model()(xAvg)
# amodel = Amodel(dnn_model)
# amodel.compile(optimizer='adam', loss='mse')
# amodel.summary()

#model.fit(x_train, y_train, epochs=100, batch_size=50)



import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def DNN_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),  # Adjust input shape to (2,)
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer
    ])
    return model

# def Amodel(model):
#     y = model.output
#     integral = tf.reduce_sum(y, axis=1)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# def Amodel(model):
#     y = model.output
#     integral = tf.reduce_sum(y, axis=1)
#     return tf.keras.Model(inputs=model.input, outputs=integral)


# def Amodel(model):
#     y = model.output
#     x = model.input
#     dx = x[:, 0] - x[:, -1]  # Assuming x is evenly spaced
#     dx = tf.reshape(dx, [-1, 1])  # Reshape dx to match the shape of y
#     integral = tf.py_function(func=np.trapz, inp=[y, dx], Tout=tf.float32, axes=1)
#     return tf.keras.Model(inputs=model.input, outputs=integral)


# def Amodel(model):
#     y = model.output
#     x = model.input
#     dx = x[:, 0] - x[:, -1]  # Assuming x is evenly spaced
#     integral = tf.py_function(func=np.trapz, inp=[y, dx], Tout=tf.float32)
#     return tf.keras.Model(inputs=model.input, outputs=integral)


# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# def Amodel(model):
#     x = model.input
#     y = model.output
#     dx = x[:, 1] - x[:, 0]  # Assuming x is evenly spaced
#     # Compute the integral using the trapezoidal rule
#     integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx[:, tf.newaxis], axis=1)
#     return tf.keras.Model(inputs=x, outputs=integral)

# def Amodel(model):
#     x = model.input
#     y = model.output
#     dx = x[:, 0] - tf.roll(x[:, 0], shift=1, axis=0)  # Assuming x is evenly spaced
#     dx = tf.concat([[dx[1]], dx[1:]], axis=0)  # Replace first element with the second element to maintain the same shape
#     # Compute the integral using the trapezoidal rule
#     #integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx[:, tf.newaxis], axis=1)
#     integral = layers.Lambda(lambda y, dx: tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx[:, tf.newaxis], axis=1))([y, dx])
#     return tf.keras.Model(inputs=x, outputs=integral)

# def Amodel(model):
#     x = model.input
#     y = model.output
#     dx = x[:, 1] - x[:, 0]  # Assuming x is evenly spaced
#     # Compute the integral using the trapezoidal rule
#     integral = layers.Lambda(lambda x: tf.reduce_sum(0.5 * (x[:, :-1] + x[:, 1:]) * dx[:, tf.newaxis], axis=1))(y)
#     return tf.keras.Model(inputs=x, outputs=integral)


# def Amodel(model):
#     tempy=model.predict(xAvg)
#     integral = np.trapz(tempy[:,0], xAvg)
#     return tf.keras.Model([xAvg], integral)

# def Amodel(model):
#     x = model.input
#     y = model.output
#     dx = x[1] - x[0]  # Assuming x is evenly spaced
#     integral = np.trapz(y[:,0], dx)
#     return tf.keras.Model(inputs=x, outputs=integral)

# Example DataFrame df with columns 'x' and 'A'
x_train = df['x'].values.reshape(-1, 1)
y_train = df['A'].values.reshape(-1, 1)

# Create the DNN model
#input_shape = (1,)  # Assuming 'x' is a single feature
dnn_model = DNN_model()

# Create the A model
amodel = Amodel(dnn_model)

# Print model summary
amodel.summary()

amodel.compile(optimizer='adam', loss='mse')
amodel.fit(x_train, y_train, epochs=100, batch_size=5)
