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


# def DNN_model(input_shape):
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=input_shape),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model

# def Amodel(model, xAvg):
#     tempy = model.predict(xAvg)
#     integral = np.trapz(tempy[:,0], xAvg)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Assuming xAvg is generated from some range of x values
# xAvg = np.linspace(min(x_train), max(x_train), 100)

# # Create the DNN model
# input_shape = (1,)  # Assuming 'x' is a single feature
# dnn_model = DNN_model(input_shape)

# # Train the DNN model
# dnn_model.compile(optimizer='adam', loss='mse')
# dnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Create the A model
# amodel = Amodel(dnn_model, xAvg)

# # Print model summary
# amodel.summary()


# def DNN_model(input_shape):
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=input_shape),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model

# def Amodel(model, xAvg):
#     tempy = model.predict(xAvg)
#     integral = np.trapz(tempy[:,0], xAvg)
#     #integral_tensor = tf.keras.layers.Layer()[integral]
#     #integral_tensor = tf.constant(integral, dtype=tf.float32)  # Convert integral to a TensorFlow tensor
#     integral_tensor = tf.constant(integral)
#     return tf.keras.Model(inputs=model.input, outputs=integral_tensor)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Assuming xAvg is generated from some range of x values
# xAvg = np.linspace(min(x_train), max(x_train), 100)

# # Create the DNN model
# input_shape = (1,)  # Assuming 'x' is a single feature
# dnn_model = DNN_model(input_shape)

# # Train the DNN model
# dnn_model.compile(optimizer='adam', loss='mse')
# dnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Create the A model
# amodel = Amodel(dnn_model, xAvg)

# # Print model summary
# amodel.summary()

# def DNN_model(input_shape):
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=input_shape),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model

# class IntegralLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(IntegralLa[model.output, yer, self).__init__(**kwargs)

#     def call(self, inputs):
#         tempy = inputs
#         integral = tf.reduce_sum(tempy)
#         return integral

# def Amodel(model, xAvg):
#     tempy = model.predict(xAvg)
#     integral_layer = IntegralLayer()(tempy)
#     return tf.keras.Model(inputs=model.input, outputs=integral_layer)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Assuming xAvg is generated from some range of x values
# xAvg = np.linspace(min(x_train), max(x_train), 100)

# # Create the DNN model
# input_shape = (1,)  # Assuming 'x' is a single feature
# dnn_model = DNN_model(input_shape)

# # Train the DNN model
# dnn_model.compile(optimizer='adam', loss='mse')
# dnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Create the A model
# amodel = Amodel(dnn_model, xAvg)

# # Print model summary
# amodel.summary()

# def DNN_model(input_shape):
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=input_shape),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model

# def integral_layer(x):
#     integral = tf.reduce_sum(x, axis=1)  # Assuming x is of shape (batch_size, num_features)
#     return integral

# def Amodel(model, xAvg):
#     tempy = model.predict(xAvg)
#     integral = layers.Lambda(integral_layer)(tempy)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Assuming xAvg is generated from some range of x values
# xAvg = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)  # Reshape to match model input shape

# # Create the DNN model
# input_shape = (1,)  # Assuming 'x' is a single feature
# dnn_model = DNN_model(input_shape)

# # Train the DNN model
# dnn_model.compile(optimizer='adam', loss='mse')
# dnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Create the A model
# amodel = Amodel(dnn_model, xAvg)

# # Print model summary
# amodel.summary()



## worked ##

# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np

# def DNN_model(input_shape):
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=input_shape),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model

# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Create the DNN model
# input_shape = (1,)  # Assuming 'x' is a single feature
# dnn_model = DNN_model(input_shape)

# # Train the DNN model
# dnn_model.compile(optimizer='adam', loss='mse')
# dnn_model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Create the A model
# amodel = Amodel(dnn_model)

# # Print model summary
# amodel.summary()


### Worked ###

# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np

# def DNN_model():
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=(1,)),  # Adjust input shape to (2,)
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)  # Output layer
#     ])
#     return model

# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Create the DNN model
# #input_shape = (1,)  # Assuming 'x' is a single feature
# dnn_model = DNN_model()

# # Create the A model
# amodel = Amodel(dnn_model)

# # Print model summary
# amodel.summary()

# amodel.compile(optimizer='adam', loss='mse')
# amodel.fit(x_train, y_train, epochs=10, batch_size=32)

# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np

# def DNN_model():
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=(1,)),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)
#     ])
#     return model

# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.py_function(func=np.trapz, inp=[x[:, 0]], Tout=tf.float32))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values
# y_train = df['A'].values

# # Create the DNN model
# dnn_model = DNN_model()

# # Create the A model with np.trapz integration
# amodel = Amodel(dnn_model)

# # Print model summary
# amodel.summary()

# # Compile and train the A model
# amodel.compile(optimizer='adam', loss='mse')
# amodel.fit(x_train, y_train, epochs=10, batch_size=32)


# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np

# def DNN_model():
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=(1,)),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)
#     ])
#     return model

# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.py_function(func=np.trapz, inp=[x[:, 0]], Tout=tf.float32))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values.reshape(-1,1)
# y_train = df['A'].values.reshape(-1,1)

# # Create the DNN model
# dnn_model = DNN_model()

# # Create the A model with np.trapz integration
# amodel = Amodel(dnn_model)

# # Print model summary
# amodel.summary()

# # Compile and train the A model
# amodel.compile(optimizer='adam', loss='mse')
# amodel.fit(x_train, y_train, epochs=10, batch_size=32)

# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np

# def DNN_model():
#     model = tf.keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=(1,)),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)
#     ])
#     return model



# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# def Amodel(model):
#     integral = layers.Lambda(lambda x: tf.py_function(func=np.trapz, inp=[x[:, 0]], Tout=tf.float32, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

# def Amodel(model):
#     yy = model.predict(xAvg)
#     integral = yy
#     #integral = np.trapz(yy[:,0],xAvg)
#     #integral = layers.Lambda(lambda yy: tf.py_function(func=np.trapz, inp=[yy[:, 0]], Tout=tf.float32, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)




# class IntegralLayer(tf.keras.layers.Layer):
#     def __init__(self, x, **kwargs):
#         super(IntegralLayer, self).__init__(**kwargs)
#         self.x = x

#     def call(self, inputs):
#         integral = trapezoidal_integral(inputs, self.x)
#         return integral

# def Amodel(model, x):
#     y = model(x)
#     integral_layer = IntegralLayer(x)
#     integral = integral_layer(y)
#     return tf.keras.Model(inputs=model.input, outputs=integral)


# # Example DataFrame df with columns 'x' and 'A'
# x_train = df['x'].values.reshape(-1, 1)
# y_train = df['A'].values.reshape(-1, 1)

# print(y_train)

# # Create the DNN model
# dnn_model = DNN_model()

# # Create the A model with np.trapz integration
# #amodel = Amodel(dnn_model)
# amodel = Amodel(dnn_model,x_train)

# # Print model summary
# amodel.summary()

# # Compile and train the A model
# amodel.compile(optimizer='adam', loss='mse')
# history = amodel.fit(x_train, [y_train], epochs=10, batch_size=32)  # Providing y_train twice as we have two outputs

# # # Example test input
# # test_input = np.array([0.1, 0.2, 0.3,0.4,0.5])  # Example input with shape (3, 1)

# # # Call the Amodel with the test input
# # output = amodel.predict(test_input)

# # # Print the shape of the output
# # print("Output", output)

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
#     integral = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(model.output)
#     return tf.keras.Model(inputs=model.input, outputs=integral)

def Amodel(model):
    x = model.input
    y = model.output
    dx = x[:, 1] - x[:, 0]  # Assuming x is evenly spaced
    # Compute the integral using the trapezoidal rule
    integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx[:, tf.newaxis], axis=1)
    return tf.keras.Model(inputs=x, outputs=integral)

# def Amodel(model):
#     x = model.input
#     y = model.output
    
#     # Reshape x to be 1D
#     x_1d = tf.squeeze(x, axis=-1)

#     # Compute dx (difference between consecutive elements of x)
#     dx = x_1d[1:] - x_1d[:-1]

#     # Compute the integral using the trapezoidal rule
#     integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx, axis=1)

#     return tf.keras.Model(inputs=x, outputs=integral)

# def trapezoidal_integral(y, x):
#     dx = x[1] - x[0]  # Assuming x is evenly spaced
#     integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx, axis=1)
#     return integral

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
amodel.fit(x_train, y_train, epochs=10, batch_size=32)
