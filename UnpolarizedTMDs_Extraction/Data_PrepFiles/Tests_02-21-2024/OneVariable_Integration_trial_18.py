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

Xvals = np.linspace(0, 1, 99)

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


def DNN_model(width=64, L1_reg=10**(-12), activation='relu'):
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


# def y_pred_array(xxa):
#     tempy = []
#     dx = xxa[1]- xxa[0]
#     for i in range(len(xxa)):
#         tx_low = xxa[i] - dx/2
#         tx_high = xxa[i] + dx/2
#         xtemp = np.linspace(tx_low, tx_high, 100)
#         yy = model.predict(xtemp)
#         integral = simps(yy[:,0],xtemp)
#         tempy.append(integral)
#     return np.array(tempy)


# def custom_loss(y_true,y_pred):
#     sqrd_diff = tf.square(y_true - y_pred)
#     return tf.reduce_mean(sqrd_diff, axis=-1)


# def custom_loss(y_true,y_pred):
#     tempy = []
#     dx = xAvg[1]- xAvg[0]
#     for i in range(len(xAvg)):
#         tx_low = xAvg[i] - dx/2
#         tx_high = xAvg[i] + dx/2
#         xtemp = np.linspace(tx_low, tx_high, 100)
#         yy = model(xtemp)
#         integral = simps(yy[:,0],xtemp)
#         tempy.append(integral)
#     y_pred = tf.convert_to_tensor(tempy)
#     sqrd_diff = tf.square(y_true - y_pred)
#     return tf.reduce_mean(sqrd_diff, axis=-1)


def custom_loss(y_true, y_pred):
    tempy = []
    dx = xAvg[1] - xAvg[0]
    for i in range(len(xAvg)):
        tx_low = xAvg[i] - dx / 2
        tx_high = xAvg[i] + dx / 2
        xtemp = tf.linspace(tx_low, tx_high, 100)
        yy = model(xtemp)
        integral = tf.reduce_sum(yy) * dx / 100  
        tempy.append(integral)
    y_pred = tf.convert_to_tensor(tempy)
    sqrd_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(sqrd_diff, axis=-1)


# def y_pred_array(xxa):
#     tempy = []
#     dx = xxa[1]- xxa[0]
#     for i in range(len(xxa)):
#         tx_low = xxa[i] - dx/2
#         tx_high = xxa[i] + dx/2
#         xtemp = np.array(np.linspace(tx_low, tx_high, 100))
#         yy = model.predict(xtemp)
#         integral = simps(yy[:,0], xtemp)
#         tempy.append(integral)
#     return np.array(tempy)


# # Custom loss function using scipy's simps
# def custom_loss(y_true, y_pred):
#     y_pred = tf.reshape(y_pred, [-1])  # Reshape prediction to match y_true shape
#     y_true = tf.cast(y_true, y_pred.dtype)
    
#     def integrate(y_pred_tensor):
#         integral = tf.py_function(func=lambda y: simps(y, xAvg), inp=[y_pred_tensor], Tout=tf.float32)
#         return integral

#     integral_pred = tf.map_fn(integrate, y_pred)
#     integral_true = tf.map_fn(integrate, y_true)

#     return tf.reduce_mean(tf.square(integral_true - integral_pred))


# def simpsons_rule(y, dx):
#     """
#     Compute the integral of y using Simpson's rule.

#     Parameters:
#     y: TensorFlow tensor
#         Array of function values to integrate.
#     dx: float
#         The spacing between consecutive points in the array.

#     Returns:
#     integral: TensorFlow tensor
#         Approximation of the integral of y using Simpson's rule.
#     """

#     n = tf.shape(y)[0]  # Number of points
#     if n % 2 == 0:
#         raise ValueError("Number of points must be odd for Simpson's rule.")

#     # Simpson's rule formula
#     integral = (dx / 3) * (
#         y[0] + 4 * tf.reduce_sum(y[1:-1:2]) + 2 * tf.reduce_sum(y[2:-2:2]) + y[-1]
#     )

#     return integral

# # Example usage
# # Replace y_values and dx with your actual data
# y_values = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
# dx = 0.1  # Example spacing between points

# integral = simpsons_rule(y_values, dx)


# def simpsons_rule(y, dx):
#     n = tf.shape(y)[0].numpy()  # Number of points
#     #print(n)
#     if n % 2 == 0:
#         raise ValueError("test test test")

#     # Simpson's rule formula
#     integral = (dx / 3) * (
#         y[0] + 4 * tf.reduce_sum(y[1:-1:2]) + 2 * tf.reduce_sum(y[2:-2:2]) + y[-1]
#     )

#     return integral



# def custom_loss(y_true,y_pred):
#     tempy = []
#     dx = xAvg[1]- xAvg[0]
#     for i in range(len(xAvg)):
#         tx_low = xAvg[i] - dx/2
#         tx_high = xAvg[i] + dx/2
#         xtemp = tf.constant(np.linspace(tx_low, tx_high, 99))
#         #print(len(xtemp))
#         yy = model(xtemp)
#         #integral = simps(yy[:,0],xtemp)
#         integral = tf.constant(simpsons_rule(yy,dx))
#         tempy.append(integral)
#     y_pred = tf.convert_to_tensor(tempy)
#     sqrd_diff = tf.square(y_true - y_pred)
#     return tf.reduce_mean(sqrd_diff, axis=-1)


# print(custom_loss(Avals,y_pred_array(xAvg)))
#print(custom_loss(Avals,Avals))

lr = 0.005
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=custom_loss)

history = model.fit(xAvg[:-1], Avals[:-1], epochs=100, verbose=2)


# # Assuming you have some training data X_train, y_train
# # Train the model
# history = model.fit(xAvg[:-1], Avals[:-1], epochs=100, verbose=0)

# # Plotting the training loss
# plt.plot(history.history['loss'])
# plt.title('Model Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()



# Generate y values for the true function f(x)
true_y_values = f(Xvals)

# Generate y values for the model predictions
predicted_y_values = model.predict(Xvals)

# Plot f(x) vs model predictions
plt.plot(Xvals, true_y_values, label='True Function f(x)', color='blue')
plt.plot(Xvals, predicted_y_values, label='Model Predictions', color='red')
plt.title('True Function vs Model Predictions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
