import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

############ Generating Pseudodata #################

def f(x):
    return 2*x + 1

Xvals = np.linspace(0, 1, 1000)

def simpsons_rule(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    return h/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])

def Apseudo(xx):
    tempx1, tempx2, tempxavg, tempA = [], [], [], []
    for i in range(len(xx)-1):
        tempx1.append(xx[i])
        tempx2.append(xx[i+1])
        tempxavg.append(0.5*(xx[i]+xx[i+1]))
        tempy = simpsons_rule(f, xx[i], xx[i+1], 50)
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

# Define the trapezoidal integral layer
class TrapezoidalIntegralLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(TrapezoidalIntegralLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        
        # Compute dx assuming x is evenly spaced
        dx = x[:, 1:] - x[:, :-1]

        # Compute the integral using the trapezoidal rule
        integral = tf.reduce_sum(0.5 * (y[:, :-1] + y[:, 1:]) * dx, axis=1)

        return integral

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],)

# Define the model containing DNN and trapezoidal integral layer
class Integral_model(Model):
    def __init__(self, model, **kwargs):
        super(Integral_model, self).__init__(**kwargs)
        self.dnn_model = model
        self.integral_layer = TrapezoidalIntegralLayer()

    def call(self, inputs):
        x = inputs
        dnn_output = self.dnn_model(x)
        integral_output = self.integral_layer([x, dnn_output])
        return integral_output

    def get_dnn_predictions(self, x):
        return self.dnn_model(x)

# Example DataFrame df with columns 'x' and 'A'
x_train = df['x'].values.reshape(-1, 1)
y_train = df['A'].values.reshape(-1, 1)

# Create the DNN model
dnn_model = DNN_model()

# Create the Integral model
amodel = Integral_model(dnn_model)

# Print model summary
amodel.compile(optimizer='adam', loss='mse')
amodel.fit(x_train, y_train, epochs=20, batch_size=5)

# Generate predictions from the DNN model
dnn_predictions = amodel.get_dnn_predictions(x_train)

# Plotting
plt.figure(1, figsize=(10, 6))
plt.scatter(x_train, dnn_predictions, label='DNN Model', color='blue')
plt.plot(x_train, Avals, label='f(x) = 2x + 1', color='red')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('Comparison between DNN Model and f(x)')
plt.legend()
plt.grid(True)
plt.show()


# Generate predictions from the Integral model
integral_predictions = amodel.predict(x_train)

# Plotting
plt.figure(2, figsize=(10, 6))

# Plotting the output of the Integral model vs Avals (true integral)
plt.plot(x_train, integral_predictions, label='Integral Model Output', color='blue')
plt.plot(x_train, Avals, label='True Integral (Avals)', color='red')

plt.xlabel('x')
plt.ylabel('Output')
plt.title('Comparison between Integral Model Output and True Integral')
plt.legend()
plt.grid(True)
plt.show()