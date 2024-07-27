import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers


############ Generating Pseudodata #################

def f(x):
    return 2*x + 1

Xvals = np.array(np.linspace(0,1,1000))

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


############ Fitting to Pseudodata #################

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# Define the DNN model
def DNN_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),  
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  
    ])
    return model

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
# def Integral_model(model):
#     x_input = layers.Input(shape=(1,))
#     dnn_output = model(x_input)
#     integral_output = TrapezoidalIntegralLayer()([x_input, dnn_output])
#     model = Model(inputs=x_input, outputs=integral_output)
#     return model

def Integral_model(model):
    x_input = layers.Input(shape=(1,))
    dnn_output = model(x_input)
    integral_output = TrapezoidalIntegralLayer()([x_input, dnn_output])
    
    # Name the dnn_output tensor
    dnn_output = layers.Lambda(lambda x: x, name='dnn_output')(dnn_output)
    
    model = Model(inputs=x_input, outputs=[integral_output, dnn_output])
    return model


# Example DataFrame df with columns 'x' and 'A'
x_train = df['x'].values.reshape(-1, 1)
y_train = df['A'].values.reshape(-1, 1)

# Create the DNN model
#input_shape = (1,)  # Assuming 'x' is a single feature
dnn_model = DNN_model()

# Create the A model
amodel = Integral_model(dnn_model)

# Print model summary
amodel.summary()

amodel.compile(optimizer='adam', loss='mse')
amodel.fit(x_train, y_train, epochs=100, batch_size=5)


# Generate predictions from the DNN model
dnn_predictions = amodel.predict(x_train)

# Generate predictions from the function f(x)
true_predictions = f(x_train)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_train, dnn_predictions, label='DNN Model', color='blue')
plt.plot(x_train, true_predictions, label='f(x) = 2x + 1', color='red')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('Comparison between DNN Model and f(x)')
plt.legend()
plt.grid(True)
plt.show()

