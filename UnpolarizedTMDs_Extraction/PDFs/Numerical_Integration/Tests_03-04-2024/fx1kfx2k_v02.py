import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from scipy.integrate import simps
from tensorflow.keras.callbacks import Callback
from scipy.integrate import simps

############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

def fk(k):
    return 2*k**2/(k**2 + 4)

def fx1kx2k(x1,x2,k):
    return f(x1)*fk(k)*f(x2)*fk(k)

x1vals = np.linspace(0.0001, 0.3, 10)
x2vals = np.linspace(0.1, 0.7, 10)

def Apseudo(x1,x2):
    tempx1, tempx2, tempk, tempA = [], [], [], []
    kk = np.linspace(0.0001,2,50)
    for i in range(len(x1)):
        for j in range(len(x2)):
            tempx1.append(x1[i])
            tempx2.append(x2[j])
            tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[j],kk), dx=(kk[1]-kk[0]))
            tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(tempA)

x1Vals, x2Vals, Avals = Apseudo(x1vals,x2vals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'A': Avals})
df.to_csv('x1x2kvsfx1kx2k.csv')

############ Fitting to Pseudodata #################

def DNN_model(width=200, L1_reg=10**(-12), activation='relu'):
    inp1 = tf.keras.Input(shape=(1,))
    inp2 = tf.keras.Input(shape=(1,))
    
    # Concatenate the inputs
    concat = tf.keras.layers.Concatenate()([inp1, inp2])
    
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(concat)
    mod = tf.keras.Model(inputs=[inp1, inp2], outputs=x)
    return mod

# Create the cross-section model
def SigmaModel():
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    k = tf.keras.Input(shape=(1,), name='k')
    fx1k_model = DNN_model()([x1, k])
    fx2k_model = DNN_model()([x2, k])
    product = tf.keras.layers.Multiply()([fx1k_model, fx2k_model])
    mod = tf.keras.Model(inputs=[x1, x2, k], outputs=product)
    return mod