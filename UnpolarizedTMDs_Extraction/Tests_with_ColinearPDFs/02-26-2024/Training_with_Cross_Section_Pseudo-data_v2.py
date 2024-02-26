import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.python.ops import array_ops
from scipy.integrate import simps

############ Generating Pseudodata #################

def S(k):
    return 4*k/(k**2 + 4)

def fxSk(fx,k):
    return fx*S(k)

def fx1Skfx2SkpT(fx1,fx2,k,pT):
    return fx1*S(k)*fx2*S(np.abs(pT-k))

x1 = np.array([0.014,0.016,0.019,0.021])
x2 = np.array([0.013,0.015,0.018,0.020])
pT = np.array([0.1,0.7,1.5,2.5])
#pT = np.array([1,2,3,4])
fux1 = np.array([0.373,0.384,0.399,0.409])
fubarx2 = np.array([0.367,0.378,0.394,0.404])

Kvals = np.linspace(0, 1, 100)

def Sigma(x1,x2,pT,kk, fux1,fubarx2):
    tempk1, tempk2, tempkavg, tempA = [], [], [], []
    tempx1, tempx2, temppT = [], [], []
    for j in range(len(pT)):
        for i in range(len(kk)-1):
            tempk1.append(kk[i])
            tempk2.append(kk[i+1])
            tempkavg.append(0.5*(kk[i]+kk[i+1]))
            tempy = simps(fx1Skfx2SkpT(fux1[j],fubarx2[j],np.linspace(kk[i], kk[i+1], 50),pT[j]), dx=(kk[i+1]-kk[i])/50)
            tempfxSk = tempy
            tempA.append(tempfxSk)
            tempx1.append(x1[j])
            tempx2.append(x2[j])
            temppT.append(pT[j])
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempk1), np.array(tempk2), np.array(tempkavg), np.array(tempA)

x1vals, x2vals, pTvals, k1vals, k2vals, kAvg, Avals = Sigma(x1,x2,pT,Kvals,fux1,fubarx2)

# df = pd.DataFrame({'x1': x1vals, 'x2': x2vals, 'pT':pTvals, 'k1': k1vals, 'k2': k2vals, 'k': kAvg, 'Sigma': Avals})
df = pd.DataFrame({'x1': x1vals, 'x2': x2vals, 'pT':pTvals, 'k': kAvg, 'Sigma': Avals})

############ Fitting to Pseudodata #################

print(df)


def DNN_model(input_shape, width=150, L1_reg=10**(-12), activation='relu'):
    inp = tf.keras.Input(shape=input_shape)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=42)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    nnout = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
    mod = tf.keras.Model(inp, nnout)
    return mod


def SigmaModel():
    x1 = tf.keras.Input(shape=(1,), name='x1')
    x2 = tf.keras.Input(shape=(1,), name='x2')
    pT = tf.keras.Input(shape=(1,), name='pT')
    k = tf.keras.Input(shape=(1,), name='k')

    pTmk = tf.keras.layers.Subtract()([pT,k])

    fx1Sk_model = DNN_model((2,), width=150)(tf.keras.layers.concatenate([x1,k]))
    fx2SkpT_model = DNN_model((2,), width=150)(tf.keras.layers.concatenate([x2,pTmk]))

    tempSigma = tf.keras.layers.Multiply()([fx1Sk_model,fx2SkpT_model])
    return tf.keras.Model([x1,x2,pT,k],tempSigma)


model = SigmaModel()

lr = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')

#model.summary()

history = model.fit([x1vals, x2vals, pTvals, kAvg], Avals, epochs=200, verbose=2)
#history = model.fit(kAvg, Avals, epochs=200, verbose=2)

# Plotting the training loss
plt.figure(1,figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Generate predictions using the trained model
predicted_Avals = model.predict([x1vals, x2vals, pTvals, kAvg])

# Plotting the comparison between actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(pTvals, Avals,'.', label='Actual A', color='blue')
plt.plot(pTvals, predicted_Avals,'.', label='Predicted A', color='red')
plt.title('Actual vs Predicted A')
plt.xlabel('pTvals')
plt.ylabel('A')
plt.legend()
plt.show()
