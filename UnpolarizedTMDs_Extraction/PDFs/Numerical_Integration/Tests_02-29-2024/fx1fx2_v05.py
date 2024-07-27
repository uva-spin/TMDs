import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from scipy.integrate import simps

############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

def fx1x2(x1,x2):
    return f(x1)*f(x2)

x1vals = np.linspace(0.0001, 0.4, 100)
x2vals = np.linspace(0.5, 1, 100)

def Apseudo(x1,x2):
    tempx1, tempx2, tempA = [], [], []
    for i in range(len(x1)):
        for j in range(len(x2)):
            tempx1.append(x1[i])
            tempx2.append(x2[j])
            tempfx1fx2 = fx1x2(x1[i],x2[j])
            tempA.append(tempfx1fx2)
    return np.array(tempx1), np.array(tempx2), np.array(tempA)

x1Vals, x2Vals, Avals = Apseudo(x1vals,x2vals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'A': Avals})
df.to_csv('x1x2vsfx1x2.csv')

############ Fitting to Pseudodata #################

input1 = tf.keras.Input(shape=(1,), name='input_1')
input2 = tf.keras.Input(shape=(1,), name='input_2')

shared_dnn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu')
], name='shared_dnn')

# Process inputs
processed_input1 = shared_dnn(input1)
processed_input2 = shared_dnn(input2)

# Outputs
output1 = tf.keras.layers.Dense(1, name='output_1')(processed_input1)
output2 = tf.keras.layers.Dense(1, name='output_2')(processed_input2)

model = tf.keras.Model(inputs=[input1,input2], outputs=[output1, output2])

# Custom training
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

@tf.function
def train_step(x1,x2):
    with tf.GradientTape() as tape:
        f1_x1, f2_x2 = model([x1,x2], training=True)
        f1_x2, f2_x1 = model([x2, x1], training=True)
        loss = tf.reduce_mean(f1_x1 + f2_x2 - (f1_x2 + f2_x1))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return loss


# Training
for epoch in range(10):
    loss = train_step(x1Vals,x2Vals)
    print(f'Epoch {epoch + 1}, Loss {loss.numpy()}')


