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

def DNN_model(width=100, L1_reg=10**(-12), activation='relu'):
    inp = tf.keras.Input(shape=(2,))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
    fx1 = tf.keras.layers.Dense(1, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg), name='fx1')(x2)
    fx2 = tf.keras.layers.Dense(1, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg), name='fx2')(x2)
    product = tf.keras.layers.Multiply()([fx1,fx2])
    mod = tf.keras.Model(inputs=inp, outputs=product)
    return mod

# Concatenate input tensors
concatenated_inputs = np.column_stack((x1Vals, x2Vals))

# Create the DNN model
model = DNN_model()


# def custom_loss(y_true, y_pred):
#     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
#     return mse_loss

# def custom_loss(y_true, y_pred):
#     f1x1 ,f2x2 = y_pred[0], y_pred[1]
#     f1x2, f2x1 = tf.split(tf.reverse(y_pred, axis=[0]), num_or_size_splits=2, axis=0)
#     loss =  tf.reduce_mean(f1x1*f2x2 - f1x2*f2x1)
#     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
#     return mse_loss + loss

# def custom_loss(y_true, y_pred):
#     fx1_output = model.get_layer('fx1').output
#     fx2_output = model.get_layer('fx2').output
#     product_pred = tf.reduce_prod(tf.concat([fx1_output, fx2_output], axis=1), axis=1)
#     product_true = tf.reduce_prod(y_true, axis=1)
#     loss = tf.reduce_mean(tf.abs(product_pred - product_true))
#     return loss

def custom_loss(y_true, y_pred):
    fx1_output = model.get_layer('fx1').output
    fx2_output = model.get_layer('fx2').output
    product_1 = tf.reduce_prod(tf.concat([fx1_output, fx2_output], axis=1), axis=1)
    product_2 = tf.reduce_prod(tf.concat([fx2_output, fx1_output], axis=1), axis=1)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = tf.reduce_mean(product_1 - product_2) + mse_loss
    return loss



lr = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
# model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=custom_loss)

history = model.fit(concatenated_inputs, Avals, epochs=200, verbose=2)

# Plotting the training loss
plt.figure(1,figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('Loss.pdf')


#from mpl_toolkits.mplot3d import Axes3D

# Generate predictions using the trained model
predictions = model.predict(concatenated_inputs)

#print(predictions)

# 3D scatter plot
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1Vals, x2Vals, Avals, c='r', marker='o', label='Actual')
# Plot the model predictions
ax.scatter(x1Vals, x2Vals, predictions, c='b', marker='^', label='Predicted')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('A')
ax.set_title('Actual vs Predicted')
ax.legend()
plt.show()
plt.savefig('Actual_vs_Predicted_Product')


#tf.keras.Model(inputs=LayerF.input, outputs=LayerF.get_layer('cff_output_layer').output)

# fx1result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx1').output)
# #fx1result = fx1result_model()(concatenated_inputs)
# print(np.array(fx1result_model))

fx1result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx1').output)
fx2result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx2').output)

fx1_result = fx1result_model.predict(concatenated_inputs)
fx2_result = fx2result_model.predict(concatenated_inputs)
fx1_true = np.array(f(x1Vals))

fig = plt.figure(3)
plt.plot(x1Vals, fx1_result,'.', label='fx1 output', c='r')
plt.plot(x1Vals, fx2_result,'.', label='fx2 output', c='b')
plt.xlabel('x1')
plt.ylabel('fx1')
plt.title('fx1 Output')
plt.legend()
plt.show()
plt.savefig('fx1Comparison.pdf')



# # Plotting the training loss
# plt.figure(3,figsize=(10, 6))
# plt.plot(x1Vals, predictions,'*', c='r')
# plt.plot(x1Vals, Avals,'*', c='b')
# plt.title('A vs x1')
# plt.xlabel('x1')
# plt.ylabel('A')
# plt.show()
# plt.savefig('Avsx1.pdf')

# # Plotting the training loss
# plt.figure(4,figsize=(10, 6))
# plt.plot(x2Vals, predictions,'*', c='r')
# plt.plot(x2Vals, Avals,'*', c='b')
# plt.title('A vs x2')
# plt.xlabel('x2')
# plt.ylabel('A')
# plt.show()
# plt.savefig('Avsx2.pdf')