import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from scipy.integrate import simps
from tensorflow.keras.callbacks import Callback

############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

def fx1x2(x1,x2):
    return f(x1)*f(x2)

x1vals = np.linspace(0.1, 0.3, 20)
x2vals = np.linspace(0.1, 0.7, 20)

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
    x3 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
    fx1 = tf.keras.layers.Dense(1, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg), name='fx1')(x3)
    fx2 = tf.keras.layers.Dense(1, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg), name='fx2')(x3)
    product = tf.keras.layers.Multiply()([fx1,fx2])
    mod = tf.keras.Model(inputs=inp, outputs=product)
    return mod

# Concatenate input tensors
concatenated_inputs = np.column_stack((x1Vals, x2Vals))
concatenated_inputs_rev = np.column_stack((x2Vals,x1Vals))

# Create the DNN model
model = DNN_model()

def product_loss(y_true, y_pred):
    # Forward pass through the model
    # fx1_output = model.layers[-3](model.layers[-4](model.layers[-5](model.layers[-6](model.layers[-7](model.input)))))
    # fx2_output = model.layers[-2](model.layers[-4](model.layers[-5](model.layers[-6](model.layers[-7](model.input)))))
    fx1_output = model.layers[-3](model.layers[-4](model.layers[-5](model.layers[-6](model.layers[-7](concatenated_inputs)))))
    fx2_output = model.layers[-2](model.layers[-4](model.layers[-5](model.layers[-6](model.layers[-7](concatenated_inputs)))))

    fx1_output_rev = model.layers[-3](model.layers[-4](model.layers[-5](model.layers[-6](model.layers[-7](concatenated_inputs_rev)))))
    fx2_output_rev = model.layers[-2](model.layers[-4](model.layers[-5](model.layers[-6](model.layers[-7](concatenated_inputs_rev)))))
    
    # Calculate product of fx1 and fx2
    product = tf.multiply(fx1_output, fx2_output)
    product_rev = tf.multiply(fx1_output_rev, fx2_output_rev)

    # Calculate mean squared error
    prod_loss = tf.reduce_mean(tf.square(product - product_rev))
    return prod_loss

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss

def custom_loss(y_true, y_pred):
    return mse_loss(y_true, y_pred) + product_loss(y_true, y_pred)


# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss)

lr = 0.00001
#lr = 0.00005
# model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
# model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=custom_loss)
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=custom_loss, metrics=[mse_loss,product_loss])

# history = model.fit(concatenated_inputs, Avals, epochs=200, verbose=2,callbacks=[CustomLossVerboseCallback()])
history = model.fit(concatenated_inputs, Avals, epochs=1000, verbose=2)

# Plotting the training loss
plt.figure(1,figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.show()
plt.savefig('Loss.pdf')


#from mpl_toolkits.mplot3d import Axes3D

# Generate predictions using the trained model
predictions = model.predict(concatenated_inputs)
predictions_rev = model.predict(concatenated_inputs_rev)

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
#plt.show()
plt.savefig('Actual_vs_Predicted_Product')


fx1result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx1').output)
fx2result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx2').output)

# fx1_result = fx1result_model.predict(concatenated_inputs)
# fx2_result = fx2result_model.predict(concatenated_inputs)
# fx1_true = np.array(f(x1Vals))

predictions_diff = np.abs(predictions - predictions_rev)

fx1_result = fx1result_model.predict(concatenated_inputs)
fx2_result = fx2result_model.predict(concatenated_inputs)
fx1rev_result = fx1result_model.predict(concatenated_inputs_rev)
fx2rev_result = fx2result_model.predict(concatenated_inputs_rev)

# fx1_fx2_diff =  np.abs(fx1_result - fx2_result)
# fx1_fx2_invrt_diff =  np.abs(fx1rev_result - fx2rev_result)

fx1_fx2_diff =  np.abs(fx1_result*fx2_result - fx1rev_result*fx2rev_result)

# fig = plt.figure(3)
# plt.hist(predictions_diff, label='Atrue - Apred',bins=100)
# plt.title('Atrue - Apred')
# plt.legend()
# plt.show()


fig = plt.figure(3)
plt.hist(fx1_fx2_diff, label='f1(x1x2)_f2(x2x1)',bins=100, range=(-0.0001, 0.0001))
plt.xticks(ticks=[-0.0001, -0.00005, 0, 0.00005, 0.0001]) 
plt.title('f(x1x2)_f(x2x1)')
plt.legend()
#plt.show()
plt.savefig('f(x1x2)-f(x2x1).pdf')

# fig = plt.figure(3)
# plt.hist(fx1_fx2_invrt_diff, label='f1(x2x1)_f2(x1x2)',bins=100, range=(-0.0001, 0.0001))
# plt.xticks(ticks=[-0.0001, -0.00005, 0, 0.00005, 0.0001]) 
# plt.title('f1(x2x1)_f2(x1x2)')
# plt.legend()
# #plt.show()
# plt.savefig('f(x2x1)-f(x1x2).pdf')

#plt.savefig('f(x1x2)_f(x2x1).pdf')



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