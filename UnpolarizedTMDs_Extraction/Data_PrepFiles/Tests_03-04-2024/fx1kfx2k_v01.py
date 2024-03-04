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

def fk(k):
    return 2*k**2/(k**2 + 4)

def fx1kx2k(x1,x2,k):
    return f(x1)*fk(k)*f(x2)*fk(k)

x1vals = np.linspace(0.0001, 0.3, 10)
x2vals = np.linspace(0.1, 0.7, 10)
kvals = np.linspace(0.0001,2,50)

def Apseudo(x1,x2,kk):
    tempx1, tempx2, tempk, tempA = [], [], [], []
              
    for i in range(len(x1)):
        for j in range(len(x2)):
            for k in range(len(kk)): 
                tempx1.append(x1[i])
                tempx2.append(x2[j])
                tempk.append(kk[k])
                tempfx1kfx2k = fx1kx2k(x1[i],x2[j],kk[k])
                tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(tempk), np.array(tempA)

x1Vals, x2Vals, kVals, Avals = Apseudo(x1vals,x2vals,kvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'k': kVals, 'A': Avals})
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

# Create the cross-section model
model = SigmaModel()

lr = 0.00005
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
model.summary()
history = model.fit([x1Vals,x2Vals,kVals], Avals, epochs=300, verbose=2)

# Fit the model and save the history for plotting loss
history = model.fit([x1Vals, x2Vals, kVals], Avals, epochs=300, verbose=2)

# Plot loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Generate predictions
predictions = model.predict([x1Vals, x2Vals, kVals])

# Comparison plot for Avals and predictions
plt.plot(Avals, label='Actual Avals')
plt.plot(predictions, label='Predicted Avals')
plt.title('Comparison between Actual and Predicted Avals')
plt.xlabel('Data Point')
plt.ylabel('Avals')
plt.legend()
plt.show()



#import tensorflow as tf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, Model
# from scipy.integrate import simps
# from tensorflow.keras.callbacks import Callback

# ############ Generating Pseudodata #################

# def f(x):
#     return (x**0.1)*((1-x)**0.3)

# def fk(k):
#     return 2*k**2/(k**2 + 4)

# def fx1kx2k(x1,x2,k):
#     return f(x1)*fk(k)*f(x2)*fk(k)

# x1vals = np.linspace(0.0001, 0.3, 100)
# x2vals = np.linspace(0.1, 0.7, 100)
# kvals = np.linspace(0.0001,2,100)

# def Apseudo(x1,x2,kk):
#     tempx1, tempx2, tempk, tempA = [], [], [], []
#     for k in range(len(kk)):           
#         for i in range(len(x1)):
#             for j in range(len(x2)):
#                 tempx1.append(x1[i])
#                 tempx2.append(x2[j])
#                 tempk.append(kk[k])
#                 tempfx1kfx2k = fx1kx2k(x1[i],x2[j],kk[k])
#                 tempA.append(tempfx1kfx2k)
#     return np.array(tempx1), np.array(tempx2), np.array(tempk), np.array(tempA)

# x1Vals, x2Vals, kVals, Avals = Apseudo(x1vals,x2vals,kvals)

# df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'k': kVals, 'A': Avals})
# df.to_csv('x1x2kvsfx1kx2k.csv')

# ############ Fitting to Pseudodata #################

# def DNN_model(width=200, L1_reg=10**(-12), activation='relu'):
#     inp = tf.keras.Input(shape=(2,))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
#     x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     x1 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     x2 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x1)
#     x3 = tf.keras.layers.Dense(width, activation=activation, kernel_initializer=initializer, activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x2)
#     mod = tf.keras.Model(inputs=inp, outputs=x3)
#     return mod


# # Create the cross-section model
# def SigmaModel():
#     x1 = tf.keras.Input(shape=(1,), name='x1')
#     x2 = tf.keras.Input(shape=(1,), name='x2')
#     k = tf.keras.Input(shape=(1,), name='k')
#     fx1k_model = DNN_model()([x1,k])
#     fx2k_model = DNN_model()([x2,k])
#     product = tf.keras.layers.Multiply()([fx1k_model,fx2k_model])
#     mod = tf.keras.Model(inputs=[x1,x2,k], outputs=product)
#     return mod

# # Create the cross-section model
# model = SigmaModel()


# lr = 0.00005
# model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
# # model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=custom_loss)

# model.summary()

#concatenated_inputs = np.column_stack((x1Vals,x2Vals,kVals))

# history = model.fit(concatenated_inputs, Avals, epochs=200, verbose=2,callbacks=[CustomLossVerboseCallback()])
#history = model.fit([x1Vals,x2Vals,kVals], Avals, epochs=300, verbose=2)

# # Plotting the training loss
# plt.figure(1,figsize=(10, 6))
# plt.plot(history.history['loss'])
# plt.title('Model Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# #plt.show()
# plt.savefig('Loss.pdf')


# #from mpl_toolkits.mplot3d import Axes3D

# # Generate predictions using the trained model
# predictions = model.predict(concatenated_inputs)
# predictions_rev = model.predict(concatenated_inputs_rev)

# #print(predictions)

# # 3D scatter plot
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1Vals, x2Vals, Avals, c='r', marker='o', label='Actual')
# # Plot the model predictions
# ax.scatter(x1Vals, x2Vals, predictions, c='b', marker='^', label='Predicted')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('A')
# ax.set_title('Actual vs Predicted')
# ax.legend()
# plt.show()
# #plt.savefig('Actual_vs_Predicted_Product')


# fx1result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx1').output)
# fx2result_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fx2').output)

# # fx1_result = fx1result_model.predict(concatenated_inputs)
# # fx2_result = fx2result_model.predict(concatenated_inputs)
# # fx1_true = np.array(f(x1Vals))

# predictions_diff = np.abs(predictions - predictions_rev)

# fx1_result = fx1result_model.predict(concatenated_inputs)
# fx2_result = fx2result_model.predict(concatenated_inputs)
# fx1rev_result = fx1result_model.predict(concatenated_inputs_rev)
# fx2rev_result = fx2result_model.predict(concatenated_inputs_rev)

# fx1_fx2_diff =  np.abs(fx1_result - fx2_result)
# fx1_fx2_invrt_diff =  np.abs(fx1rev_result - fx2rev_result)

# fig = plt.figure(3)
# plt.hist(predictions_diff, label='Atrue - Apred',bins=100)
# plt.title('Atrue - Apred')
# plt.legend()
# plt.show()


# fig = plt.figure(3)
# plt.hist(fx1_fx2_diff, label='f1(x1x2)_f2(x2x1)',bins=100)
# plt.title('f(x1x2)_f(x2x1)')
# plt.legend()
# plt.show()

# fig = plt.figure(3)
# plt.hist(fx1_fx2_invrt_diff, label='f1(x2x1)_f2(x1x2)',bins=100)
# plt.title('f1(x2x1)_f2(x1x2)')
# plt.legend()
# plt.show()

# #plt.savefig('f(x1x2)_f(x2x1).pdf')



# # # Plotting the training loss
# # plt.figure(3,figsize=(10, 6))
# # plt.plot(x1Vals, predictions,'*', c='r')
# # plt.plot(x1Vals, Avals,'*', c='b')
# # plt.title('A vs x1')
# # plt.xlabel('x1')
# # plt.ylabel('A')
# # plt.show()
# # plt.savefig('Avsx1.pdf')

# # # Plotting the training loss
# # plt.figure(4,figsize=(10, 6))
# # plt.plot(x2Vals, predictions,'*', c='r')
# # plt.plot(x2Vals, Avals,'*', c='b')
# # plt.title('A vs x2')
# # plt.xlabel('x2')
# # plt.ylabel('A')
# # plt.show()
# # plt.savefig('Avsx2.pdf')