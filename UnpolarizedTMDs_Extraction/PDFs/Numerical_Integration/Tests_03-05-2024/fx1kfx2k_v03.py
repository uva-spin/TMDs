import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
from scipy.integrate import simps


############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

def Sk(k):
    return 2*k**2/(k**2 + 4)

def fx1kx2k(x1,x2,pT,k):
    return f(x1)*Sk(k)*f(x2)*Sk(pT-k)

x1vals = np.linspace(0.0001, 0.3, 10)
x2vals = np.linspace(0.1, 0.7, 10)
pTvals = np.linspace(0.1,2,10)

def Apseudo(x1,x2,pT):
    tempx1, tempx2, temppT, tempA = [], [], [], []
    kk = np.linspace(0.0001,2,50)
    for i in range(len(x1)):
        for j in range(len(x2)):
            tempx1.append(x1[i])
            tempx2.append(x2[j])
            temppT.append(pT[j])
            tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[j],pT[j],kk), dx=(kk[1]-kk[0]))
            tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

x1Vals, x2Vals, pTVals, Avals = Apseudo(x1vals,x2vals,pTvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'A': Avals})
#df

Hidden_Layers=5
Nodes_per_HL=60
Learning_Rate = 0.001
L1_reg = 10**(-12)
EPOCHS = 5

### Here we create models for each quark-flavor inputs are x and k ###
# def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
#     inp = tf.keras.Input(shape=(2))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
#     x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
#     for i in range(hidden_layers-1):
#         x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
#     nnout = tf.keras.layers.Dense(1, kernel_initializer = initializer)(x)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod
# def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
#     inp = tf.keras.Input(shape=(2))
#     x = tf.keras.layers.Dense(width, activation=activation)(inp)
#     for i in range(hidden_layers-1):
#         x = tf.keras.layers.Dense(width, activation=activation)(x)
#     nnout = tf.keras.layers.Dense(1, activation=activation)(x)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod
def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu6'):
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    inp = tf.keras.Input(shape=(2))
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer)(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer)(x)
    nnout = tf.keras.layers.Dense(1, activation=activation)(x)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')

    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    k_values = tf.linspace(0.0, 2.0, 50)  # Generate k values
    dk = k_values[1] - k_values[0]

    tmd_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val])

        nnu_x1 = tf.abs(modnnu(nnu_input))
        nnubar_x2 = tf.abs(modnnubar(nnubar_input))

        tmd_list.append(dk*tf.keras.layers.Multiply()([nnu_x1, nnubar_x2]))

    # Summing over all k values
    tmd_sum = tf.keras.layers.Add()(tmd_list)

    return tf.keras.Model([x1, x2, qT], tmd_sum)

TMD_Model_DY = createModel_DY()

# Define loss function and optimizer
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model = createModel_DY()
model.compile(optimizer=optimizer, loss=loss_function)

# Train the model on the entire dataset
history = model.fit([df['x1'],df['x2'],df['pT']], df['A'], epochs=300, batch_size=32, verbose=2)


# Retrieve the nnu model from the trained model
modnnu = model.get_layer('nnubar')

# Generate x1 values
x1_values = np.array(np.linspace(0.0001, 0.3, 100))
kk_values = np.array(np.linspace(0.0001, 2, 100))

# Calculate the true values using the function f(x1)*Sk(k)
def f(x):
    return (x**0.1) * ((1 - x)**0.3)

def Sk(k):
    return 2 * k**2 / (k**2 + 4)

true_values = f(x1_values) * Sk(kk_values)  


#tf.keras.layers.Concatenate()([x1, qT*0 + k_val])
concatenated_inputs = np.column_stack((x1_values,kk_values))
predicted_values = modnnu.predict(concatenated_inputs)

# 5. Plot the true and predicted nnu values for comparison
plt.figure(figsize=(10, 6))
plt.plot(x1_values, true_values, label='True nnu', linestyle='--')
plt.plot(x1_values, predicted_values, label='Predicted nnu')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x1')
plt.ylabel('nnu')
plt.legend()
plt.grid(True)
plt.savefig('True_Pred_fxk_q_v0.pdf')

# predicted_A = model.predict([df['x1'],df['x2'],df['pT']])
# true_A = df['A']

# print(true_A,predicted_A)