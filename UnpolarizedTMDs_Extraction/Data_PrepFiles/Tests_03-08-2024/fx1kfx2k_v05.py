import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
from scipy.integrate import simps


############ Generating Pseudodata #################

def f(x):
    return (x**0.1)*((1-x)**0.3)

def Sk1(k):
    return 2*k**2/(k**2 + 4)

def Sk2(k):
    return np.exp(-k**2/4)/4/np.pi

def fx1kx2k(x1,x2,pT,k):
    return f(x1)*Sk1(k)*f(x2)*Sk2(pT-k)

x1vals = np.linspace(0.0001, 0.3, 10)
x2vals = np.linspace(0.1, 0.7, 10)
pTvals = np.linspace(0.1,2,5)
kvals = np.linspace(0.0001,2,100)

def Apseudo(x1,x2,pT,kk):
    tempx1, tempx2, temppT, tempk, tempA = [], [], [], [], []
              
    for i in range(len(x1)):
        for j in range(len(x2)):
            for k in range(len(pT)):
                for l in range(len(kk)): 
                    tempx1.append(x1[i])
                    tempx2.append(x2[j])
                    temppT.append(pT[k])
                    tempk.append(kk[l])
                    tempfx1kfx2k = fx1kx2k(x1[i],x2[j],pT[k],kk[l])
                    tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempk), np.array(tempA)

x1Vals, x2Vals, pTVals, kVals, Avals = Apseudo(x1vals,x2vals,pTvals,kvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'k': kVals, 'A': Avals})
df.to_csv('pseudodata.csv')

Hidden_Layers=5
Nodes_per_HL=100
Learning_Rate = 0.0001
L1_reg = 10**(-12)
EPOCHS = 5


def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu6', seed=(42)):
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
    k = tf.keras.Input(shape=(1), name='k')

    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    nnu_input = tf.keras.layers.Concatenate()([x1, k])
    nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k])

    nnu_x1 = tf.abs(modnnu(nnu_input))
    nnubar_x2 = tf.abs(modnnubar(nnubar_input))

    tmd= tf.keras.layers.Multiply()([nnu_x1, nnubar_x2])

    return tf.keras.Model([x1, x2, qT, k], tmd)

TMD_Model_DY = createModel_DY()

# Define loss function and optimizer
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model = createModel_DY()
model.compile(optimizer=optimizer, loss=loss_function)


history = model.fit([df['x1'],df['x2'],df['pT'],df['k']], df['A'], epochs=100, batch_size=32, verbose=2)


# Plotting the training loss
plt.figure(1,figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('loss_plot.pdf')

# Use the trained model to predict A values
predicted_A = model.predict([df['x1'],df['x2'],df['pT'],df['k']])
true_A = df['A']

# Calculate the difference between true A and predicted A
difference_A = np.abs(true_A - predicted_A.flatten())

# Plot 2D density plot
plt.figure(2,figsize=(10, 8))
plt.hist2d(df['x1'], df['pT'], bins=30, weights=difference_A, cmap='coolwarm')
plt.colorbar(label='TrueA - PredictedA')
plt.title('2D Density Plot of TrueA - PredictedA with x1 and pT')
plt.xlabel('x1')
plt.ylabel('pT')
plt.show()
plt.savefig('HeatMap_sigma_true-pred.pdf')


modnnu = model.get_layer('nnu')
modnnubar = model.get_layer('nnubar')

x1_values = np.array(np.linspace(0.0001, 0.3, 100))
kk_values = np.array(np.linspace(0.0001, 2, 100))
x2_values = np.array(np.linspace(0.1, 0.7, 100))
pT_values = np.array(np.linspace(0.1,2,100))
pT_k_values = pT_values - kk_values

true_values_1 = f(x1_values) * Sk1(kk_values)  
true_values_2 = f(x2_values) * Sk2(pT_k_values)  

concatenated_inputs_1 = np.column_stack((x1_values,kk_values))
predicted_values_1 = modnnu.predict(concatenated_inputs_1)

concatenated_inputs_2 = np.column_stack((x2_values,pT_k_values))
predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

plt.figure(3, figsize=(10, 6))
plt.plot(x1_values, true_values_1, label='True nnu', linestyle='--')
plt.plot(x1_values, predicted_values_1, label='Predicted nnu')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x1')
plt.ylabel('nnu')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('True_Pred_fxk_q.pdf')

plt.figure(4, figsize=(10, 6))
plt.plot(x2_values, true_values_2, label='True nnubar', linestyle='--')
plt.plot(x2_values, predicted_values_2, label='Predicted nnubar')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x2')
plt.ylabel('nnubar')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('True_Pred_fxk_qbar.pdf')