######################################################################
## Here we have A(x1,x2,pT) = f(x1)*Sk(k)*f(x2)*Sk(pT-k)
## k-integration:  was done within the function ##########
## Loss function: three types of losses are defined   ##
## (1) MSE
## (2) f(x,k) --> f(x) when k -> 0
## (3) f(x)S(k) = DNN(x,k)
## Here is an example with simple fucntions for f(x) and S(k)
##  f(x) = fu(x) and fubar(x) from NNPDF40
##  S(k) = k + 1
############################

import tensorflow as tf
import pandas as pd
import numpy as np
#import lhapdf
import matplotlib.pyplot as plt
from scipy.integrate import simps


np.random.seed(42)  # Seed for reproducibility


############ Generating Pseudodata #################



lhapdf_df = pd.read_csv('NNPDF4_nlo.csv').dropna(axis=0, how='all').dropna(axis=1, how='all')



def Skq(k):
    return np.exp(-4*k**2/(4*k**2 + 4))

def Skqbar(k):
    return np.exp(-4*k**2/(4*k**2 + 1))


## Trial 1: Tried with np.exp(-k**2/4) for both
## Tiral 2: np.exp(-k**2/4) for quarks and np.exp(-4*k**2/(4*k**2 + 4)) for anti-quarks
## Trial 3: np.exp(-k**2/4) for quarks and np.exp(-4*k**2/(4*k**2 + 1)) for anti-quarks

# def Sk(k):
#     return 1


fu = np.array(lhapdf_df['fu'])
fubar = np.array(lhapdf_df['fubar'])

# fu = np.array(np.zeros_like(fu)+1)
# fubar = np.array(np.zeros_like(fubar)+1)


def f_map(x,f_array):
    mapping = dict(zip(x,f_array))
    return mapping


#x1vals = np.linspace(0.1, 0.3, 10)
x1vals = np.array(lhapdf_df['x'])
x2vals = np.array(lhapdf_df['x'])
pTvals = np.linspace(0.1,4,len(x1vals))
Kvals = np.linspace(0.1,2,len(x1vals))
pT_k_vals = pTvals - Kvals
kk_values_loss = np.array(np.linspace(0,0,len(x1vals)))


def fu_val(x):
    map = f_map(x1vals,fu)
    return map.get(x,None)

def fubar_val(x):
    map = f_map(x2vals,fubar)
    return map.get(x,None)

#print(fubar_val(x1vals[0]))

def fx1kx2k(x1,x2,pT,k):
    return fu_val(x1)*Skq(k)*fubar_val(x2)*Skqbar(pT-k)


def Apseudo(x1,x2,pT):
    tempx1, tempx2, temppT, tempA = [], [], [], []
    kk = np.linspace(0.0,2.0,len(x1vals))
    for i in range(len(x1)):
        tempx1.append(x1[i])
        tempx2.append(x2[i])
        temppT.append(pT[i])
        tempfx1kfx2k = simps(fx1kx2k(x1[i],x2[i],pT[i],kk), dx=(kk[1]-kk[0]))
        tempA.append(tempfx1kfx2k)
    return np.array(tempx1), np.array(tempx2), np.array(temppT), np.array(tempA)

x1Vals, x2Vals, pTVals, Avals = Apseudo(x1vals,x2vals,pTvals)

df = pd.DataFrame({'x1': x1Vals, 'x2': x2Vals, 'pT': pTVals, 'A': Avals})
#df



################ Defining the DNN model ####################
Hidden_Layers=7
Nodes_per_HL=500
Learning_Rate = 0.0001
L1_reg = 10**(-12)
EPOCHS = 500


# def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation='relu'):
#     inp = tf.keras.Input(shape=(2))
#     x = tf.keras.layers.Dense(width, activation=activation)(inp)
#     for i in range(hidden_layers-1):
#         x = tf.keras.layers.Dense(width, activation=activation)(x)
#     nnout = tf.keras.layers.Dense(1, activation=activation)(x)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod


def create_nn_model(name):
    inp = tf.keras.Input(shape=(2))
    x = tf.keras.layers.Dense(224, activation='tanh')(inp)
    x1 = tf.keras.layers.Dense(384, activation='tanh')(x)
    x2 = tf.keras.layers.Dense(160, activation='tanh')(x1)
    x3 = tf.keras.layers.Dense(32, activation='tanh')(x2)
    x4 = tf.keras.layers.Dense(512, activation='relu')(x3)
    nnout = tf.keras.layers.Dense(1, activation='relu')(x4)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

# def create_nn_model(name):
#     inp = tf.keras.Input(shape=(2))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.03,maxval=0.03,seed=42)
#     x = tf.keras.layers.Dense(224, activation='relu6', kernel_initializer = initializer)(inp)
#     x1 = tf.keras.layers.Dense(384, activation='relu6', kernel_initializer = initializer)(x)
#     x2 = tf.keras.layers.Dense(160, activation='relu6', kernel_initializer = initializer)(x1)
#     x3 = tf.keras.layers.Dense(32, activation='relu6', kernel_initializer = initializer)(x2)
#     x4 = tf.keras.layers.Dense(512, activation='relu6', kernel_initializer = initializer)(x3)
#     nnout = tf.keras.layers.Dense(1)(x4)
#     mod = tf.keras.Model(inp, nnout, name=name)
#     return mod


# def create_nn_model(name):
#     inp = tf.keras.Input(shape=(2))
#     initializer = tf.keras.initializers.RandomUniform(minval=-0.02,maxval=0.02,seed=42)
#     x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer = initializer)(inp)
#     x1 = tf.keras.layers.Dense(352, activation='relu', kernel_initializer = initializer)(x)
#     x2 = tf.keras.layers.Dense(192, activation='tanh', kernel_initializer = initializer)(x1)
#     x3 = tf.keras.layers.Dense(448, activation='relu', kernel_initializer = initializer)(x2)
#     #nnout = tf.keras.layers.Dense(1, activation='relu', kernel_initializer = initializer)(x3)
#     mod = tf.keras.Model(inp, x3, name=name)
#     return mod


def createModel_DY():
    x1 = tf.keras.Input(shape=(1), name='x1')
    x2 = tf.keras.Input(shape=(1), name='x2')
    qT = tf.keras.Input(shape=(1), name='qT')


    modnnu = create_nn_model('nnu')
    modnnubar = create_nn_model('nnubar')

    pdf_k_val = 0    
    nnu_pdf_input = tf.keras.layers.Concatenate()([x1, qT*0 + pdf_k_val])
    nnubar_pdf_input = tf.keras.layers.Concatenate()([x2, qT*0 - pdf_k_val])

    modnnu_pdf_eval = modnnu(nnu_pdf_input)
    modnnubar_pdf_eval = modnnubar(nnubar_pdf_input)

    k_values = tf.linspace(0.1, 2.0, 100)
    dk = k_values[1] - k_values[0]

    tmd1_list, tmd2_list = [], []
    product_list = []  # List to store TMD values for each k value
    for k_val in k_values:      
        nnu_input = tf.keras.layers.Concatenate()([x1, qT*0 + k_val])
        nnubar_input = tf.keras.layers.Concatenate()([x2, qT - k_val])

        nnu_x1 = modnnu(nnu_input)
        nnubar_x2 = modnnubar(nnubar_input)

        nnu_x1_rev = modnnu(nnubar_input)
        nnubar_x2_rev = modnnubar(nnu_input)

        product_1 = tf.multiply(nnu_x1, nnubar_x2)
        product_2 = tf.multiply(nnu_x1_rev, nnubar_x2_rev)

        result = tf.add(product_1,product_2)
        product_list.append(result)

    # Summing over all k values using tf.reduce_sum
    tmd_product_sum = tf.reduce_sum(product_list, axis=0) * dk 

    return tf.keras.Model([x1, x2, qT], [tmd_product_sum, modnnu_pdf_eval, modnnubar_pdf_eval])

model = createModel_DY()

def mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


def custom_loss(y_true, y_pred):
    return mse_loss(y_true, y_pred) 


plt.figure(1, figsize=(10, 6))
plt.plot(x1vals, fu, label='$f_u(x)$')
plt.plot(x2vals, fubar, label='$f_{\\bar{u}}(x)$')
plt.title('NNPDF4 fu and fubar')
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('PDFs.pdf')

model.compile(optimizer='adam', loss=custom_loss)
# history = model.fit([df['x1'],df['x2'],df['pT']], df['A'], epochs=EPOCHS, batch_size=32, verbose=2)
history = model.fit([df['x1'],df['x2'],df['pT']], [df['A'], fu, fubar], epochs=EPOCHS, batch_size=32, verbose=2)


modnnu = model.get_layer('nnu')
modnnubar = model.get_layer('nnubar')


predictions = model.predict([df['x1'],df['x2'],df['pT']])[0]

# 3D scatter plot
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1Vals, pTVals, df['A'], c='r', marker='o', label='Actual')
ax.scatter(x1Vals, pTVals, predictions, c='b', marker='^', label='Predicted')
ax.set_xlabel('x1')
ax.set_ylabel('pT')
ax.set_zlabel('A')
ax.set_title('Actual vs Predicted')
ax.legend()
#plt.show()
plt.savefig('Actual_vs_Predicted_Integral.pdf')


modnnu = model.get_layer('nnu')
modnnubar = model.get_layer('nnubar')

true_values_1 = fu * Skq(Kvals)  
true_values_2 = fubar * Skqbar(pT_k_vals) 

concatenated_inputs_1 = np.column_stack((x1vals,Kvals))
concatenated_inputs_2 = np.column_stack((x2vals,pT_k_vals))

predicted_values_1 = modnnu.predict(concatenated_inputs_1)
predicted_values_2 = modnnubar.predict(concatenated_inputs_2)

# predicted_values_1 = model.predict([df['x1'],df['x2'],df['pT']])[1]
# predicted_values_2 = model.predict([df['x1'],df['x2'],df['pT']])[2]


plt.figure(3, figsize=(10, 6))
plt.plot(x1vals, true_values_1, label='True nnu', linestyle='--')
plt.plot(x1vals, predicted_values_1, label='Predicted nnu')
plt.title('Comparison of True and Predicted nnu Values')
plt.xlabel('x1')
plt.ylabel('nnu')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('True_Pred_fxk_q.pdf')

plt.figure(4, figsize=(10, 6))
plt.plot(x2vals, true_values_2, label='True nnubar', linestyle='--')
plt.plot(x2vals, predicted_values_2, label='Predicted nnubar')
plt.title('Comparison of True and Predicted nnubar Values')
plt.xlabel('x2')
plt.ylabel('nnubar')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('True_Pred_fxk_qbar.pdf')